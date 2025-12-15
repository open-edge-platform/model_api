#!/usr/bin/env -S uv run --script
#
# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
PyTorch to OpenVINO Model Converter

Usage:
    uv run python model_converter.py config.json -o ./output_models

"""

import argparse
import importlib
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from typing import Tuple


# =============================================================================
# Export Wrappers for Detection Models
# =============================================================================
# These wrappers convert detection models that return complex dictionary outputs
# into models that return tuple of tensors, suitable for ONNX/OpenVINO export.


class SSDExportWrapper(nn.Module):
    """Wrapper to make SSD-like models exportable to ONNX.

    Torchvision SSD models return List[Dict] with 'boxes', 'scores', 'labels'.
    This wrapper converts to tuple output for ONNX compatibility.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning tuple of (boxes, scores, labels)."""
        # Detection models expect list of tensors
        image_list = [images[i] for i in range(images.shape[0])]
        outputs = self.model(image_list)
        # Return first image's detections (batch size 1 for export)
        return outputs[0]["boxes"], outputs[0]["scores"], outputs[0]["labels"]


class FasterRCNNExportWrapper(nn.Module):
    """Wrapper to make Faster R-CNN models exportable to ONNX.

    Torchvision Faster R-CNN models return List[Dict] with 'boxes', 'scores', 'labels'.
    This wrapper converts to tuple output for ONNX compatibility.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning tuple of (boxes, scores, labels)."""
        image_list = [images[i] for i in range(images.shape[0])]
        outputs = self.model(image_list)
        return outputs[0]["boxes"], outputs[0]["scores"], outputs[0]["labels"]


class RetinaNetExportWrapper(nn.Module):
    """Wrapper to make RetinaNet models exportable to ONNX.

    Torchvision RetinaNet models return List[Dict] with 'boxes', 'scores', 'labels'.
    This wrapper converts to tuple output for ONNX compatibility.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning tuple of (boxes, scores, labels)."""
        image_list = [images[i] for i in range(images.shape[0])]
        outputs = self.model(image_list)
        return outputs[0]["boxes"], outputs[0]["scores"], outputs[0]["labels"]


class MaskRCNNExportWrapper(nn.Module):
    """Wrapper to make Mask R-CNN models exportable to ONNX.

    Torchvision Mask R-CNN models return List[Dict] with 'boxes', 'labels', 'scores', 'masks'.
    This wrapper converts to tuple output for ONNX compatibility.
    
    Output format matches model_api MaskRCNNModel expectations:
    - boxes: [N, 5] where columns are [x1, y1, x2, y2, score]
    - labels: [N] 
    - masks: [N, H, W] - probability masks in [0, 1] range (will be thresholded by postprocessing)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning tuple of (boxes_with_scores, labels, masks)."""
        image_list = [images[i] for i in range(images.shape[0])]
        outputs = self.model(image_list)
        
        boxes = outputs[0]["boxes"]  # [N, 4]
        scores = outputs[0]["scores"]  # [N]
        labels = outputs[0]["labels"]  # [N]
        masks = outputs[0]["masks"]  # [N, 1, H, W] - float probabilities in [0, 1]
        
        # Concatenate scores to boxes: [N, 4] + [N, 1] -> [N, 5]
        boxes_with_scores = torch.cat([boxes, scores.unsqueeze(1)], dim=1)
        
        # Squeeze masks from [N, 1, H, W] to [N, H, W]
        # Keep as float probabilities - postprocessing will threshold at 0.5
        masks = masks.squeeze(1)
        
        return boxes_with_scores, labels, masks


# Mapping of model types to their export wrappers
EXPORT_WRAPPERS = {
    "SSD": SSDExportWrapper,
    "FasterRCNNModel": FasterRCNNExportWrapper,
    "RetinaNet": RetinaNetExportWrapper,
    "MaskRCNN": MaskRCNNExportWrapper,
}

# Default output names for detection models
DETECTION_OUTPUT_NAMES = {
    "SSD": ["boxes", "scores", "labels"],
    "FasterRCNNModel": ["boxes", "scores", "labels"],
    "RetinaNet": ["boxes", "scores", "labels"],
    "MaskRCNN": ["boxes", "labels", "masks"],  # boxes includes scores as 5th column
}


class ModelConverter:
    """Handles conversion of PyTorch models to OpenVINO format."""

    def __init__(
        self,
        output_dir: Path,
        cache_dir: Path,
        verbose: bool = False,
    ):
        """
        Initialize the ModelConverter.

        Args:
            output_dir: Directory to save converted models
            cache_dir: Directory to cache downloaded weights
            verbose: Enable verbose logging
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def get_labels(self, label_set: str) -> Optional[str]:
        """
        Get label list for a given label set.

        Args:
            label_set: Name of the label set (e.g., "IMAGENET1K_V1", "COCO")

        Returns:
            Space-separated string of labels, or None if not found
        """
        if label_set == "IMAGENET1K_V1":
            from torchvision.models._meta import _IMAGENET_CATEGORIES

            categories = _IMAGENET_CATEGORIES
            categories = [label.replace(" ", "_") for label in categories]
            return " ".join(categories)

        if label_set == "COCO":
            # COCO 91 classes (including background at index 0)
            # TorchVision detection models use 1-indexed labels (1-90)
            coco_categories = [
                "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                "train", "truck", "boat", "traffic_light", "fire_hydrant", "N/A", "stop_sign",
                "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
                "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball",
                "kite", "baseball_bat", "baseball_glove", "skateboard", "surfboard", "tennis_racket",
                "bottle", "N/A", "wine_glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                "apple", "sandwich", "orange", "broccoli", "carrot", "hot_dog", "pizza", "donut",
                "cake", "chair", "couch", "potted_plant", "bed", "N/A", "dining_table", "N/A",
                "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell_phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock",
                "vase", "scissors", "teddy_bear", "hair_drier", "toothbrush"
            ]
            return " ".join(coco_categories)

        return None

    def download_weights(
        self,
        url: str,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Download model weights from URL with caching.

        Args:
            url: URL to download weights from
            filename: Optional filename to save as (default: extract from URL)

        Returns:
            Path to the downloaded/cached weights file
        """
        if filename is None:
            filename = url.split("/")[-1]

        cached_file = self.cache_dir / filename

        if cached_file.exists():
            self.logger.info(f"Using cached weights: {cached_file}")
            return cached_file

        self.logger.info(f"Downloading weights from: {url}")
        self.logger.info(f"Saving to: {cached_file}")

        try:
            urllib.request.urlretrieve(  # noqa: S310  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
                url,
                cached_file,
            )
            self.logger.info("✓ Download complete")
            return cached_file
        except Exception as e:
            self.logger.error(f"Failed to download weights: {e}")
            raise

    def load_model_class(
        self,
        class_path: str,
    ) -> type:
        """
        Dynamically load a model class from a Python path.

        Args:
            class_path: Full Python path to the class (e.g., 'torchvision.models.resnet.resnet50')

        Returns:
            The model class
        """
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            self.logger.debug(f"Importing module: {module_path}")
            # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
            module = importlib.import_module(
                module_path,
            )
            model_class = getattr(module, class_name)
            self.logger.debug(f"Loaded class: {class_name}")
            return model_class
        except Exception as e:
            self.logger.error(f"Failed to import module {module_path}: {e}")
            raise

    def load_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> Dict[str, Any]:
        """
        Load PyTorch checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint dictionary
        """
        try:
            checkpoint = torch.load(  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.logger.debug(f"Loaded checkpoint from: {checkpoint_path}")
            return checkpoint
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise

    def create_model(
        self,
        model_class: type,
        checkpoint: Dict[str, Any],
        model_params: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """
        Create and initialize model instance.

        Args:
            model_class: Model class to instantiate
            checkpoint: Checkpoint containing model weights
            model_params: Optional parameters for model initialization

        Returns:
            Initialized model instance
        """
        try:
            # Handle torch.nn.Module case (checkpoint contains full model)
            if model_class == torch.nn.Module:
                if "model" in checkpoint:
                    model = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    # Cannot reconstruct architecture from state_dict alone
                    error_msg = (
                        "Checkpoint contains only state_dict. "
                        "Please specify the model class instead of torch.nn.Module"
                    )
                    raise ValueError(error_msg)
                else:
                    # Assume checkpoint is the model itself
                    model = checkpoint

                if not isinstance(model, nn.Module):
                    error_msg = "Checkpoint does not contain a valid model"
                    raise ValueError(error_msg)
            else:
                # Instantiate model class
                model = model_class(**model_params) if model_params else model_class()

                # Load weights
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    if isinstance(checkpoint["model"], nn.Module):
                        return checkpoint["model"]
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint

                model.load_state_dict(state_dict, strict=False)

            model.eval()
            self.logger.info("✓ Model created and loaded successfully")
            return model

        except Exception as e:
            self.logger.error(f"Failed to create model: {e}")
            raise

    def export_to_openvino(
        self,
        model: nn.Module,
        input_shape: List[int],
        output_path: Path,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        metadata: Optional[Dict[tuple, str]] = None,
        export_method: str = "direct",
        model_type: Optional[str] = None,
        onnx_opset_version: int = 17,
    ) -> Path:
        """
        Export PyTorch model to OpenVINO format.

        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape [batch, channels, height, width]
            output_path: Path to save the model (without extension)
            input_names: Names for input tensors
            output_names: Names for output tensors
            metadata: Metadata to embed in the model
            export_method: Export method - "direct" or "onnx"
            model_type: Model type (used for selecting export wrapper)
            onnx_opset_version: ONNX opset version (default: 17)

        Returns:
            Path to the exported .xml file
        """
        import openvino as ov

        try:
            model.eval()

            # Use ONNX export path for detection models or when explicitly requested
            if export_method == "onnx" or model_type in EXPORT_WRAPPERS:
                self.logger.info(f"Using ONNX export path (model_type: {model_type})")
                onnx_path = self.export_to_onnx(
                    model=model,
                    input_shape=input_shape,
                    output_path=output_path,
                    input_names=input_names,
                    output_names=output_names,
                    model_type=model_type,
                    opset_version=onnx_opset_version,
                )
                self.logger.info(f"Converting ONNX to OpenVINO: {onnx_path}")
                ov_model = ov.convert_model(onnx_path)
                self.logger.info("✓ ONNX to OpenVINO conversion complete")
            else:
                # Direct PyTorch to OpenVINO conversion
                dummy_input = torch.randn(*input_shape)
                self.logger.info("Direct PyTorch to OpenVINO conversion")
                ov_model = ov.convert_model(model, example_input=dummy_input)
                self.logger.info("✓ PyTorch to OpenVINO conversion complete")

                # Reshape model to fixed input shape (remove dynamic dimensions)
                first_input = ov_model.input(0)
                input_name_for_reshape = next(iter(first_input.get_names())) if first_input.get_names() else 0

                self.logger.debug(f"Setting fixed input shape: {input_shape}")
                ov_model.reshape({input_name_for_reshape: input_shape})

            # Post-process the model
            ov_model = self._postprocess_openvino_model(
                ov_model,
                input_names=input_names,
                output_names=output_names,
                metadata=metadata,
            )

            # Save the model
            xml_path = output_path.with_suffix(".xml")
            ov.save_model(ov_model, xml_path)
            self.logger.info(f"✓ Model saved: {xml_path}")

            return xml_path

        except Exception as e:
            self.logger.error(f"Failed to export model: {e}")
            raise

    def export_to_onnx(
        self,
        model: nn.Module,
        input_shape: List[int],
        output_path: Path,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        opset_version: int = 17,
    ) -> Path:
        """
        Export PyTorch model to ONNX format.

        For detection models (SSD, Faster R-CNN, Mask R-CNN, RetinaNet), this method
        wraps the model to convert dictionary outputs to tensor tuples.

        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape [batch, channels, height, width]
            output_path: Path to save the model (without extension)
            input_names: Names for input tensors
            output_names: Names for output tensors
            model_type: Model type (used for selecting export wrapper)
            opset_version: ONNX opset version (default: 17)

        Returns:
            Path to the exported .onnx file
        """
        try:
            model.eval()

            # Wrap detection models for ONNX export
            if model_type in EXPORT_WRAPPERS:
                wrapper_class = EXPORT_WRAPPERS[model_type]
                self.logger.info(f"Wrapping model with {wrapper_class.__name__}")
                export_model = wrapper_class(model)
                # Use default output names for detection models if not specified
                if output_names is None or output_names == ["result"]:
                    output_names = DETECTION_OUTPUT_NAMES.get(model_type, ["output"])
            else:
                export_model = model
                if output_names is None:
                    output_names = ["output"]

            if input_names is None:
                input_names = ["image"]

            # Create dummy input
            dummy_input = torch.randn(*input_shape)

            # Configure dynamic axes for detection models
            # Only outputs have variable length (number of detections)
            # Input should be FIXED to ensure model_api sets orig_width/orig_height properly
            dynamic_axes = {}
            if model_type in EXPORT_WRAPPERS:
                # Detection outputs have variable length
                for output_name in output_names:
                    dynamic_axes[output_name] = {0: "num_detections"}

            # Export to ONNX
            onnx_path = output_path.with_suffix(".onnx")
            self.logger.info(f"Exporting to ONNX (opset {opset_version}): {onnx_path}")

            # Use legacy TorchScript-based export (dynamo=False) for detection models
            # The new TorchDynamo export doesn't support data-dependent control flow
            # in operations like batched_nms
            torch.onnx.export(
                export_model,
                dummy_input,
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=opset_version,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                dynamo=False,
            )

            self.logger.info(f"✓ ONNX export complete: {onnx_path}")
            return onnx_path

        except Exception as e:
            self.logger.error(f"Failed to export to ONNX: {e}")
            raise

    def _postprocess_openvino_model(
        self,
        model: Any,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        metadata: Optional[Dict[tuple, str]] = None,
    ) -> Any:
        """
        Post-process OpenVINO model (set names, add metadata).

        Args:
            model: OpenVINO model
            input_names: Names for input tensors
            output_names: Names for output tensors
            metadata: Metadata to embed

        Returns:
            Post-processed model
        """
        # Set input names
        if input_names:
            for i, name in enumerate(input_names):
                if i < len(model.inputs):
                    model.input(i).set_names({name})
                    self.logger.debug(f"Set input {i} name to: {name}")

        # Set output names
        if output_names:
            for i, name in enumerate(output_names):
                if i < len(model.outputs):
                    model.output(i).set_names({name})
                    self.logger.debug(f"Set output {i} name to: {name}")

        # Add metadata
        if metadata:
            for key, value in metadata.items():
                model.set_rt_info(value, list(key))
                self.logger.debug(f"Set metadata {key}: {value}")

        return model

    def process_model_config(self, config: Dict[str, Any]) -> bool:
        """
        Process a single model configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            True if successful, False otherwise
        """
        model_short_name = config.get("model_short_name", "unknown")

        try:
            self.logger.info("=" * 80)
            self.logger.info(f"Processing model: {config.get('model_full_name', model_short_name)}")
            self.logger.info(f"Short name: {model_short_name}")
            if "description" in config:
                self.logger.info(f"Description: {config['description']}")
            self.logger.info("=" * 80)

            # Download weights
            weights_url = config["weights_url"]
            weights_path = self.download_weights(weights_url)

            # Load model class
            model_class_name = config.get("model_class_name", "torch.nn.Module")
            model_class = self.load_model_class(model_class_name)

            # Load checkpoint
            checkpoint = self.load_checkpoint(weights_path)

            # Create model
            model_params = config.get("model_params")
            model = self.create_model(model_class, checkpoint, model_params)

            # Prepare export parameters
            input_shape = config.get("input_shape", [1, 3, 224, 224])
            input_names = config.get("input_names", ["input"])
            output_names = config.get("output_names", ["result"])

            # Extract height and width from input_shape [batch, channels, height, width]
            orig_height = str(input_shape[2])
            orig_width = str(input_shape[3])

            # Prepare metadata from config (with defaults for ImageNet normalization)
            reverse_input_channels = config.get("reverse_input_channels", True)
            mean_values = config.get("mean_values", "123.675 116.28 103.53")
            scale_values = config.get("scale_values", "58.395 57.12 57.375")

            metadata = {
                ("model_info", "model_type"): config.get("model_type", ""),
                ("model_info", "model_short_name"): model_short_name,
                ("model_info", "reverse_input_channels"): str(reverse_input_channels),
                ("model_info", "mean_values"): mean_values,
                ("model_info", "scale_values"): scale_values,
                ("model_info", "orig_height"): orig_height,
                ("model_info", "orig_width"): orig_width,
            }

            # Add labels if specified in config
            labels_config = config.get("labels")
            if labels_config:
                labels = self.get_labels(labels_config)
                if labels:
                    metadata["model_info", "labels"] = labels
                    self.logger.info(f"Added {labels_config} labels to metadata")
                else:
                    self.logger.warning(f"Could not load labels for: {labels_config}")

            # Get export method and model type
            model_type = config.get("model_type", "")
            export_method = config.get("export_method", "direct")
            onnx_opset_version = config.get("onnx_opset_version", 17)

            # Auto-detect export method for detection models
            if model_type in EXPORT_WRAPPERS and export_method == "direct":
                self.logger.info(f"Auto-switching to ONNX export for {model_type} model")
                export_method = "onnx"

            output_path = self.output_dir / model_short_name
            self.export_to_openvino(
                model=model,
                input_shape=input_shape,
                output_path=output_path,
                input_names=input_names,
                output_names=output_names,
                metadata=metadata,
                export_method=export_method,
                model_type=model_type,
                onnx_opset_version=onnx_opset_version,
            )

            self.logger.info(f"✓ Successfully converted {model_short_name}")
            return True

        except (ValueError, RuntimeError, ImportError, FileNotFoundError) as e:
            self.logger.error(f"✗ Failed to process model {model_short_name}: {e}")
            import traceback

            self.logger.debug(traceback.format_exc())
            return False

    def process_config_file(
        self,
        config_path: Path,
        model_filter: Optional[str] = None,
    ) -> tuple[int, int]:
        """
        Process models from a configuration file.

        Args:
            config_path: Path to JSON configuration file
            model_filter: Optional model short name to process (process only this model)

        Returns:
            Tuple of (successful_count, failed_count)
        """
        try:
            with Path(config_path).open() as f:
                config = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load configuration file: {e}")
            raise

        models = config.get("models", [])

        if not models:
            self.logger.warning("No models found in configuration file")
            return 0, 0

        self.logger.info(f"Configuration validated: {len(models)} models found")

        # Filter models if requested
        if model_filter:
            models = [m for m in models if m.get("model_short_name") == model_filter]
            if not models:
                self.logger.error(f"Model '{model_filter}' not found in configuration")
                return 0, 0
            self.logger.info(f"Processing only model: {model_filter}")

        successful = 0
        failed = 0

        for model_config in models:
            if self.process_model_config(model_config):
                successful += 1
            else:
                failed += 1

        return successful, failed


def list_models(config_path: Path):
    """List all models in a configuration file."""
    try:
        with config_path.open() as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return

    models = config.get("models", [])

    if not models:
        print("No models found in configuration")
        return

    print(f"\nFound {len(models)} models:\n")
    print(f"{'Short Name':<30} {'Full Name':<40} {'Type':<20}")
    print("-" * 90)

    for model in models:
        short_name = model.get("model_short_name", "N/A")
        full_name = model.get("model_full_name", "N/A")
        model_type = model.get("model_type", "N/A")
        print(f"{short_name:<30} {full_name:<40} {model_type:<20}")

    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to OpenVINO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all models in config
  uv run python model_converter.py config.json -o ./models

  # Convert a specific model
  uv run python model_converter.py config.json -o ./models --model resnet50

  # List all models in config
  uv run python model_converter.py config.json --list

  # Enable verbose logging
  uv run python model_converter.py config.json -o ./models -v
        """,
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to JSON configuration file",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./converted_models"),
        help="Output directory for converted models (default: ./converted_models)",
    )

    parser.add_argument(
        "-c",
        "--cache",
        type=Path,
        default=Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
        help="Cache directory for downloaded weights (default: ~/.cache/torch/hub/checkpoints)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Process only the specified model (by model_short_name)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all models in the configuration file and exit",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Check if config file exists
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
        return 1

    # List models and exit
    if args.list:
        list_models(args.config)
        return 0

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading configuration from: {args.config}")

    try:
        # Create converter
        converter = ModelConverter(
            output_dir=args.output,
            cache_dir=args.cache,
            verbose=args.verbose,
        )

        logger.info(f"Output directory: {args.output}")
        logger.info(f"Cache directory: {args.cache}")

        # Process models
        successful, failed = converter.process_config_file(
            config_path=args.config,
            model_filter=args.model,
        )

        # Print summary
        logger.info("=" * 80)
        logger.info("Conversion Summary:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total: {successful + failed}")
        logger.info("=" * 80)

        return 0 if failed == 0 else 1
    except (ValueError, RuntimeError, ImportError, FileNotFoundError) as e:
        logger.error(f"Failed to process model: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
