#!/usr/bin/env python3
"""
PyTorch to OpenVINO Model Converter

Usage:
    python model_converter.py config.json -o ./output_models

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
            label_set: Name of the label set (e.g., "IMAGENET1K_V1")

        Returns:
            Space-separated string of labels, or None if not found
        """
        if label_set == "IMAGENET1K_V1":
            from torchvision.models._meta import _IMAGENET_CATEGORIES
            categories = _IMAGENET_CATEGORIES
            categories = [label.replace(" ", "_") for label in categories]
            return " ".join(categories)
        
        return None

    def download_weights(self, url: str, filename: Optional[str] = None) -> Path:
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
            urllib.request.urlretrieve(url, cached_file)
            self.logger.info("✓ Download complete")
            return cached_file
        except Exception as e:
            self.logger.error(f"Failed to download weights: {e}")
            raise

    def load_model_class(self, class_path: str) -> type:
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
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            self.logger.debug(f"Loaded class: {class_name}")
            return model_class
        except Exception as e:
            self.logger.error(f"Failed to import module {module_path}: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load PyTorch checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint dictionary
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
                    raise ValueError(
                        "Checkpoint contains only state_dict. "
                        "Please specify the model class instead of torch.nn.Module"
                    )
                else:
                    # Assume checkpoint is the model itself
                    model = checkpoint
                
                if not isinstance(model, nn.Module):
                    raise ValueError("Checkpoint does not contain a valid model")
            else:
                # Instantiate model class
                if model_params:
                    model = model_class(**model_params)
                else:
                    model = model_class()

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
        metadata: Optional[Dict[tuple, str]] = None
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

        Returns:
            Path to the exported .xml file
        """
        import openvino as ov

        try:
            model.eval()
            dummy_input = torch.randn(*input_shape)
            self.logger.info("Direct PyTorch to OpenVINO conversion")
            ov_model = ov.convert_model(model, example_input=dummy_input)
            self.logger.info("✓ PyTorch to OpenVINO conversion complete")

            # Reshape model to fixed input shape (remove dynamic dimensions)
            first_input = ov_model.input(0)
            input_name_for_reshape = list(first_input.get_names())[0] if first_input.get_names() else 0
            
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
            }

            # Add labels if specified in config
            labels_config = config.get("labels")
            if labels_config:
                labels = self.get_labels(labels_config)
                if labels:
                    metadata[("model_info", "labels")] = labels
                    self.logger.info(f"Added {labels_config} labels to metadata")
                else:
                    self.logger.warning(f"Could not load labels for: {labels_config}")
            output_path = self.output_dir / model_short_name
            self.export_to_openvino(
                model=model,
                input_shape=input_shape,
                output_path=output_path,
                input_names=input_names,
                output_names=output_names,
                metadata=metadata
            )

            self.logger.info(f"✓ Successfully converted {model_short_name}")
            return True

        except Exception as e:
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
            with open(config_path) as f:
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
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
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
  python model_converter.py config.json -o ./models

  # Convert a specific model
  python model_converter.py config.json -o ./models --model resnet50

  # List all models in config
  python model_converter.py config.json --list

  # Enable verbose logging
  python model_converter.py config.json -o ./models -v
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
    except Exception as e:
        logger.error(f"Failed to process model: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
