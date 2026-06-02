#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""PyTorch-based converter shared by torchvision and timm converters."""

import importlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from model_converter.adapters import get_adapter
from model_converter.converters.base import BaseConverter
from model_converter.reporting import AccuracyResults

_MODEL_API_METADATA_FIELDS = (
    "resize_type",
    "pad_value",
    "input_dtype",
    "confidence_threshold",
    "postprocess_semantic_masks",
    "nms_execute",
    "iou_threshold",
    "agnostic_nms",
    "nms_max_predictions",
)


class PyTorchConverter(BaseConverter):
    """Shared converter for PyTorch-based models (torchvision, timm).

    Provides common export-to-OpenVINO logic, model loading utilities,
    and quantization workflow.
    """

    def get_labels(self, label_set: str) -> str | None:
        """Get label list for a given label set.

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

        if label_set == "IMAGENET21K":
            from timm.data import ImageNetInfo

            info = ImageNetInfo("imagenet21k")
            categories = info.label_descriptions()
            categories = [desc.split(",")[0].strip().replace(" ", "_") for desc in categories]
            return " ".join(categories)

        if label_set == "COCO_V1":
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

            categories = MaskRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"]
            categories = [label.replace(" ", "_") for label in categories]
            return " ".join(categories)

        return None

    def load_model_class(self, class_path: str) -> type:
        """Dynamically load a model class from a Python path.

        Args:
            class_path: Full Python path to the class (e.g., 'torchvision.models.resnet.resnet50')

        Returns:
            The model class
        """
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            self.logger.debug(f"Importing module: {module_path}")
            # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            self.logger.debug(f"Loaded class: {class_name}")
            return model_class
        except Exception as e:
            self.logger.error(f"Failed to import module {module_path}: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """Load PyTorch checkpoint file.

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
        checkpoint: dict[str, Any],
        model_params: dict[str, Any] | None = None,
    ) -> nn.Module:
        """Create and initialize model instance.

        Args:
            model_class: Model class to instantiate
            checkpoint: Checkpoint containing model weights
            model_params: Optional parameters for model initialization

        Returns:
            Initialized model instance
        """
        try:
            if model_class == torch.nn.Module:
                if "model" in checkpoint:
                    model = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    error_msg = (
                        "Checkpoint contains only state_dict. Please specify the model class instead of torch.nn.Module"
                    )
                    raise ValueError(error_msg)
                else:
                    model = checkpoint

                if not isinstance(model, nn.Module):
                    error_msg = "Checkpoint does not contain a valid model"
                    raise ValueError(error_msg)
            else:
                model = model_class(**model_params) if model_params else model_class()

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
        input_shape: list[int],
        output_path: Path,
        model_config: dict[str, Any],
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        metadata: dict[tuple[str, str], str] | None = None,
    ) -> tuple[Path, Path]:
        """Export PyTorch model to OpenVINO format.

        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape [batch, channels, height, width]
            output_path: Path to save the model (without extension)
            model_config: Model configuration used for README rendering
            input_names: Names for input tensors
            output_names: Names for output tensors
            metadata: Metadata to embed in the model

        Returns:
            Tuple of (fp16_model_path, fp32_model_path) - FP16 for final use, FP32 for quantization
        """
        import openvino as ov

        try:
            model = self._prepare_model_for_export(model, model_config)
            model.eval()
            dummy_input = self._create_example_input(input_shape, model_config)
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

            # Create output folder with -fp16-ov suffix
            model_name = output_path.name
            output_folder = output_path.parent / f"{model_name}-fp16-ov"
            output_folder.mkdir(parents=True, exist_ok=True)

            # Save FP32 model for quantization (temporary)
            fp32_xml_path = output_folder / f"{model_name}_fp32.xml"
            ov.save_model(ov_model, fp32_xml_path, compress_to_fp16=False)
            self.logger.debug(f"Saved FP32 model for quantization: {fp32_xml_path}")

            # Save the FP16 model (final)
            xml_path = output_folder / f"{model_name}.xml"
            ov.save_model(ov_model, xml_path, compress_to_fp16=True)
            self.logger.info(f"✓ Model saved: {xml_path}")

            # Save model_info as config.json to track downloads
            with (output_folder / "config.json").open("w") as f:
                json.dump(ov_model.get_rt_info(["model_info"]).value, f, indent=4)

            # Copy .gitattributes file
            gitattributes_template = Path(__file__).parent.parent / "templates" / ".gitattributes"
            if gitattributes_template.exists():
                shutil.copy2(gitattributes_template, output_folder / ".gitattributes")
                self.logger.debug(f"Copied .gitattributes to: {output_folder}")

            # Copy README for FP16 model
            self.copy_readme(
                model_config,
                output_folder,
                variant="fp16",
            )

            return xml_path, fp32_xml_path

        except Exception as e:
            self.logger.error(f"Failed to export model: {e}")
            raise

    def _prepare_model_for_export(self, model: nn.Module, model_config: dict[str, Any]) -> nn.Module:
        """Prepare model for OpenVINO conversion."""
        model_type = str(model_config.get("model_type", ""))
        adapted = get_adapter(model_type, model)
        if adapted is not model:
            self.logger.info(f"Applied export adapter for model type: {model_type}")
        return adapted

    def _create_example_input(self, input_shape: list[int], model_config: dict[str, Any]) -> torch.Tensor:
        """Create example input suitable for the configured model type."""
        if str(model_config.get("model_type", "")).lower() == "maskrcnn":
            return torch.rand(*input_shape)
        return torch.randn(*input_shape)

    def _postprocess_openvino_model(
        self,
        model: Any,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        metadata: dict[tuple[str, str], str] | None = None,
    ) -> Any:
        """Post-process OpenVINO model (set names, add metadata).

        Args:
            model: OpenVINO model
            input_names: Names for input tensors
            output_names: Names for output tensors
            metadata: Metadata to embed

        Returns:
            Post-processed model
        """
        if input_names:
            for i, name in enumerate(input_names):
                if i < len(model.inputs):
                    model.input(i).set_names({name})
                    self.logger.debug(f"Set input {i} name to: {name}")

        if output_names:
            for i, name in enumerate(output_names):
                if i < len(model.outputs):
                    model.output(i).set_names({name})
                    self.logger.debug(f"Set output {i} name to: {name}")

        if metadata:
            for key, value in metadata.items():
                model.set_rt_info(value, list(key))
                self.logger.debug(f"Set metadata {key}: {value}")

        return model

    def _build_metadata(self, config: dict[str, Any]) -> dict[tuple[str, str], str]:
        """Build metadata dictionary from model config."""
        model_short_name = config.get("model_short_name", "unknown")
        reverse_input_channels = config.get("reverse_input_channels", True)
        mean_values = config.get("mean_values", "123.675 116.28 103.53")
        scale_values = config.get("scale_values", "58.395 57.12 57.375")
        model_type = config.get("model_type", "")

        metadata = {
            ("model_info", "model_type"): model_type,
            ("model_info", "model_short_name"): model_short_name,
            ("model_info", "reverse_input_channels"): self._metadata_value(reverse_input_channels),
            ("model_info", "mean_values"): mean_values,
            ("model_info", "scale_values"): scale_values,
        }

        for metadata_field in _MODEL_API_METADATA_FIELDS:
            if metadata_field in config and config[metadata_field] is not None:
                metadata["model_info", metadata_field] = self._metadata_value(config[metadata_field])

        # Add labels if specified in config
        labels_config = config.get("labels")
        if labels_config:
            labels = self.get_labels(labels_config)
            if labels:
                metadata["model_info", "labels"] = labels
                self.logger.info(f"Added {labels_config} labels to metadata")
            else:
                self.logger.warning(f"Could not load labels for: {labels_config}")

        return metadata

    def validate_torch_model(
        self,
        model: nn.Module,
        validation_data: list[Any],
        labels: list[int],
    ) -> float | None:
        """Validate the original PyTorch model and compute top-1 accuracy.

        Runs inference on the same preprocessed validation tensors used for the
        OpenVINO models, so the result is comparable to the FP32/FP16/INT8
        accuracies.

        Args:
            model: The original PyTorch model (before OpenVINO conversion).
            validation_data: Preprocessed validation images (NCHW numpy arrays).
            labels: Ground truth class labels.

        Returns:
            Top-1 accuracy (0.0 to 1.0), or ``None`` if validation failed.
        """
        try:
            model.eval()
            predictions: list[int] = []
            with torch.no_grad():
                for img in validation_data:
                    output = model(torch.from_numpy(img).float())
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    predictions.append(int(torch.argmax(output, dim=1)[0].item()))

            correct = sum(predicted == label for predicted, label in zip(predictions, labels))
            return correct / len(labels)

        except (RuntimeError, TypeError, ValueError) as e:
            self.logger.error(f"Failed to validate PyTorch model: {e}")
            return None

    def _quantize_and_cleanup(self, config: dict[str, Any], fp32_model_path: Path, **kwargs: Any) -> AccuracyResults:
        """Run INT8 quantization and clean up temporary FP32 model files.

        Returns:
            The accuracies measured during quantization and the INT8 success flag.
        """
        model_type = kwargs["model_type"]
        accuracy = AccuracyResults()
        self.logger.info("Creating calibration dataset for INT8 quantization")
        return_validation_labels = model_type == "Classification" and bool(config.get("labels"))

        if return_validation_labels:
            self.logger.info("Creating validation dataset for accuracy measurement")
        validation_data, validation_labels = self.create_calibration_dataset(
            input_shape=kwargs["input_shape"],
            mean_values=kwargs["mean_values"],
            scale_values=kwargs["scale_values"],
            reverse_input_channels=kwargs["reverse_input_channels"],
            subset_size=300,
            return_labels=return_validation_labels,
        )

        if validation_data:
            torch_model = kwargs.get("torch_model")
            if validation_labels and torch_model is not None:
                self.logger.info("Validating original PyTorch model accuracy...")
                original_accuracy = self.validate_torch_model(torch_model, validation_data, validation_labels)
                if original_accuracy is not None:
                    self.logger.info(f"Original Top-1 Accuracy: {original_accuracy * 100:.2f}%")
                    accuracy.original_accuracy = original_accuracy
                    accuracy.measured = True

            self.quantize_model(
                model_path=fp32_model_path,
                calibration_data=validation_data,
                model_config=config,
                preset="mixed",
                validation_data=validation_data if validation_labels else None,
                validation_labels=validation_labels or None,
                accuracy_results=accuracy,
            )

        # Clean up temporary FP32 model after quantization
        try:
            if fp32_model_path.exists():
                fp32_model_path.unlink()
                self.logger.debug(f"Removed temporary FP32 model: {fp32_model_path}")
            fp32_bin_path = fp32_model_path.with_suffix(".bin")
            if fp32_bin_path.exists():
                fp32_bin_path.unlink()
                self.logger.debug(f"Removed temporary FP32 weights: {fp32_bin_path}")
        except OSError as e:
            self.logger.warning(f"Failed to remove temporary FP32 files: {e}")

        return accuracy
