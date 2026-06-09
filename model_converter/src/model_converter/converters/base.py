#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Base converter class for model conversion pipelines."""

import json
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from model_converter.reporting import (
    AccuracyResults,
    ConversionResult,
    determine_status,
    original_url_for_config,
    upsert_result,
)

if TYPE_CHECKING:
    from model_converter.dataset_registry import DatasetRegistry


class BaseConverter(ABC):
    """Abstract base class for model converters.

    Provides shared functionality for calibration dataset creation,
    quantization, README rendering, and model validation.
    """

    def __init__(
        self,
        output_dir: Path,
        cache_dir: Path,
        verbose: bool = False,
        dataset_registry: "DatasetRegistry | None" = None,
        report_path: Path | None = None,
    ):
        """Initialize the BaseConverter.

        Args:
            output_dir: Directory to save converted models
            cache_dir: Directory to cache downloaded weights
            verbose: Enable verbose logging
            dataset_registry: Dataset registry for resolving dataset paths
            report_path: Path to the Markdown report file.  When set, each
                non-skipped result is upserted into the file immediately after
                conversion.
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.dataset_registry = dataset_registry
        self.report_path = Path(report_path) if report_path else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.results: list[ConversionResult] = []

        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def process_model_config(self, config: dict[str, Any]) -> bool:
        """Process a single model configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            True if successful, False otherwise
        """

    def _resolve_dataset_path(self, config: dict[str, Any]) -> Path | None:
        """Resolve dataset path from model configuration.

        Extracts the 'dataset_type' field from the model config and resolves
        it to a filesystem path using the dataset registry. Returns None if
        no dataset_type is specified or no registry is available.

        Args:
            config: Model configuration dictionary

        Returns:
            Path to the dataset directory, or None if not applicable

        Raises:
            ValueError: If dataset_type is specified but not in registry
        """
        if self.dataset_registry is None:
            return None

        return self.dataset_registry.resolve_from_config(config)

    def _build_result(self, config: dict[str, Any]) -> ConversionResult:
        """Create a ConversionResult seeded from a model configuration."""
        model_short_name = str(config.get("model_short_name", "unknown"))
        return ConversionResult(
            model_short_name=model_short_name,
            model_full_name=str(config.get("model_full_name", model_short_name)),
            model_type=str(config.get("model_type", "")),
            model_library=str(config.get("model_library", "")),
            original_url=original_url_for_config(config),
        )

    def _record_result(
        self,
        result: ConversionResult,
        *,
        converted: bool,
        quantized: bool,
        skipped: bool = False,
        accuracy: AccuracyResults | None = None,
    ) -> ConversionResult:
        """Finalize a ConversionResult: copy accuracies, set status, and store it."""
        if accuracy is not None and accuracy.measured:
            result.original_accuracy = accuracy.original_accuracy
            result.fp32_accuracy = accuracy.fp32_accuracy
            result.fp16_accuracy = accuracy.fp16_accuracy
            result.int8_accuracy = accuracy.int8_accuracy
        result.status, result.status_detail = determine_status(
            result,
            converted=converted,
            quantized=quantized,
            skipped=skipped,
        )
        self.results.append(result)
        if self.report_path is not None and not skipped:
            upsert_result(result, self.report_path)
        return result

    def copy_readme(
        self,
        model_config: dict[str, Any],
        output_folder: Path,
        variant: str = "fp16",
    ) -> None:
        """Copy README template to model folder and replace placeholders.

        Args:
            model_config: Model configuration used to fill template placeholders
            output_folder: Folder where the model is saved
            variant: Model variant ('fp16' or 'int8')
        """
        try:
            model_short_name = str(model_config.get("model_short_name", "")).strip()
            model_library = str(model_config.get("model_library", "timm")).strip()
            model_license = str(model_config.get("license", "")).strip()
            model_license_link = str(model_config.get("license_link", "")).strip()
            docs = str(model_config.get("docs", "")).strip()

            def template_placeholder(name: str) -> str:
                return f"<<{name}>>"

            if not model_short_name:
                error_msg = "Model config must define a non-empty model_short_name"
                raise ValueError(error_msg)

            if not model_license_link:
                error_msg = f"Model '{model_short_name}' must define a non-empty license_link"
                raise ValueError(error_msg)

            if not model_license:
                error_msg = f"Model '{model_short_name}' must define a non-empty license"
                raise ValueError(error_msg)

            if not docs:
                self.logger.warning(
                    f"Model '{model_short_name}' does not define 'docs' field. Placeholder will be empty.",
                )

            # Determine which README template to use based on model library
            template_name = f"README-{model_library}-{variant}.md"
            template_path = Path(__file__).parent.parent / "templates" / template_name

            if not template_path.exists():
                self.logger.warning(f"README template not found: {template_path}")
                return

            # Read template
            readme_content = template_path.read_text()

            placeholders = {
                template_placeholder("license"): model_license,
                template_placeholder("license_link"): model_license_link,
                template_placeholder("model_name"): model_short_name,
                template_placeholder("model_short_name"): model_short_name,
                template_placeholder("variant"): variant,
                template_placeholder("docs"): docs,
            }

            # Handle tags list → YAML formatting
            tags = model_config.get("tags")
            if tags and isinstance(tags, list):
                tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
                placeholders[template_placeholder("tags_yaml")] = tags_yaml
            else:
                placeholders[template_placeholder("tags_yaml")] = ""

            for key, value in model_config.items():
                if value is None:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    placeholders[template_placeholder(key)] = str(value)

            for placeholder, value in placeholders.items():
                readme_content = readme_content.replace(placeholder, value)

            # Write to model folder
            output_readme = output_folder / "README.md"
            output_readme.write_text(readme_content)
            self.logger.debug(f"Copied README to: {output_readme}")

        except (OSError, UnicodeError, ValueError) as e:
            self.logger.warning(f"Failed to copy README: {e}")

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

    def _collect_dataset_entries(self, image_dir: Path) -> list[tuple[Path, int]]:
        """Collect dataset image paths with their class labels."""
        image_entries: list[tuple[Path, int]] = []
        for class_dir in sorted(image_dir.iterdir()):
            if class_dir.is_dir():
                class_label = int(class_dir.name)
                for pattern in ["*.JPEG", "*.jpg", "*.png"]:
                    for img_path in class_dir.glob(pattern):
                        image_entries.append((img_path, class_label))
        return image_entries

    @staticmethod
    def _crop_resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
        """Center-crop to target aspect ratio, then resize.

        Matches the standard ImageNet evaluation pipeline used by timm and
        torchvision (resize shorter side, then center-crop), giving better
        accuracy than a plain stretch-to-fit resize.

        Args:
            img: Input image in HWC layout.
            width: Target width in pixels.
            height: Target height in pixels.

        Returns:
            Cropped and resized image.
        """
        h, w = img.shape[:2]
        desired_ar = width / height
        if desired_ar == 1:
            side = min(h, w)
            y0 = (h - side) // 2
            x0 = (w - side) // 2
            img = img[y0 : y0 + side, x0 : x0 + side]
        elif w / h > desired_ar:  # image is too wide — crop width
            new_w = int(h * desired_ar)
            x0 = (w - new_w) // 2
            img = img[:, x0 : x0 + new_w]
        else:  # image is too tall — crop height
            new_h = int(w / desired_ar)
            y0 = (h - new_h) // 2
            img = img[y0 : y0 + new_h, :]
        return cv2.resize(img, (width, height))

    def _preprocess_calibration_image(
        self,
        img_path: Path,
        width: int,
        height: int,
        mean: np.ndarray,
        scale: np.ndarray,
        reverse_input_channels: bool,
        resize_type: str = "standard",
    ) -> np.ndarray | None:
        """Load and preprocess a single calibration image."""
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        img = self._crop_resize(img, width, height) if resize_type == "crop" else cv2.resize(img, (width, height))
        img = img.astype(np.float32)

        if reverse_input_channels:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = (img - mean) / scale
        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, axis=0)

    def create_calibration_dataset(
        self,
        input_shape: list[int],
        mean_values: str | None = None,
        scale_values: str | None = None,
        reverse_input_channels: bool = True,
        subset_size: int = 5000,
        return_labels: bool = False,
        resize_type: str = "standard",
        dataset_path: Path | None = None,
    ) -> tuple[list[np.ndarray], list[int]]:
        """Create calibration dataset from sample validation images.

        Args:
            input_shape: Target input shape [batch, channels, height, width]
            mean_values: Space-separated mean values for normalization
            scale_values: Space-separated scale values for normalization
            reverse_input_channels: Whether to reverse RGB to BGR
            subset_size: Number of images to use for calibration
            return_labels: Whether to return labels along with images
            resize_type: Resize strategy — ``"crop"`` center-crops to the
                target aspect ratio before resizing (matches standard ImageNet
                evaluation), ``"standard"`` stretches directly to the target
                size.
            dataset_path: Path to dataset directory (overrides registry resolution)

        Returns:
            Tuple of (images, labels); both empty lists when dataset is unavailable.
        """
        if not dataset_path or not dataset_path.exists():
            self.logger.warning("Dataset path not provided or doesn't exist. Skipping quantization.")
            return [], []

        # Parse mean and scale values
        mean = np.array([float(x) for x in mean_values.split()]) if mean_values else np.array([0, 0, 0])
        scale = np.array([float(x) for x in scale_values.split()]) if scale_values else np.array([1, 1, 1])

        _, _, height, width = input_shape
        calibration_data: list[np.ndarray] = []

        # Find all images in the dataset
        image_dir = dataset_path
        if not image_dir.exists():
            self.logger.error(f"Image directory not found: {image_dir}")
            return ([], [])

        image_entries = self._collect_dataset_entries(image_dir)
        if not image_entries:
            self.logger.error("No images found in dataset")
            return ([], [])

        self.logger.info(f"Found {len(image_entries)} images in dataset")
        self.logger.info(f"Using {min(subset_size, len(image_entries))} images for calibration")

        if return_labels:
            labels: list[int] = []
            for i, (img_path, class_label) in enumerate(image_entries[:subset_size]):
                try:
                    img = self._preprocess_calibration_image(
                        img_path=img_path,
                        width=width,
                        height=height,
                        mean=mean,
                        scale=scale,
                        reverse_input_channels=reverse_input_channels,
                        resize_type=resize_type,
                    )
                    if img is None:
                        continue

                    calibration_data.append(img)
                    labels.append(class_label)

                    if (i + 1) % 50 == 0:
                        self.logger.debug(f"Processed {i + 1}/{subset_size} images")

                except (cv2.error, OSError, TypeError, ValueError) as e:
                    self.logger.warning(f"Failed to process {img_path}: {e}")
                    continue

            self.logger.info(f"✓ Created calibration dataset with {len(calibration_data)} images")
            return calibration_data, labels

        for i, (img_path, _) in enumerate(image_entries[:subset_size]):
            try:
                img = self._preprocess_calibration_image(
                    img_path=img_path,
                    width=width,
                    height=height,
                    mean=mean,
                    scale=scale,
                    reverse_input_channels=reverse_input_channels,
                    resize_type=resize_type,
                )
                if img is None:
                    continue

                calibration_data.append(img)

                if (i + 1) % 50 == 0:
                    self.logger.debug(f"Processed {i + 1}/{subset_size} images")

            except (cv2.error, OSError, TypeError, ValueError) as e:
                self.logger.warning(f"Failed to process {img_path}: {e}")
                continue

        self.logger.info(f"✓ Created calibration dataset with {len(calibration_data)} images")
        return calibration_data, []

    def validate_model(
        self,
        model_path: Path,
        validation_data: list[np.ndarray],
        labels: list[int],
    ) -> float:
        """Validate OpenVINO model and compute top-1 accuracy.

        Args:
            model_path: Path to the OpenVINO model (.xml)
            validation_data: List of validation images
            labels: List of ground truth labels

        Returns:
            Top-1 accuracy (0.0 to 1.0)
        """
        try:
            import openvino as ov

            core = ov.Core()
            model = core.read_model(model_path)
            compiled_model = core.compile_model(model, device_name="CPU")
            output_layer = compiled_model.outputs[0]

            predictions: list[int] = []
            for img in validation_data:
                result = compiled_model(img)[output_layer]
                pred_class = np.argmax(result, axis=1)[0]
                predictions.append(pred_class)

            # Compute accuracy
            correct = sum(predicted == label for predicted, label in zip(predictions, labels))
            return correct / len(labels)

        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as e:
            self.logger.error(f"Failed to validate model: {e}")
            return 0.0

    def quantize_model(
        self,
        model_path: Path,
        calibration_data: list[np.ndarray],
        model_config: dict[str, Any],
        preset: str = "accuracy",
        validation_data: list[np.ndarray] | None = None,
        validation_labels: list[int] | None = None,
        accuracy_results: AccuracyResults | None = None,
    ) -> Path:
        """Quantize OpenVINO model to INT8 using NNCF.

        Args:
            model_path: Path to the FP32 OpenVINO model (.xml)
            calibration_data: List of calibration images
            model_config: Model configuration used for README rendering
            preset: Quantization preset ('accuracy', 'performance', 'mixed')
            validation_data: Optional validation images for accuracy measurement
            validation_labels: Optional validation labels for accuracy measurement
            accuracy_results: Optional collector populated with measured accuracies
                and the INT8 success flag.

        Returns:
            Path to the quantized model
        """
        if not calibration_data:
            self.logger.warning("No calibration data provided. Skipping quantization.")
            return model_path

        try:
            import nncf
            import openvino as ov

            self.logger.info(f"Quantizing model with {len(calibration_data)} calibration samples")
            self.logger.info(f"Using preset: {preset}")

            # Load the model
            core = ov.Core()
            model = core.read_model(model_path)

            # Map preset string to NNCF enum
            preset_map = {
                "performance": nncf.QuantizationPreset.PERFORMANCE,
                "mixed": nncf.QuantizationPreset.MIXED,
            }
            nncf_preset = preset_map.get(preset.lower(), nncf.QuantizationPreset.MIXED)

            quantize_kwargs: dict[str, Any] = {}
            model_type = model_config.get("quantization_model_type")
            if model_type and model_type.lower() == "transformer":
                quantize_kwargs["model_type"] = nncf.ModelType.TRANSFORMER

            quantized_model = nncf.quantize(
                model,
                calibration_dataset=nncf.Dataset(calibration_data),
                preset=nncf_preset,
                subset_size=len(calibration_data),
                **quantize_kwargs,
            )

            # Extract model name from the FP32 model path
            model_name = model_path.stem
            if model_name.endswith("_fp32"):
                model_name = model_name[:-5]

            # Create output folder with -int8-ov suffix
            output_folder = model_path.parent.parent / f"{model_name}-int8-ov"
            output_folder.mkdir(parents=True, exist_ok=True)

            # Save quantized model with model name inside the folder
            output_path = output_folder / f"{model_name}.xml"
            ov.save_model(quantized_model, output_path, compress_to_fp16=True)
            self.logger.info(f"✓ Quantized model saved: {output_path}")
            if accuracy_results is not None:
                accuracy_results.int8_succeeded = True

            # Save model_info as config.json to track downloads
            with (output_folder / "config.json").open("w") as f:
                json.dump(quantized_model.get_rt_info(["model_info"]).value, f, indent=4)

            # Validate accuracy if validation data provided
            if validation_data and validation_labels:
                self.logger.info("Validating FP32 model accuracy...")
                fp32_accuracy = self.validate_model(model_path, validation_data, validation_labels)
                self.logger.info(f"FP32 Top-1 Accuracy: {fp32_accuracy * 100:.2f}%")

                fp16_model_path = model_path.parent / f"{model_name}.xml"
                fp16_accuracy: float | None = None
                if fp16_model_path.exists():
                    self.logger.info("Validating FP16 model accuracy...")
                    fp16_accuracy = self.validate_model(fp16_model_path, validation_data, validation_labels)
                    self.logger.info(f"FP16 Top-1 Accuracy: {fp16_accuracy * 100:.2f}%")
                else:
                    self.logger.warning(f"FP16 model not found for accuracy measurement: {fp16_model_path}")

                self.logger.info("Validating INT8 model accuracy...")
                int8_accuracy = self.validate_model(output_path, validation_data, validation_labels)
                self.logger.info(f"INT8 Top-1 Accuracy: {int8_accuracy * 100:.2f}%")

                accuracy_drop = (fp32_accuracy - int8_accuracy) * 100
                self.logger.info(f"Accuracy Drop: {accuracy_drop:.2f}%")

                if accuracy_results is not None:
                    accuracy_results.fp32_accuracy = fp32_accuracy
                    accuracy_results.fp16_accuracy = fp16_accuracy
                    accuracy_results.int8_accuracy = int8_accuracy
                    accuracy_results.measured = True

            # Copy .gitattributes file
            gitattributes_template = Path(__file__).parent.parent / "templates" / ".gitattributes"
            if gitattributes_template.exists():
                shutil.copy2(gitattributes_template, output_folder / ".gitattributes")
                self.logger.debug(f"Copied .gitattributes to: {output_folder}")

            # Copy README for INT8 model
            self.copy_readme(
                model_config,
                output_folder,
                variant="int8",
            )

            return output_path

        except ImportError:
            self.logger.error("NNCF not installed. Install with: pip install nncf")
            return model_path
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            self.logger.error(f"Failed to quantize model: {e}")
            import traceback

            self.logger.debug(traceback.format_exc())
            return model_path

    @staticmethod
    def _metadata_value(value: Any) -> str:
        """Convert config values to Model API rt_info string values."""
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value)
        return str(value)
