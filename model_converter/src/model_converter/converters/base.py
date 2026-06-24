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

from model_converter.datasets import CalibrationSample, reader_for
from model_converter.metrics.coco_detection import COCO80_TO_COCO91
from model_converter.reporting import (
    AccuracyResults,
    ConversionResult,
    determine_status,
    original_url_for_config,
    upsert_result,
)

if TYPE_CHECKING:
    from model_converter.dataset_registry import DatasetRegistry
    from model_converter.metrics import CocoDetectionMAP, Metric


def _get_human_license_name(name: str) -> str:
    mapping = {
        "apache-2.0": "Apache-2.0",
    }

    return mapping.get(name, name)


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
        measure_accuracy: bool = True,
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
            measure_accuracy: When ``False``, skip per-model accuracy
                measurement even if a metric strategy is available.
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.dataset_registry = dataset_registry
        self.report_path = Path(report_path) if report_path else None
        self.measure_accuracy = measure_accuracy
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

    def _skip_if_already_converted(
        self,
        config: dict[str, Any],
        model_short_name: str,
        xml_stem: str | None = None,
    ) -> bool:
        """Record a skip and return ``True`` when FP16 and INT8 models both exist.

        ``xml_stem`` overrides the ``.xml`` file name inside the variant folders
        (defaults to ``model_short_name``); YOLO uses the Ultralytics
        ``yolo_version`` here.
        """
        xml_stem = xml_stem or model_short_name
        fp16_model_path = self.output_dir / f"{model_short_name}-fp16-ov" / f"{xml_stem}.xml"
        int8_model_path = self.output_dir / f"{model_short_name}-int8-ov" / f"{xml_stem}.xml"
        if fp16_model_path.exists() and int8_model_path.exists():
            self.logger.info(f"Skipping {model_short_name}: FP16 and INT8 models already exist")
            self._record_result(self._build_result(config), converted=False, quantized=False, skipped=True)
            return True
        return False

    def _validate_license(self, config: dict[str, Any], model_short_name: str) -> None:
        """Raise ``ValueError`` when ``license`` or ``license_link`` is missing."""
        if not config.get("license"):
            error_msg = f"Model '{model_short_name}' must define 'license' in configuration"
            raise ValueError(error_msg)
        if not config.get("license_link"):
            error_msg = f"Model '{model_short_name}' must define 'license_link' in configuration"
            raise ValueError(error_msg)

    def _log_model_banner(self, config: dict[str, Any], model_short_name: str, *, label: str = "model") -> None:
        """Log the ``"=" * 80`` processing banner shared by the converters."""
        self.logger.info("=" * 80)
        self.logger.info(f"Processing {label}: {config.get('model_full_name', model_short_name)}")
        self.logger.info(f"Short name: {model_short_name}")
        if "description" in config:
            self.logger.info(f"Description: {config['description']}")
        self.logger.info("=" * 80)

    def _finalize_success(
        self,
        config: dict[str, Any],
        model_short_name: str,
        *,
        accuracy: "AccuracyResults | None",
        quantization_attempted: bool,
    ) -> bool:
        """Record a successful conversion result and log it; always returns ``True``."""
        quantized = accuracy.int8_succeeded if quantization_attempted and accuracy is not None else True
        self._record_result(
            self._build_result(config),
            converted=True,
            quantized=quantized,
            accuracy=accuracy,
        )
        self.logger.info(f"✓ Successfully converted {model_short_name}")
        return True

    def _record_failure(
        self,
        config: dict[str, Any],
        model_short_name: str,
        error: Exception,
        *,
        label: str = "model",
    ) -> bool:
        """Log a conversion failure, record a failed result, and return ``False``."""
        import traceback

        self.logger.error(f"✗ Failed to process {label} {model_short_name}: {error}")
        self._record_result(self._build_result(config), converted=False, quantized=False)
        self.logger.debug(traceback.format_exc())
        return False

    def _select_accuracy_metric(
        self,
        config: dict[str, Any],
        dataset_path: Path | None,
        accuracy: "AccuracyResults",
    ) -> "tuple[Metric | None, bool]":
        """Resolve the metric for a config and flag whether it is top-1.

        Returns ``(metric, is_top1)``. ``metric`` is ``None`` when accuracy
        measurement is disabled or no metric applies. When a metric is found its
        ``name`` is copied onto ``accuracy``.
        """
        from model_converter.metrics import TopOneAccuracy

        metric = self._metric_for_config(config, dataset_path) if self.measure_accuracy else None
        if metric is not None:
            accuracy.metric_name = metric.name
        return metric, isinstance(metric, TopOneAccuracy)

    def _collect_metric_validation_samples(
        self,
        metric: "Metric | None",
        is_top1: bool,
        dataset_path: Path | None,
        dataset_type: str | None,
    ) -> "list[CalibrationSample] | None":
        """Collect raw validation samples for Model API-based metrics, else ``None``.

        Top-1 classification and the no-metric case use the preprocessed-tensor
        path and do not need raw samples.
        """
        if metric is None or is_top1:
            return None
        return self._collect_validation_samples(dataset_path, dataset_type, subset_size=500) or None

    def _cleanup_fp32(self, fp32_model_path: Path) -> None:
        """Remove the temporary FP32 ``.xml``/``.bin`` pair used for quantization."""
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

    def _metric_for_config(
        self,
        config: dict[str, Any],
        dataset_path: Path | None,
    ) -> "Metric | None":
        """Build the task-appropriate :class:`Metric` for a model config.

        Resolves the COCO annotation file when the dataset_type calls for it
        and forwards ``getitune_task`` so that ``Classification`` model_type
        with ``MULTI_LABEL_CLS`` is routed to multilabel mAP rather than
        top-1.
        """
        from model_converter.datasets.factory import _COCO_ANNOTATION_FILES
        from model_converter.metrics import metric_for

        dataset_type = config.get("dataset_type")
        model_type = config.get("model_type")
        task = config.get("getitune_task")
        annotation_file: Path | None = None
        if dataset_path is not None and dataset_type in _COCO_ANNOTATION_FILES:
            annotation_file = dataset_path / "annotations" / _COCO_ANNOTATION_FILES[dataset_type]
        return metric_for(dataset_type, model_type, annotation_file=annotation_file, task=task)

    def _collect_validation_samples(
        self,
        dataset_path: Path | None,
        dataset_type: str | None,
        subset_size: int = 500,
    ) -> "list[CalibrationSample]":
        """Collect up to ``subset_size`` raw samples from the dataset reader.

        Used by metric paths that need raw image paths (and per-task GT
        pointers like COCO ``image_id`` or ADE20K ``mask_path``) rather than
        preprocessed calibration tensors. Returns an empty list when the
        dataset is unavailable, unreadable, or unknown.
        """
        if dataset_path is None or not dataset_path.exists():
            return []
        try:
            reader = reader_for(dataset_type, dataset_path)
        except ValueError:
            return []
        samples: list[CalibrationSample] = []
        try:
            for sample in reader:
                samples.append(sample)
                if len(samples) >= subset_size:
                    break
        except (FileNotFoundError, OSError, ValueError) as e:
            self.logger.warning(f"Failed to enumerate validation samples: {e}")
            return []
        return samples

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
            model_license_name = _get_human_license_name(model_license)
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
                template_placeholder("license"): model_license_name,
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

        if label_set == "COCO_80":
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

            from model_converter.metrics.coco_detection import COCO80_TO_COCO91

            categories = MaskRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"]
            categories = [categories[cat_id].replace(" ", "_") for cat_id in COCO80_TO_COCO91]
            return " ".join(categories)

        return None

    def _collect_dataset_entries(
        self,
        image_dir: Path,
        dataset_type: str | None = None,
    ) -> list[tuple[Path, int]]:
        """Collect dataset image paths with their class labels.

        Dispatches by ``dataset_type`` to the appropriate
        :class:`~model_converter.datasets.DatasetReader`. ``None`` (the
        default) preserves the legacy class-folder behaviour.
        """
        reader = reader_for(dataset_type, image_dir)
        return [(sample.image_path, sample.label) for sample in reader]

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
        subset_size: int = 500,
        return_labels: bool = False,
        resize_type: str = "standard",
        dataset_path: Path | None = None,
        dataset_type: str | None = None,
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
            dataset_type: Optional dataset_type identifier (e.g. ``"coco-detection"``)
                used to dispatch path enumeration to the matching reader. Defaults
                to ``None`` for backward-compatible class-folder behaviour.

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

        image_entries = self._collect_dataset_entries(image_dir, dataset_type=dataset_type)
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

    def _measure_metric(
        self,
        model_path: Path,
        samples: "list[CalibrationSample]",
        metric: "Metric",
    ) -> float | None:
        """Measure a task-specific :class:`Metric` over an OpenVINO model via Model API.

        Loads the model through ``model_api.models.Model.create_model`` so that
        task-correct preprocessing and postprocessing (including resize-info
        reversal for detection and per-class argmax for segmentation) are
        applied. Each sample's raw image is decoded with OpenCV and passed to
        the wrapper, whose result is dispatched into ``metric.update`` by
        metric type.

        Args:
            model_path: Path to the OpenVINO ``.xml`` model whose rt_info
                identifies the model_type used by ``Model.create_model``.
            samples: Per-image ground-truth pointers (image path, optional
                COCO ``image_id``, optional ADE20K ``mask_path``).
            metric: A reset-or-fresh metric instance. The metric is reset
                before iterating and ``compute()`` is returned at the end.

        Returns:
            The scalar metric value, or ``None`` if the Model API import
            fails or model loading raises.
        """
        try:
            from model_api.models import Model
        except ImportError as e:
            self.logger.error(f"Failed to import model_api: {e}")
            return None
        try:
            metric.reset()
            wrapper = Model.create_model(str(model_path))
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            self.logger.error(f"Failed to load model {model_path}: {e}")
            return None
        for sample in samples:
            img = cv2.imread(str(sample.image_path))
            if img is None:
                continue
            try:
                result = wrapper(img)
            except (RuntimeError, TypeError, ValueError) as e:
                self.logger.debug(f"Inference failed for {sample.image_path}: {e}")
                continue
            try:
                self._update_metric_with_result(metric, result, sample)
            except (RuntimeError, TypeError, ValueError, IndexError) as e:
                self.logger.debug(f"Metric update failed for {sample.image_path}: {e}")
                continue
        return float(metric.compute())

    def _update_metric_with_result(
        self,
        metric: "Metric",
        result: Any,
        sample: "CalibrationSample",
    ) -> None:
        """Translate a Model API result into the right ``metric.update`` call."""
        from model_converter.metrics import (
            CocoDetectionMAP,
            MultilabelMAP,
            SemSegMIoU,
        )

        if isinstance(metric, MultilabelMAP):
            scores = getattr(result, "raw_scores", None)
            if scores is None:
                return
            gt = np.zeros(metric.num_labels, dtype=np.int64)
            if 0 <= sample.label < metric.num_labels:
                gt[sample.label] = 1
            metric.update(prediction=np.asarray(scores, dtype=np.float32), ground_truth=gt)
            return

        if isinstance(metric, CocoDetectionMAP):
            if sample.image_id is None:
                return
            if metric.iou_type == "bbox":
                self._feed_bbox_predictions(metric, result, sample)
            return

        if isinstance(metric, SemSegMIoU):
            if sample.mask_path is None:
                return
            gt_raw = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_raw is None:
                return
            # ADE20K convention: 0=unlabeled (ignore), 1..N=classes - shift down by 1.
            gt_mask = gt_raw.astype(np.int32) - 1
            gt_mask[gt_mask < 0] = metric.ignore_index
            pred_mask = getattr(result, "resultImage", result)
            pred_mask = np.asarray(pred_mask)
            if pred_mask.shape[:2] != gt_mask.shape[:2]:
                pred_mask = cv2.resize(
                    pred_mask,
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            metric.update(pred_mask, gt_mask)

    @staticmethod
    def _build_coco_prediction(
        image_id: Any,
        label: Any,
        bbox_xyxy: Any,
        score: Any,
    ) -> dict[str, Any]:
        """Build one COCO-format prediction dict from an xyxy box and 80-class label.

        The 0-79 ``label`` index is mapped to the original COCO 91-class category
        ID via :data:`COCO80_TO_COCO91`; out-of-range indices fall back to
        ``label + 1``. The ``[x_min, y_min, x_max, y_max]`` box is converted to
        COCO ``[x, y, w, h]`` format.
        """
        x_min, y_min, x_max, y_max = (float(v) for v in bbox_xyxy)
        n = int(label)
        coco_cat_id = COCO80_TO_COCO91[n] if n < len(COCO80_TO_COCO91) else n + 1
        return {
            "image_id": int(image_id),
            "category_id": coco_cat_id,
            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            "score": float(score),
        }

    @staticmethod
    def _feed_bbox_predictions(
        metric: "CocoDetectionMAP",
        result: Any,
        sample: "CalibrationSample",
    ) -> None:
        bboxes = getattr(result, "bboxes", None)
        labels = getattr(result, "labels", None)
        scores = getattr(result, "scores", None)
        if bboxes is None or labels is None or scores is None:
            return
        image_id = sample.image_id if sample.image_id is not None else 0
        preds = [
            BaseConverter._build_coco_prediction(image_id, label, bbox, score)
            for bbox, label, score in zip(bboxes, labels, scores)
        ]
        metric.update(predictions=preds)

    def quantize_model(
        self,
        model_path: Path,
        calibration_data: list[np.ndarray],
        model_config: dict[str, Any],
        preset: str = "accuracy",
        validation_data: list[np.ndarray] | None = None,
        validation_labels: list[int] | None = None,
        accuracy_results: AccuracyResults | None = None,
        validation_samples: "list[CalibrationSample] | None" = None,
        metric: "Metric | None" = None,
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
            validation_samples: Optional per-image GT pointers consumed by
                Model API-based metrics (multilabel mAP, COCO mAP, mIoU).
            metric: Optional task-specific :class:`Metric`. When supplied and
                not a :class:`TopOneAccuracy`, accuracy is measured via the
                Model API path using ``validation_samples`` instead of the
                preprocessed-tensor classification path.

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
            self._write_config_json(output_folder, quantized_model.get_rt_info(["model_info"]).value)

            # Validate accuracy if validation data provided.
            # Two paths: (1) Top-1 classification uses preprocessed validation_data + labels via raw
            # OpenVINO; (2) other task metrics (multilabel mAP, COCO mAP, mIoU) iterate raw image
            # paths through the Model API wrapper so per-task postprocessing is applied.
            from model_converter.metrics import TopOneAccuracy

            fp16_model_path = model_path.parent / f"{model_name}.xml"
            metric_path_active = (
                metric is not None and not isinstance(metric, TopOneAccuracy) and bool(validation_samples)
            )
            if metric_path_active:
                assert metric is not None  # guaranteed by metric_path_active
                assert validation_samples is not None  # guaranteed by metric_path_active
                metric_name = getattr(metric, "name", "metric")
                self.logger.info(f"Validating FP32 model {metric_name}...")
                fp32_metric = self._measure_metric(model_path, validation_samples, metric)
                self.logger.info(f"FP32 {metric_name}: {fp32_metric}")

                fp16_metric: float | None = None
                if fp16_model_path.exists():
                    self.logger.info(f"Validating FP16 model {metric_name}...")
                    fp16_metric = self._measure_metric(fp16_model_path, validation_samples, metric)
                    self.logger.info(f"FP16 {metric_name}: {fp16_metric}")
                else:
                    self.logger.warning(f"FP16 model not found for accuracy measurement: {fp16_model_path}")

                self.logger.info(f"Validating INT8 model {metric_name}...")
                int8_metric = self._measure_metric(output_path, validation_samples, metric)
                self.logger.info(f"INT8 {metric_name}: {int8_metric}")

                if accuracy_results is not None:
                    accuracy_results.fp32_accuracy = fp32_metric
                    accuracy_results.fp16_accuracy = fp16_metric
                    accuracy_results.int8_accuracy = int8_metric
                    accuracy_results.measured = True
            elif validation_data and validation_labels:
                self.logger.info("Validating FP32 model accuracy...")
                fp32_accuracy = self.validate_model(model_path, validation_data, validation_labels)
                self.logger.info(f"FP32 Top-1 Accuracy: {fp32_accuracy * 100:.2f}%")

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
            self._copy_gitattributes(output_folder)

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

    def _copy_gitattributes(self, output_folder: Path) -> None:
        """Copy the shared ``.gitattributes`` template into ``output_folder`` if present."""
        gitattributes_template = Path(__file__).parent.parent / "templates" / ".gitattributes"
        if gitattributes_template.exists():
            shutil.copy2(gitattributes_template, output_folder / ".gitattributes")
            self.logger.debug(f"Copied .gitattributes to: {output_folder}")

    @staticmethod
    def _write_config_json(output_folder: Path, model_info: Any) -> None:
        """Write ``model_info`` rt_info as ``config.json`` to track downloads."""
        with (output_folder / "config.json").open("w") as f:
            json.dump(model_info, f, indent=4)

    @staticmethod
    def _metadata_value(value: Any) -> str:
        """Convert config values to Model API rt_info string values."""
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value)
        return str(value)
