#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""YOLO model converter."""

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
from defusedxml.ElementTree import ParseError, parse

from model_converter.converters.base import BaseConverter
from model_converter.metrics.coco_detection import COCO80_TO_COCO91, CocoDetectionMAP
from model_converter.reporting import AccuracyResults

if TYPE_CHECKING:
    from model_converter.datasets import CalibrationSample

MODEL_VERSIONS = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]


# Re-export for backward compatibility (the canonical definition lives in metrics/coco_detection.py).
_COCO80_TO_COCO91 = COCO80_TO_COCO91


class YoloConverter(BaseConverter):
    """Converter for Ultralytics YOLO models.

    Uses the Ultralytics library to export YOLO models to OpenVINO format,
    then repackages and adds metadata.
    """

    def process_model_config(self, config: dict[str, Any]) -> bool:
        """Process a YOLO model configuration.

        Args:
            config: Model configuration dictionary with 'yolo_version' field

        Returns:
            True if successful, False otherwise
        """
        model_short_name = config.get("model_short_name", "unknown")
        yolo_version = config.get("yolo_version", model_short_name)

        # Check if both FP16 and INT8 models already exist
        fp16_folder = self.output_dir / f"{model_short_name}-fp16-ov"
        int8_folder = self.output_dir / f"{model_short_name}-int8-ov"

        if self._skip_if_already_converted(config, model_short_name, xml_stem=yolo_version):
            return True

        try:
            from ultralytics import YOLO

            self.logger.info("=" * 80)
            self.logger.info(f"Processing YOLO model: {model_short_name}")
            self.logger.info("=" * 80)

            yolo_size = yolo_version[-1]  # n, s, m, l, or x

            # Load model from cache directory to avoid polluting the working directory
            model_path = self.cache_dir / f"{yolo_version}.pt"
            model = YOLO(str(model_path))

            # Export regular OpenVINO model (FP16)
            self.logger.info(f"Exporting {yolo_version} to OpenVINO FP16 format...")
            model.export(format="openvino", half=True)

            # Ultralytics exports next to the .pt file in the cache directory
            old_name = self.cache_dir / f"{yolo_version}_openvino_model"
            fp16_folder.mkdir(parents=True, exist_ok=True)
            if old_name.exists():
                if fp16_folder.exists():
                    shutil.rmtree(fp16_folder)
                old_name.rename(fp16_folder)

                # Update model_type in XML metadata
                xml_file = fp16_folder / f"{yolo_version}.xml"
                if xml_file.exists():
                    self._update_model_type_in_xml(xml_file, "YOLO11")

                # Copy README template for fp16
                self._copy_yolo_readme("README-yolo-fp16.md", fp16_folder, yolo_size)

            # Export INT8 quantized model
            self.logger.info(f"Exporting {yolo_version} to OpenVINO INT8 format...")
            model.export(format="openvino", int8=True, data="coco128.yaml")

            # Rename output folder for INT8
            old_name_int8 = self.cache_dir / f"{yolo_version}_int8_openvino_model"
            if old_name_int8.exists():
                if int8_folder.exists():
                    shutil.rmtree(int8_folder)
                old_name_int8.rename(int8_folder)

                # Update model_type in XML metadata
                xml_file = int8_folder / f"{yolo_version}.xml"
                if xml_file.exists():
                    self._update_model_type_in_xml(xml_file, "YOLO11")

                # Copy README template for int8
                self._copy_yolo_readme("README-yolo-int8.md", int8_folder, yolo_size)

            self.logger.info(f"✓ Successfully converted {model_short_name}")
            quantized = (int8_folder / f"{yolo_version}.xml").exists()
            accuracy: AccuracyResults | None = None
            if self.measure_accuracy and quantized:
                accuracy = self._measure_yolo_accuracy(config, yolo_version, fp16_folder, int8_folder)
            self._record_result(self._build_result(config), converted=True, quantized=quantized, accuracy=accuracy)
            return True

        except (ValueError, RuntimeError, ImportError, FileNotFoundError, OSError) as e:
            return self._record_failure(config, model_short_name, e, label="YOLO model")

    def _measure_yolo_accuracy(
        self,
        config: dict[str, Any],
        yolo_version: str,
        fp16_folder: Path,
        int8_folder: Path,
    ) -> AccuracyResults | None:
        """Measure original PT model, FP16 OV, and INT8 OV mAP on the COCO validation subset.

        Uses the same 500-image COCO subset for all three measurements so the
        numbers in the report are directly comparable.

        Args:
            config: Model configuration dictionary (must contain ``dataset_type``).
            yolo_version: Ultralytics model identifier (e.g. ``"yolo11n"``).
            fp16_folder: Directory containing the exported FP16 OpenVINO model.
            int8_folder: Directory containing the exported INT8 OpenVINO model.

        Returns:
            Populated :class:`AccuracyResults`, or ``None`` when the dataset or
            metric is unavailable.
        """
        from model_converter.datasets.factory import _COCO_ANNOTATION_FILES

        dataset_path = self._resolve_dataset_path(config)
        if dataset_path is None or not dataset_path.exists():
            self.logger.warning("COCO dataset not available — skipping accuracy measurement for YOLO")
            return None

        dataset_type = config.get("dataset_type")
        if dataset_type not in _COCO_ANNOTATION_FILES:
            self.logger.warning(f"Unsupported dataset_type {dataset_type!r} — skipping accuracy measurement")
            return None

        annotation_file = dataset_path / "annotations" / _COCO_ANNOTATION_FILES[dataset_type]
        if not annotation_file.exists():
            self.logger.warning(f"COCO annotation file not found: {annotation_file}")
            return None

        samples = self._collect_validation_samples(dataset_path, dataset_type, subset_size=500)
        if not samples:
            self.logger.warning("No validation samples found — skipping accuracy measurement for YOLO")
            return None

        metric = self._metric_for_config(config, dataset_path)
        if metric is None:
            self.logger.warning("No metric available for this config — skipping accuracy measurement for YOLO")
            return None

        accuracy = AccuracyResults()
        accuracy.metric_name = metric.name

        # Original PT model accuracy — run Ultralytics native inference on each sample.
        pt_model_path = self.cache_dir / f"{yolo_version}.pt"
        original_map = self._measure_original_accuracy(pt_model_path, samples, annotation_file)
        accuracy.original_accuracy = original_map

        # FP16 OV model accuracy via Model API.
        fp16_model_path = fp16_folder / f"{yolo_version}.xml"
        if fp16_model_path.exists():
            self.logger.info("Measuring FP16 model mAP...")
            accuracy.fp16_accuracy = self._measure_metric(fp16_model_path, samples, metric)
            self.logger.info(f"FP16 mAP: {accuracy.fp16_accuracy}")

        # INT8 OV model accuracy via Model API.
        int8_model_path = int8_folder / f"{yolo_version}.xml"
        if int8_model_path.exists():
            self.logger.info("Measuring INT8 model mAP...")
            accuracy.int8_accuracy = self._measure_metric(int8_model_path, samples, metric)
            self.logger.info(f"INT8 mAP: {accuracy.int8_accuracy}")

        accuracy.measured = True
        return accuracy

    def _measure_original_accuracy(
        self,
        pt_model_path: Path,
        samples: "list[CalibrationSample]",
        annotation_file: Path,
    ) -> float | None:
        """Measure mAP of the original YOLO PT model using direct Ultralytics inference.

        Runs the ``.pt`` model on each COCO sample, converts Ultralytics 0-79
        class indices to COCO 91-class category IDs via :data:`COCO80_TO_COCO91`,
        and evaluates with :class:`CocoDetectionMAP`.

        Args:
            pt_model_path: Path to the Ultralytics ``.pt`` weights file.
            samples: COCO validation samples with ``image_path`` and ``image_id``.
            annotation_file: Path to the COCO ``instances_val2017.json`` file.

        Returns:
            mAP@IoU=0.50:0.95, or ``None`` on failure.
        """
        try:
            from ultralytics import YOLO

            model = YOLO(str(pt_model_path))
        except (ImportError, FileNotFoundError, RuntimeError) as e:
            self.logger.error(f"Failed to load PT model for original accuracy measurement: {e}")
            return None

        metric = CocoDetectionMAP(annotation_file=annotation_file, iou_type="bbox")
        predictions: list[dict[str, Any]] = []

        for sample in samples:
            if sample.image_id is None:
                continue
            img = cv2.imread(str(sample.image_path))
            if img is None:
                continue
            try:
                results = model(img, verbose=False)[0]
            except (RuntimeError, TypeError, ValueError) as e:
                self.logger.debug(f"PT inference failed for {sample.image_path}: {e}")
                continue

            boxes_xyxy = results.boxes.xyxy.cpu().tolist()
            cls_ids = results.boxes.cls.cpu().tolist()
            scores = results.boxes.conf.cpu().tolist()

            for (x1, y1, x2, y2), cls_idx, score in zip(boxes_xyxy, cls_ids, scores):
                predictions.append(
                    self._build_coco_prediction(sample.image_id, cls_idx, (x1, y1, x2, y2), score),
                )

        metric.update(predictions=predictions)
        original_map = float(metric.compute())
        self.logger.info(f"Original PT model mAP: {original_map}")
        return original_map

    def _copy_yolo_readme(self, template_name: str, dest_dir: Path, yolo_size: str) -> None:
        """Copy a YOLO README template, replacing <<yolo_size>>."""
        template_path = Path(__file__).parent.parent / "templates" / template_name
        if not template_path.exists():
            self.logger.warning(f"YOLO README template not found: {template_path}")
            return
        content = template_path.read_text()
        content = content.replace("<<yolo_size>>", yolo_size)
        (dest_dir / "README.md").write_text(content)
        self.logger.debug(f"Copied {template_name} -> {dest_dir / 'README.md'} (size={yolo_size})")

    def _update_model_type_in_xml(self, xml_path: Path, model_type: str = "YOLO11") -> None:
        """Update the model_type value in the OpenVINO XML file."""
        try:
            tree = parse(xml_path)
            root = tree.getroot()

            for rt_info in root.findall(".//rt_info"):
                for model_info in rt_info.findall(".//model_info"):
                    for model_type_elem in model_info.findall(".//model_type"):
                        model_type_elem.set("value", model_type)

            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            self.logger.debug(f"Updated model_type to {model_type} in {xml_path}")

            model_info_dict = {}
            for rt_info in root.findall(".//rt_info"):
                for model_info in rt_info.findall(".//model_info"):
                    for child in model_info:
                        model_info_dict[child.tag] = child.attrib["value"]
            with (xml_path.parent / "config.json").open("w") as f:
                json.dump(model_info_dict, f, indent=4)

        except (OSError, ParseError) as error:
            self.logger.warning(f"Failed to update {xml_path}: {error}")
