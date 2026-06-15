#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for YoloConverter — including COCO mAP accuracy measurement."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from model_converter.converters.yolo import _COCO80_TO_COCO91, YoloConverter
from model_converter.datasets.base import CalibrationSample
from model_converter.metrics.coco_detection import COCO80_TO_COCO91
from model_converter.reporting import AccuracyResults

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_yolo_converter(tmp_output_dir, tmp_cache_dir, *, measure_accuracy=True, registry=None):
    return YoloConverter(
        output_dir=tmp_output_dir,
        cache_dir=tmp_cache_dir,
        verbose=True,
        dataset_registry=registry,
        measure_accuracy=measure_accuracy,
    )


def _make_coco_dataset(root: Path) -> Path:
    """Create a minimal COCO-style dataset directory."""
    images_dir = root / "images"
    annotations_dir = root / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img_path = images_dir / "000000000001.jpg"
    cv2.imwrite(str(img_path), img)

    annotation = {
        "images": [{"id": 1, "file_name": "000000000001.jpg"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    (annotations_dir / "instances_val2017.json").write_text(json.dumps(annotation))
    return root


def _make_coco_sample(image_path: Path) -> CalibrationSample:
    return CalibrationSample(image_path=image_path, label=0, image_id=1)


# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------


class TestCoco80ToCoco91:
    def test_length_is_80(self):
        assert len(_COCO80_TO_COCO91) == 80

    def test_first_and_last_values(self):
        assert _COCO80_TO_COCO91[0] == 1
        assert _COCO80_TO_COCO91[-1] == 90

    def test_no_duplicates(self):
        assert len(set(_COCO80_TO_COCO91)) == 80

    def test_alias_matches_canonical(self):
        assert _COCO80_TO_COCO91 == COCO80_TO_COCO91

    def test_class_11_maps_to_13_not_12(self):
        """Stop sign (class 11) maps to COCO category 13, not 12 (missing category)."""
        assert _COCO80_TO_COCO91[11] == 13


# ---------------------------------------------------------------------------
# _measure_original_accuracy
# ---------------------------------------------------------------------------


class TestMeasureOriginalAccuracy:
    """Unit tests for YoloConverter._measure_original_accuracy."""

    @pytest.fixture
    def converter(self, tmp_path):
        return _make_yolo_converter(tmp_path / "out", tmp_path / "cache")

    def test_returns_map_on_success(self, converter, tmp_path):
        """Returns a float mAP when inference and evaluation succeed."""
        coco_root = _make_coco_dataset(tmp_path / "coco")
        annotation_file = coco_root / "annotations" / "instances_val2017.json"
        img_path = coco_root / "images" / "000000000001.jpg"
        sample = _make_coco_sample(img_path)

        fake_result = MagicMock()
        fake_result.boxes.xyxy.cpu.return_value.tolist.return_value = [[5.0, 5.0, 25.0, 25.0]]
        fake_result.boxes.cls.cpu.return_value.tolist.return_value = [0.0]
        fake_result.boxes.conf.cpu.return_value.tolist.return_value = [0.9]

        fake_model = MagicMock(return_value=[fake_result])

        with patch("model_converter.converters.yolo.CocoDetectionMAP") as mock_metric_cls:
            mock_metric = MagicMock()
            mock_metric.compute.return_value = 0.42
            mock_metric_cls.return_value = mock_metric

            with patch("ultralytics.YOLO", return_value=fake_model):
                result = converter._measure_original_accuracy(
                    pt_model_path=tmp_path / "cache" / "yolo11n.pt",
                    samples=[sample],
                    annotation_file=annotation_file,
                )

        assert result == pytest.approx(0.42)
        mock_metric.update.assert_called_once()
        preds = mock_metric.update.call_args[1]["predictions"]
        assert len(preds) == 1
        assert preds[0]["image_id"] == 1
        assert preds[0]["category_id"] == _COCO80_TO_COCO91[0]  # class 0 → COCO cat 1

    def test_returns_none_when_ultralytics_import_fails(self, converter, tmp_path):
        """Returns None gracefully when Ultralytics is not importable."""
        coco_root = _make_coco_dataset(tmp_path / "coco")
        annotation_file = coco_root / "annotations" / "instances_val2017.json"
        sample = _make_coco_sample(coco_root / "images" / "000000000001.jpg")

        with patch("ultralytics.YOLO", side_effect=ImportError("no ultralytics")):
            result = converter._measure_original_accuracy(
                pt_model_path=tmp_path / "cache" / "yolo11n.pt",
                samples=[sample],
                annotation_file=annotation_file,
            )

        assert result is None

    def test_returns_none_when_pt_file_not_found(self, converter, tmp_path):
        """Returns None gracefully when the .pt weights file is absent."""
        coco_root = _make_coco_dataset(tmp_path / "coco")
        annotation_file = coco_root / "annotations" / "instances_val2017.json"
        sample = _make_coco_sample(coco_root / "images" / "000000000001.jpg")

        with patch("ultralytics.YOLO", side_effect=FileNotFoundError("not found")):
            result = converter._measure_original_accuracy(
                pt_model_path=tmp_path / "cache" / "missing.pt",
                samples=[sample],
                annotation_file=annotation_file,
            )

        assert result is None

    def test_skips_samples_without_image_id(self, converter, tmp_path):
        """Samples with image_id=None are silently skipped."""
        coco_root = _make_coco_dataset(tmp_path / "coco")
        annotation_file = coco_root / "annotations" / "instances_val2017.json"
        img_path = coco_root / "images" / "000000000001.jpg"
        sample_no_id = CalibrationSample(image_path=img_path, label=0, image_id=None)

        fake_model = MagicMock()

        with patch("model_converter.converters.yolo.CocoDetectionMAP") as mock_metric_cls:
            mock_metric = MagicMock()
            mock_metric.compute.return_value = 0.0
            mock_metric_cls.return_value = mock_metric

            with patch("ultralytics.YOLO", return_value=fake_model):
                converter._measure_original_accuracy(
                    pt_model_path=tmp_path / "cache" / "yolo11n.pt",
                    samples=[sample_no_id],
                    annotation_file=annotation_file,
                )

        fake_model.assert_not_called()

    def test_skips_unreadable_images(self, converter, tmp_path):
        """Samples whose image file cannot be read are silently skipped."""
        coco_root = _make_coco_dataset(tmp_path / "coco")
        annotation_file = coco_root / "annotations" / "instances_val2017.json"
        sample = CalibrationSample(
            image_path=tmp_path / "does_not_exist.jpg",
            label=0,
            image_id=1,
        )

        fake_model = MagicMock()

        with patch("model_converter.converters.yolo.CocoDetectionMAP") as mock_metric_cls:
            mock_metric = MagicMock()
            mock_metric.compute.return_value = 0.0
            mock_metric_cls.return_value = mock_metric

            with patch("ultralytics.YOLO", return_value=fake_model):
                converter._measure_original_accuracy(
                    pt_model_path=tmp_path / "cache" / "yolo11n.pt",
                    samples=[sample],
                    annotation_file=annotation_file,
                )

        fake_model.assert_not_called()

    def test_skips_failed_inference_samples(self, converter, tmp_path):
        """Per-sample inference errors are swallowed; remaining samples still processed."""
        coco_root = _make_coco_dataset(tmp_path / "coco")
        annotation_file = coco_root / "annotations" / "instances_val2017.json"
        img_path = coco_root / "images" / "000000000001.jpg"
        sample = _make_coco_sample(img_path)

        fake_model = MagicMock(side_effect=RuntimeError("inference exploded"))

        with patch("model_converter.converters.yolo.CocoDetectionMAP") as mock_metric_cls:
            mock_metric = MagicMock()
            mock_metric.compute.return_value = 0.0
            mock_metric_cls.return_value = mock_metric

            with patch("ultralytics.YOLO", return_value=fake_model):
                result = converter._measure_original_accuracy(
                    pt_model_path=tmp_path / "cache" / "yolo11n.pt",
                    samples=[sample],
                    annotation_file=annotation_file,
                )

        assert result == pytest.approx(0.0)
        mock_metric.update.assert_called_once_with(predictions=[])

    def test_uses_fallback_category_id_for_out_of_range_cls(self, converter, tmp_path):
        """Class indices >= 80 fall back to idx+1 instead of the lookup table."""
        coco_root = _make_coco_dataset(tmp_path / "coco")
        annotation_file = coco_root / "annotations" / "instances_val2017.json"
        img_path = coco_root / "images" / "000000000001.jpg"
        sample = _make_coco_sample(img_path)

        fake_result = MagicMock()
        fake_result.boxes.xyxy.cpu.return_value.tolist.return_value = [[0.0, 0.0, 10.0, 10.0]]
        fake_result.boxes.cls.cpu.return_value.tolist.return_value = [99.0]  # out of range
        fake_result.boxes.conf.cpu.return_value.tolist.return_value = [0.5]
        fake_model = MagicMock(return_value=[fake_result])

        with patch("model_converter.converters.yolo.CocoDetectionMAP") as mock_metric_cls:
            mock_metric = MagicMock()
            mock_metric.compute.return_value = 0.0
            mock_metric_cls.return_value = mock_metric

            with patch("ultralytics.YOLO", return_value=fake_model):
                converter._measure_original_accuracy(
                    pt_model_path=tmp_path / "cache" / "yolo11n.pt",
                    samples=[sample],
                    annotation_file=annotation_file,
                )

        preds = mock_metric.update.call_args[1]["predictions"]
        assert preds[0]["category_id"] == 100  # 99 + 1 fallback


# ---------------------------------------------------------------------------
# _measure_yolo_accuracy
# ---------------------------------------------------------------------------


class TestMeasureYoloAccuracy:
    """Unit tests for YoloConverter._measure_yolo_accuracy."""

    @pytest.fixture
    def coco_dataset(self, tmp_path):
        return _make_coco_dataset(tmp_path / "coco")

    @pytest.fixture
    def mock_registry(self, coco_dataset):
        mock = MagicMock()
        mock.resolve_from_config.return_value = coco_dataset
        return mock

    @pytest.fixture
    def converter(self, tmp_path, mock_registry):
        return _make_yolo_converter(tmp_path / "out", tmp_path / "cache", registry=mock_registry)

    @pytest.fixture
    def yolo_config(self):
        return {
            "model_short_name": "YOLO11n",
            "model_full_name": "YOLO11n",
            "model_library": "yolo",
            "yolo_version": "yolo11n",
            "model_type": "YOLO11",
            "dataset_type": "coco-detection",
            "license": "agpl-3.0",
            "license_link": "https://spdx.org/licenses/AGPL-3.0-only.html",
        }

    def test_returns_accuracy_results_on_success(self, converter, yolo_config, coco_dataset, tmp_path):
        """Full happy path: all three mAP values are populated."""
        fp16_folder = tmp_path / "out" / "YOLO11n-fp16-ov"
        int8_folder = tmp_path / "out" / "YOLO11n-int8-ov"
        fp16_folder.mkdir(parents=True)
        int8_folder.mkdir(parents=True)
        (fp16_folder / "yolo11n.xml").write_text("<net/>")
        (int8_folder / "yolo11n.xml").write_text("<net/>")

        with (
            patch.object(converter, "_collect_validation_samples") as mock_samples,
            patch.object(converter, "_metric_for_config") as mock_metric_fn,
            patch.object(converter, "_measure_metric") as mock_measure,
            patch.object(converter, "_measure_original_accuracy") as mock_original,
        ):
            mock_samples.return_value = [_make_coco_sample(coco_dataset / "images" / "000000000001.jpg")]
            mock_metric = MagicMock()
            mock_metric.name = "mAP"
            mock_metric_fn.return_value = mock_metric
            mock_original.return_value = 0.50
            mock_measure.side_effect = [0.48, 0.45]  # fp16, int8

            result = converter._measure_yolo_accuracy(yolo_config, "yolo11n", fp16_folder, int8_folder)

        assert result is not None
        assert result.measured is True
        assert result.original_accuracy == pytest.approx(0.50)
        assert result.fp16_accuracy == pytest.approx(0.48)
        assert result.int8_accuracy == pytest.approx(0.45)
        assert result.metric_name == "mAP"

    def test_returns_none_when_dataset_path_missing(self, tmp_path, yolo_config):
        """Returns None when dataset registry resolves to a non-existent path."""
        mock_registry = MagicMock()
        mock_registry.resolve_from_config.return_value = tmp_path / "nonexistent"
        conv = _make_yolo_converter(tmp_path / "out", tmp_path / "cache", registry=mock_registry)

        result = conv._measure_yolo_accuracy(
            yolo_config,
            "yolo11n",
            tmp_path / "fp16",
            tmp_path / "int8",
        )

        assert result is None

    def test_returns_none_when_dataset_path_is_none(self, tmp_path, yolo_config):
        """Returns None when dataset registry returns None."""
        mock_registry = MagicMock()
        mock_registry.resolve_from_config.return_value = None
        conv = _make_yolo_converter(tmp_path / "out", tmp_path / "cache", registry=mock_registry)

        result = conv._measure_yolo_accuracy(
            yolo_config,
            "yolo11n",
            tmp_path / "fp16",
            tmp_path / "int8",
        )

        assert result is None

    def test_returns_none_when_dataset_type_not_coco(
        self,
        tmp_path,
        coco_dataset,
        mock_registry,
    ):
        """Returns None for non-COCO dataset types (no annotation file entry)."""
        conv = _make_yolo_converter(tmp_path / "out", tmp_path / "cache", registry=mock_registry)
        config = {
            "model_short_name": "YOLO11n",
            "model_type": "YOLO11",
            "dataset_type": "imagenet-1k",  # not in _COCO_ANNOTATION_FILES
        }

        result = conv._measure_yolo_accuracy(config, "yolo11n", tmp_path / "fp16", tmp_path / "int8")

        assert result is None

    def test_returns_none_when_annotation_file_missing(self, tmp_path, yolo_config):
        """Returns None when the COCO annotation JSON file is absent."""
        coco_root = tmp_path / "coco"
        (coco_root / "images").mkdir(parents=True)
        # No annotations/ directory created
        mock_registry = MagicMock()
        mock_registry.resolve_from_config.return_value = coco_root
        conv = _make_yolo_converter(tmp_path / "out", tmp_path / "cache", registry=mock_registry)

        result = conv._measure_yolo_accuracy(
            yolo_config,
            "yolo11n",
            tmp_path / "fp16",
            tmp_path / "int8",
        )

        assert result is None

    def test_returns_none_when_no_validation_samples(
        self,
        converter,
        yolo_config,
        tmp_path,
    ):
        """Returns None when _collect_validation_samples returns an empty list."""
        with patch.object(converter, "_collect_validation_samples", return_value=[]):
            result = converter._measure_yolo_accuracy(
                yolo_config,
                "yolo11n",
                tmp_path / "fp16",
                tmp_path / "int8",
            )

        assert result is None

    def test_returns_none_when_metric_is_none(self, converter, yolo_config, coco_dataset, tmp_path):
        """Returns None when _metric_for_config cannot produce a metric."""
        sample = _make_coco_sample(coco_dataset / "images" / "000000000001.jpg")
        with (
            patch.object(converter, "_collect_validation_samples", return_value=[sample]),
            patch.object(converter, "_metric_for_config", return_value=None),
        ):
            result = converter._measure_yolo_accuracy(
                yolo_config,
                "yolo11n",
                tmp_path / "fp16",
                tmp_path / "int8",
            )

        assert result is None

    def test_skips_fp16_when_xml_missing(self, converter, yolo_config, coco_dataset, tmp_path):
        """FP16 accuracy is not measured when the FP16 XML file does not exist."""
        fp16_folder = tmp_path / "fp16"  # no XML inside
        int8_folder = tmp_path / "int8"
        fp16_folder.mkdir()
        int8_folder.mkdir()
        (int8_folder / "yolo11n.xml").write_text("<net/>")

        sample = _make_coco_sample(coco_dataset / "images" / "000000000001.jpg")
        mock_metric = MagicMock()
        mock_metric.name = "mAP"

        with (
            patch.object(converter, "_collect_validation_samples", return_value=[sample]),
            patch.object(converter, "_metric_for_config", return_value=mock_metric),
            patch.object(converter, "_measure_original_accuracy", return_value=0.5),
            patch.object(converter, "_measure_metric", return_value=0.44) as mock_measure,
        ):
            result = converter._measure_yolo_accuracy(yolo_config, "yolo11n", fp16_folder, int8_folder)

        assert result is not None
        assert result.fp16_accuracy is None
        # _measure_metric called once for INT8 only
        assert mock_measure.call_count == 1

    def test_skips_int8_when_xml_missing(self, converter, yolo_config, coco_dataset, tmp_path):
        """INT8 accuracy is not measured when the INT8 XML file does not exist."""
        fp16_folder = tmp_path / "fp16"
        int8_folder = tmp_path / "int8"  # no XML inside
        fp16_folder.mkdir()
        int8_folder.mkdir()
        (fp16_folder / "yolo11n.xml").write_text("<net/>")

        sample = _make_coco_sample(coco_dataset / "images" / "000000000001.jpg")
        mock_metric = MagicMock()
        mock_metric.name = "mAP"

        with (
            patch.object(converter, "_collect_validation_samples", return_value=[sample]),
            patch.object(converter, "_metric_for_config", return_value=mock_metric),
            patch.object(converter, "_measure_original_accuracy", return_value=0.5),
            patch.object(converter, "_measure_metric", return_value=0.48) as mock_measure,
        ):
            result = converter._measure_yolo_accuracy(yolo_config, "yolo11n", fp16_folder, int8_folder)

        assert result is not None
        assert result.int8_accuracy is None
        # _measure_metric called once for FP16 only
        assert mock_measure.call_count == 1


# ---------------------------------------------------------------------------
# process_model_config — accuracy integration
# ---------------------------------------------------------------------------


class TestProcessModelConfigAccuracy:
    """Integration-style tests verifying accuracy is wired into process_model_config."""

    @pytest.fixture
    def yolo_config(self):
        return {
            "model_short_name": "YOLO11n",
            "model_full_name": "YOLO11n",
            "model_library": "yolo",
            "yolo_version": "yolo11n",
            "model_type": "YOLO11",
            "dataset_type": "coco-detection",
            "license": "agpl-3.0",
            "license_link": "https://spdx.org/licenses/AGPL-3.0-only.html",
        }

    def _stub_ultralytics_export(self, cache_dir: Path, model_short_name: str, yolo_version: str) -> MagicMock:
        """Return a fake YOLO model that fakes both FP16 and INT8 exports."""
        fp16_export = cache_dir / f"{yolo_version}_openvino_model"
        int8_export = cache_dir / f"{yolo_version}_int8_openvino_model"
        fp16_export.mkdir(parents=True)
        int8_export.mkdir(parents=True)
        (fp16_export / f"{yolo_version}.xml").write_text("<net/>")
        (int8_export / f"{yolo_version}.xml").write_text("<net/>")

        fake_model = MagicMock()

        def fake_export(format, **kwargs):
            pass

        fake_model.export.side_effect = fake_export
        return fake_model

    def test_accuracy_measured_when_flag_true_and_dataset_available(
        self,
        tmp_path,
        yolo_config,
    ):
        """process_model_config calls _measure_yolo_accuracy when measure_accuracy=True."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        conv = _make_yolo_converter(output_dir, cache_dir)
        fake_model = self._stub_ultralytics_export(cache_dir, "YOLO11n", "yolo11n")
        fake_accuracy = AccuracyResults()
        fake_accuracy.original_accuracy = 0.5
        fake_accuracy.fp16_accuracy = 0.48
        fake_accuracy.int8_accuracy = 0.45
        fake_accuracy.measured = True

        with (
            patch("ultralytics.YOLO", return_value=fake_model),
            patch.object(conv, "_update_model_type_in_xml"),
            patch.object(conv, "_copy_yolo_readme"),
            patch.object(conv, "_measure_yolo_accuracy", return_value=fake_accuracy) as mock_acc,
        ):
            success = conv.process_model_config(yolo_config)

        assert success is True
        mock_acc.assert_called_once()
        assert len(conv.results) == 1
        result = conv.results[0]
        assert result.original_accuracy == pytest.approx(0.5)
        assert result.fp16_accuracy == pytest.approx(0.48)
        assert result.int8_accuracy == pytest.approx(0.45)

    def test_accuracy_skipped_when_measure_accuracy_false(self, tmp_path, yolo_config):
        """process_model_config skips accuracy when measure_accuracy=False."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        conv = _make_yolo_converter(output_dir, cache_dir, measure_accuracy=False)
        fake_model = self._stub_ultralytics_export(cache_dir, "YOLO11n", "yolo11n")

        with (
            patch("ultralytics.YOLO", return_value=fake_model),
            patch.object(conv, "_update_model_type_in_xml"),
            patch.object(conv, "_copy_yolo_readme"),
            patch.object(conv, "_measure_yolo_accuracy") as mock_acc,
        ):
            success = conv.process_model_config(yolo_config)

        assert success is True
        mock_acc.assert_not_called()
        result = conv.results[0]
        assert result.original_accuracy is None

    def test_accuracy_skipped_when_int8_not_produced(self, tmp_path, yolo_config):
        """process_model_config skips accuracy when INT8 XML is missing."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        conv = _make_yolo_converter(output_dir, cache_dir)

        # Only produce FP16 export, not INT8
        fp16_export = cache_dir / "yolo11n_openvino_model"
        fp16_export.mkdir(parents=True)
        (fp16_export / "yolo11n.xml").write_text("<net/>")

        fake_model = MagicMock()
        fake_model.export.return_value = None

        with (
            patch("ultralytics.YOLO", return_value=fake_model),
            patch.object(conv, "_update_model_type_in_xml"),
            patch.object(conv, "_copy_yolo_readme"),
            patch.object(conv, "_measure_yolo_accuracy") as mock_acc,
        ):
            success = conv.process_model_config(yolo_config)

        assert success is True
        mock_acc.assert_not_called()

    def test_skipped_models_do_not_measure_accuracy(self, tmp_path, yolo_config):
        """process_model_config skips accuracy for already-converted models."""
        cache_dir = tmp_path / "cache"
        output_dir = tmp_path / "out"
        fp16_folder = output_dir / "YOLO11n-fp16-ov"
        int8_folder = output_dir / "YOLO11n-int8-ov"
        fp16_folder.mkdir(parents=True)
        int8_folder.mkdir(parents=True)
        (fp16_folder / "yolo11n.xml").write_text("<net/>")
        (int8_folder / "yolo11n.xml").write_text("<net/>")

        conv = _make_yolo_converter(output_dir, cache_dir)

        with patch.object(conv, "_measure_yolo_accuracy") as mock_acc:
            success = conv.process_model_config(yolo_config)

        assert success is True
        mock_acc.assert_not_called()

    def test_failed_conversion_does_not_measure_accuracy(self, tmp_path, yolo_config):
        """process_model_config records failure without measuring accuracy."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        conv = _make_yolo_converter(output_dir, cache_dir)

        with (
            patch("ultralytics.YOLO", side_effect=RuntimeError("export failed")),
            patch.object(conv, "_measure_yolo_accuracy") as mock_acc,
        ):
            success = conv.process_model_config(yolo_config)

        assert success is False
        mock_acc.assert_not_called()
        assert conv.results[0].original_accuracy is None
