#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for shared converter helpers via TorchvisionConverter."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from model_converter.converters.torchvision import TorchvisionConverter


def _set_template_root(monkeypatch: pytest.MonkeyPatch, template_dir: Path) -> None:
    """Redirect BaseConverter template lookup to a test template directory."""
    import model_converter.converters.base as base_module

    monkeypatch.setattr(base_module, "__file__", str(template_dir.parent / "converters" / "base.py"))


class TestCopyReadme:
    """Tests for BaseConverter.copy_readme via TorchvisionConverter."""

    def test_renders_placeholders_and_tags_yaml(self, converter, template_dir, tmp_path, monkeypatch):
        """copy_readme fills placeholders from config and renders tags as YAML."""
        _set_template_root(monkeypatch, template_dir)
        (template_dir / "README-torchvision-fp16.md").write_text(
            "# <<model_short_name>>\n"
            "License: <<license>>\n"
            "License link: <<license_link>>\n"
            "Docs: <<docs>>\n"
            "Variant: <<variant>>\n"
            "Tags:\n"
            "<<tags_yaml>>",
        )
        output_folder = tmp_path / "test_model-fp16-ov"
        output_folder.mkdir()

        converter.copy_readme(
            {
                "model_short_name": "test_model",
                "model_library": "torchvision",
                "license": "Apache-2.0",
                "license_link": "https://apache.org/licenses/LICENSE-2.0",
                "docs": "https://docs.example.com",
                "tags": ["vision", "classification"],
            },
            output_folder,
            variant="fp16",
        )

        content = (output_folder / "README.md").read_text()
        assert "# test_model" in content
        assert "License: Apache-2.0" in content
        assert "License link: https://apache.org/licenses/LICENSE-2.0" in content
        assert "Docs: https://docs.example.com" in content
        assert "Variant: fp16" in content
        assert "  - vision" in content
        assert "  - classification" in content

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            (
                {"model_short_name": "", "license": "MIT", "license_link": "https://mit.edu"},
                "non-empty model_short_name",
            ),
            ({"model_short_name": "test_model", "license": "MIT", "license_link": ""}, "non-empty license_link"),
            ({"model_short_name": "test_model", "license": "", "license_link": "https://mit.edu"}, "non-empty license"),
        ],
    )
    def test_logs_validation_failures(self, converter, tmp_path, caplog, config, expected):
        """copy_readme logs warnings when required metadata is missing."""
        output_folder = tmp_path / "test_model-fp16-ov"
        output_folder.mkdir()

        converter.copy_readme(config, output_folder)

        assert expected in caplog.text
        assert not (output_folder / "README.md").exists()

    def test_missing_docs_logs_warning_and_still_writes_readme(
        self,
        converter,
        template_dir,
        tmp_path,
        monkeypatch,
        caplog,
    ):
        """copy_readme leaves the docs placeholder empty when docs are omitted."""
        _set_template_root(monkeypatch, template_dir)
        (template_dir / "README-torchvision-fp16.md").write_text("Docs: <<docs>>")
        output_folder = tmp_path / "test_model-fp16-ov"
        output_folder.mkdir()

        converter.copy_readme(
            {
                "model_short_name": "test_model",
                "model_library": "torchvision",
                "license": "Apache-2.0",
                "license_link": "https://apache.org/licenses/LICENSE-2.0",
            },
            output_folder,
            variant="fp16",
        )

        assert "does not define 'docs' field" in caplog.text
        assert (output_folder / "README.md").read_text() == "Docs: "

    def test_logs_warning_when_template_not_found(self, converter, tmp_path, monkeypatch, caplog):
        """copy_readme logs a warning and does nothing when the template file doesn't exist."""
        import model_converter.converters.base as base_module

        empty_templates = tmp_path / "empty_templates"
        empty_templates.mkdir()
        monkeypatch.setattr(base_module, "__file__", str(empty_templates.parent / "converters" / "base.py"))

        output_folder = tmp_path / "test_model-fp16-ov"
        output_folder.mkdir()

        converter.copy_readme(
            {
                "model_short_name": "test_model",
                "model_library": "nonexistent_library",
                "license": "Apache-2.0",
                "license_link": "https://apache.org/licenses/LICENSE-2.0",
            },
            output_folder,
            variant="fp16",
        )

        assert "README template not found" in caplog.text
        assert not (output_folder / "README.md").exists()

    def test_skips_none_config_values_in_placeholders(self, converter, template_dir, tmp_path, monkeypatch):
        """copy_readme skips None values in model_config without crashing."""
        _set_template_root(monkeypatch, template_dir)
        (template_dir / "README-torchvision-fp16.md").write_text("<<model_short_name>>")
        output_folder = tmp_path / "test_model-fp16-ov"
        output_folder.mkdir()

        converter.copy_readme(
            {
                "model_short_name": "test_model",
                "model_library": "torchvision",
                "license": "Apache-2.0",
                "license_link": "https://apache.org/licenses/LICENSE-2.0",
                "some_field": None,
            },
            output_folder,
            variant="fp16",
        )

        content = (output_folder / "README.md").read_text()
        assert "test_model" in content


class TestCollectDatasetEntries:
    """Tests for BaseConverter._collect_dataset_entries via TorchvisionConverter."""

    def test_collects_supported_images_with_numeric_labels(self, converter, dataset_dir):
        """Dataset collection returns supported image paths paired with class labels."""
        extra_class_dir = dataset_dir / "2"
        extra_class_dir.mkdir()
        ignored_file = extra_class_dir / "ignore.bmp"
        ignored_file.write_bytes(b"bmp")
        (dataset_dir / "README.txt").write_text("ignore me")

        entries = converter._collect_dataset_entries(dataset_dir)

        assert [(path.name, label) for path, label in entries] == [
            ("image_001.jpg", 0),
            ("image_002.jpg", 1),
        ]

    def test_dataset_type_dispatches_to_coco_reader(self, converter, tmp_path):
        """Passing dataset_type='coco-detection' uses the COCO images reader."""
        images = tmp_path / "images"
        annotations = tmp_path / "annotations"
        images.mkdir()
        annotations.mkdir()
        (images / "img1.jpg").write_bytes(b"")
        (annotations / "instances_val2017.json").write_text("{}")

        entries = converter._collect_dataset_entries(tmp_path, dataset_type="coco-detection")

        assert len(entries) == 1
        assert entries[0][0].name == "img1.jpg"
        assert entries[0][1] == 0  # placeholder label for non-classification layouts

    def test_dataset_type_dispatches_to_ade20k_reader(self, converter, tmp_path):
        """Passing dataset_type='ade20k' uses the ADE20K image/mask reader."""
        images = tmp_path / "images"
        annotations = tmp_path / "annotations"
        images.mkdir()
        annotations.mkdir()
        (images / "ADE_val_00000001.jpg").write_bytes(b"")
        (annotations / "ADE_val_00000001.png").write_bytes(b"")

        entries = converter._collect_dataset_entries(tmp_path, dataset_type="ade20k")

        assert len(entries) == 1
        assert entries[0][0].name == "ADE_val_00000001.jpg"


class TestCropResize:
    """Tests for BaseConverter._crop_resize."""

    def test_square_target_tall_image(self, converter):
        """Tall image is center-cropped to a square before resizing."""
        img = np.zeros((60, 40, 3), dtype=np.uint8)
        # Mark the center column so we can verify cropping happened
        img[:, 10:30, :] = 255
        result = converter._crop_resize(img, width=40, height=40)
        assert result.shape == (40, 40, 3)

    def test_square_target_wide_image(self, converter):
        """Wide image is center-cropped to a square before resizing."""
        img = np.zeros((40, 80, 3), dtype=np.uint8)
        result = converter._crop_resize(img, width=20, height=20)
        assert result.shape == (20, 20, 3)

    def test_square_target_already_square(self, converter):
        """Square image is resized directly without cropping."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = converter._crop_resize(img, width=50, height=50)
        assert result.shape == (50, 50, 3)

    def test_non_square_wide_target(self, converter):
        """Target wider than source aspect ratio crops height."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = converter._crop_resize(img, width=200, height=100)
        assert result.shape == (100, 200, 3)

    def test_non_square_tall_target(self, converter):
        """Target taller than source aspect ratio crops width."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = converter._crop_resize(img, width=100, height=200)
        assert result.shape == (200, 100, 3)


class TestPreprocessCalibrationImage:
    """Tests for BaseConverter._preprocess_calibration_image via TorchvisionConverter."""

    def test_preprocesses_image_and_reverses_channels(self, converter, tmp_path):
        """Image preprocessing resizes, normalizes, and switches BGR to RGB when requested."""
        img_path = tmp_path / "pixel.png"
        img = np.array([[[10, 20, 30]]], dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        result = converter._preprocess_calibration_image(
            img_path=img_path,
            width=1,
            height=1,
            mean=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            scale=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            reverse_input_channels=True,
        )

        assert result is not None
        assert result.shape == (1, 3, 1, 1)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result[:, :, 0, 0], np.array([[30.0, 20.0, 10.0]], dtype=np.float32))

    def test_returns_none_for_invalid_image(self, converter, tmp_path):
        """Image preprocessing returns None when cv2 cannot decode the file."""
        bad_path = tmp_path / "not-an-image.txt"
        bad_path.write_text("not an image")

        result = converter._preprocess_calibration_image(
            img_path=bad_path,
            width=8,
            height=8,
            mean=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            scale=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            reverse_input_channels=False,
        )

        assert result is None

    def test_crop_resize_type_produces_square_output(self, converter, tmp_path):
        """resize_type='crop' center-crops a wide image before resizing to target."""
        img_path = tmp_path / "wide.png"
        # 4x8 wide image (H=4, W=8)
        img = np.zeros((4, 8, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        result = converter._preprocess_calibration_image(
            img_path=img_path,
            width=2,
            height=2,
            mean=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            scale=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            reverse_input_channels=False,
            resize_type="crop",
        )

        assert result is not None
        assert result.shape == (1, 3, 2, 2)

    def test_standard_resize_type_is_default(self, converter, tmp_path):
        """Omitting resize_type uses the 'standard' plain-resize path."""
        img_path = tmp_path / "img.png"
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        result = converter._preprocess_calibration_image(
            img_path=img_path,
            width=4,
            height=4,
            mean=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            scale=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            reverse_input_channels=False,
        )

        assert result is not None
        assert result.shape == (1, 3, 4, 4)


class TestCreateCalibrationDataset:
    """Tests for BaseConverter.create_calibration_dataset via TorchvisionConverter."""

    def test_returns_empty_list_when_dataset_path_is_missing(self, tmp_path):
        """Calibration dataset creation is skipped when dataset_path is None."""
        converter = TorchvisionConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_registry=None,
        )

        assert converter.create_calibration_dataset(input_shape=[1, 3, 224, 224]) == ([], [])

    def test_returns_empty_tuple_when_dataset_dir_does_not_exist(self, tmp_path):
        """Calibration dataset returns empty tuple when image_dir disappears between checks."""
        converter = TorchvisionConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_registry=None,
        )
        # Mock dataset_path to simulate a race condition: exists() returns True first then False
        mock_path = MagicMock(spec=Path)
        mock_path.exists.side_effect = [True, False]

        result = converter.create_calibration_dataset(input_shape=[1, 3, 224, 224], dataset_path=mock_path)

        assert result == ([], [])

    def test_returns_empty_tuple_when_no_images_found(self, tmp_path):
        """Calibration dataset returns empty tuple when dataset directory has no images."""
        empty_dataset = tmp_path / "empty_dataset"
        empty_dataset.mkdir()

        converter = TorchvisionConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_registry=None,
        )

        result = converter.create_calibration_dataset(input_shape=[1, 3, 224, 224], dataset_path=empty_dataset)

        assert result == ([], [])

    def test_returns_images_and_labels(self, tmp_path, dataset_dir):
        """Calibration dataset returns preprocessed images and class labels when requested."""
        converter = TorchvisionConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_registry=None,
        )

        images, labels = converter.create_calibration_dataset(
            input_shape=[1, 3, 224, 224],
            return_labels=True,
            dataset_path=dataset_dir,
        )

        assert len(images) == 2
        assert labels == [0, 1]
        assert all(image.shape == (1, 3, 224, 224) for image in images)

    def test_passes_normalization_options_and_subset_size(self, tmp_path, dataset_dir):
        """Calibration dataset forwards normalization options to per-image preprocessing."""
        converter = TorchvisionConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_registry=None,
        )
        seen_calls: list[dict[str, object]] = []

        def fake_preprocess(**kwargs):
            seen_calls.append(kwargs)
            return np.zeros((1, 3, kwargs["height"], kwargs["width"]), dtype=np.float32)

        converter._preprocess_calibration_image = fake_preprocess  # type: ignore[method-assign]

        images, labels = converter.create_calibration_dataset(
            input_shape=[1, 3, 224, 224],
            mean_values="1 2 3",
            scale_values="4 5 6",
            reverse_input_channels=False,
            subset_size=1,
            return_labels=False,
            dataset_path=dataset_dir,
        )

        assert len(images) == 1
        assert labels == []
        assert len(seen_calls) == 1
        np.testing.assert_array_equal(seen_calls[0]["mean"], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(seen_calls[0]["scale"], np.array([4.0, 5.0, 6.0]))
        assert seen_calls[0]["reverse_input_channels"] is False

    def test_skips_unreadable_images_in_return_labels_path(self, tmp_path):
        """Calibration dataset skips images that fail preprocessing when return_labels=True."""
        dataset_path = tmp_path / "dataset"
        class_dir = dataset_path / "0"
        class_dir.mkdir(parents=True)

        (class_dir / "bad_image.jpg").write_text("not an image")
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(class_dir / "good_image.png"), img)

        converter = TorchvisionConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_registry=None,
        )

        images, labels = converter.create_calibration_dataset(
            input_shape=[1, 3, 8, 8],
            return_labels=True,
            dataset_path=dataset_path,
        )

        assert len(images) == 1
        assert labels == [0]

    def test_handles_preprocess_exception_in_return_labels_path(self, tmp_path, dataset_dir):
        """Calibration dataset logs warning and continues when preprocessing raises."""
        converter = TorchvisionConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_registry=None,
        )

        def raising_preprocess(**kwargs):
            disk_error_msg = "disk error"
            raise OSError(disk_error_msg)

        converter._preprocess_calibration_image = raising_preprocess  # type: ignore[method-assign]

        images, labels = converter.create_calibration_dataset(
            input_shape=[1, 3, 224, 224],
            return_labels=True,
            dataset_path=dataset_dir,
        )

        assert images == []
        assert labels == []

    def test_skips_unreadable_images_in_non_return_labels_path(self, tmp_path):
        """Calibration dataset skips images that fail preprocessing when return_labels=False."""
        dataset_path = tmp_path / "dataset"
        class_dir = dataset_path / "0"
        class_dir.mkdir(parents=True)

        (class_dir / "bad_image.jpg").write_text("not an image")
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(class_dir / "good_image.png"), img)

        converter = TorchvisionConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_registry=None,
        )

        images, labels = converter.create_calibration_dataset(
            input_shape=[1, 3, 8, 8],
            return_labels=False,
            dataset_path=dataset_path,
        )

        assert len(images) == 1
        assert labels == []

    def test_handles_preprocess_exception_in_non_return_labels_path(self, tmp_path, dataset_dir):
        """Calibration dataset logs warning and continues when preprocessing raises (no labels)."""
        converter = TorchvisionConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_registry=None,
        )

        def raising_preprocess(**kwargs):
            bad_pixel_msg = "bad pixel data"
            raise ValueError(bad_pixel_msg)

        converter._preprocess_calibration_image = raising_preprocess  # type: ignore[method-assign]

        images, labels = converter.create_calibration_dataset(
            input_shape=[1, 3, 224, 224],
            return_labels=False,
            dataset_path=dataset_dir,
        )

        assert images == []
        assert labels == []

    def test_logs_progress_every_50_images_return_labels(self, tmp_path):
        """Calibration dataset logs progress every 50 images in return_labels path."""
        dataset_path = tmp_path / "dataset"
        class_dir = dataset_path / "0"
        class_dir.mkdir(parents=True)
        # Create 50 image files to trigger the modulo-50 logging
        for i in range(50):
            (class_dir / f"image_{i:04d}.jpg").write_bytes(b"fake")

        converter = TorchvisionConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_registry=None,
        )

        call_count = [0]

        def fake_preprocess(**kwargs):
            call_count[0] += 1
            return np.zeros((1, 3, kwargs["height"], kwargs["width"]), dtype=np.float32)

        converter._preprocess_calibration_image = fake_preprocess  # type: ignore[method-assign]

        images, _ = converter.create_calibration_dataset(
            input_shape=[1, 3, 8, 8],
            return_labels=True,
            subset_size=50,
            dataset_path=dataset_path,
        )

        assert len(images) == 50
        assert call_count[0] == 50

    def test_logs_progress_every_50_images_no_labels(self, tmp_path):
        """Calibration dataset logs progress every 50 images in non-return_labels path."""
        dataset_path = tmp_path / "dataset"
        class_dir = dataset_path / "0"
        class_dir.mkdir(parents=True)
        for i in range(50):
            (class_dir / f"image_{i:04d}.jpg").write_bytes(b"fake")

        converter = TorchvisionConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_registry=None,
        )

        def fake_preprocess(**kwargs):
            return np.zeros((1, 3, kwargs["height"], kwargs["width"]), dtype=np.float32)

        converter._preprocess_calibration_image = fake_preprocess  # type: ignore[method-assign]

        images, labels = converter.create_calibration_dataset(
            input_shape=[1, 3, 8, 8],
            return_labels=False,
            subset_size=50,
            dataset_path=dataset_path,
        )

        assert len(images) == 50
        assert labels == []

    def test_forwards_resize_type_to_preprocess(self, tmp_path, dataset_dir):
        """create_calibration_dataset passes resize_type to _preprocess_calibration_image."""
        converter = TorchvisionConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_registry=None,
        )
        seen_resize_types: list[str] = []

        def fake_preprocess(**kwargs):
            seen_resize_types.append(kwargs.get("resize_type", "standard"))
            return np.zeros((1, 3, kwargs["height"], kwargs["width"]), dtype=np.float32)

        converter._preprocess_calibration_image = fake_preprocess  # type: ignore[method-assign]

        converter.create_calibration_dataset(
            input_shape=[1, 3, 8, 8],
            resize_type="crop",
            return_labels=False,
            dataset_path=dataset_dir,
        )

        assert all(rt == "crop" for rt in seen_resize_types)

    def test_forwards_resize_type_to_preprocess_with_labels(self, tmp_path, dataset_dir):
        """create_calibration_dataset passes resize_type to _preprocess_calibration_image (return_labels=True)."""
        converter = TorchvisionConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_registry=None,
        )
        seen_resize_types: list[str] = []

        def fake_preprocess(**kwargs):
            seen_resize_types.append(kwargs.get("resize_type", "standard"))
            return np.zeros((1, 3, kwargs["height"], kwargs["width"]), dtype=np.float32)

        converter._preprocess_calibration_image = fake_preprocess  # type: ignore[method-assign]

        converter.create_calibration_dataset(
            input_shape=[1, 3, 8, 8],
            resize_type="crop",
            return_labels=True,
            dataset_path=dataset_dir,
        )

        assert all(rt == "crop" for rt in seen_resize_types)

    """Tests for BaseConverter.validate_model via TorchvisionConverter."""

    def test_computes_top1_accuracy(self, converter, tmp_path):
        """Model validation returns the fraction of correct predictions."""
        output_layer = object()
        compiled_model = MagicMock()
        compiled_model.outputs = [output_layer]
        compiled_model.side_effect = [
            {output_layer: np.array([[0.1, 0.9, 0.0]])},
            {output_layer: np.array([[0.8, 0.1, 0.1]])},
        ]
        fake_ov = ModuleType("openvino")
        fake_ov.Core = MagicMock(
            return_value=SimpleNamespace(
                read_model=MagicMock(return_value="model"),
                compile_model=MagicMock(return_value=compiled_model),
            ),
        )

        with patch.dict(sys.modules, {"openvino": fake_ov}):
            accuracy = converter.validate_model(
                model_path=tmp_path / "model.xml",
                validation_data=[np.zeros((1, 3, 224, 224)), np.zeros((1, 3, 224, 224))],
                labels=[1, 0],
            )

        assert accuracy == pytest.approx(1.0)

    def test_returns_zero_when_openvino_fails(self, converter, tmp_path):
        """Model validation returns 0.0 when OpenVINO raises an error."""
        fake_ov = ModuleType("openvino")
        fake_ov.Core = MagicMock(side_effect=RuntimeError("OV error"))

        with patch.dict(sys.modules, {"openvino": fake_ov}):
            accuracy = converter.validate_model(
                model_path=tmp_path / "model.xml",
                validation_data=[np.zeros((1, 3, 224, 224))],
                labels=[0],
            )

        assert accuracy == pytest.approx(0.0)


class TestValidateTorchModel:
    """Tests for PyTorchConverter.validate_torch_model via TorchvisionConverter."""

    def test_computes_top1_accuracy(self, converter):
        """Top-1 accuracy is computed from the original PyTorch model outputs."""
        import torch

        model = MagicMock()
        model.side_effect = [
            (torch.tensor([[0.1, 0.9, 0.0]]),),
            torch.tensor([[0.8, 0.1, 0.1]]),
        ]

        accuracy = converter.validate_torch_model(
            model,
            [np.zeros((1, 3, 224, 224), dtype=np.float64), np.zeros((1, 3, 224, 224), dtype=np.float64)],
            [1, 0],
        )

        model.eval.assert_called_once()
        assert accuracy == pytest.approx(1.0)
        # Inputs are cast to float32 to match PyTorch model weights.
        assert model.call_args.args[0].dtype == torch.float32

    def test_returns_none_when_inference_fails(self, converter):
        """Validation returns None when the PyTorch model raises an error."""
        model = MagicMock(side_effect=RuntimeError("forward failed"))

        accuracy = converter.validate_torch_model(
            model,
            [np.zeros((1, 3, 224, 224), dtype=np.float32)],
            [0],
        )

        assert accuracy is None

    """Tests for BaseConverter.quantize_model via TorchvisionConverter."""

    def test_returns_original_model_when_no_calibration_data(self, converter, sample_model_config, tmp_path):
        """Quantization is skipped when calibration data is empty."""
        model_path = tmp_path / "model.xml"

        assert converter.quantize_model(model_path, [], sample_model_config) == model_path

    def test_quantizes_model_and_writes_artifacts(
        self,
        converter,
        sample_model_config,
        template_dir,
        tmp_path,
        monkeypatch,
    ):
        """Quantization saves the INT8 model plus metadata, README, and gitattributes."""
        _set_template_root(monkeypatch, template_dir)
        (template_dir / "README-torchvision-int8.md").write_text(
            "# <<model_name>>\nVariant: <<variant>>\nTags:\n<<tags_yaml>>\nDocs: <<docs>>\n",
        )
        fp16_dir = tmp_path / "test_model-fp16-ov"
        fp16_dir.mkdir(parents=True)
        model_path = fp16_dir / "test_model_fp32.xml"
        model_path.write_text("<net/>")
        calibration_data = [
            np.zeros((1, 3, 224, 224), dtype=np.float32),
            np.ones((1, 3, 224, 224), dtype=np.float32),
        ]

        quantized_model = MagicMock()
        quantized_model.get_rt_info.return_value = SimpleNamespace(value={"model_type": "Classification"})
        core = SimpleNamespace(read_model=MagicMock(return_value="ov_model"))
        save_model = MagicMock(
            side_effect=lambda _model, path, compress_to_fp16=True: Path(path).write_text("<int8/>") or None,
        )
        fake_ov = ModuleType("openvino")
        fake_ov.Core = MagicMock(return_value=core)
        fake_ov.save_model = save_model

        dataset_factory = MagicMock(side_effect=lambda generator: list(generator))
        quantize = MagicMock(return_value=quantized_model)
        fake_nncf = ModuleType("nncf")
        fake_nncf.Dataset = dataset_factory
        fake_nncf.quantize = quantize
        fake_nncf.QuantizationPreset = SimpleNamespace(PERFORMANCE="performance", MIXED="mixed")

        model_config = sample_model_config | {"tags": ["vision", "classification"]}

        with patch.dict(sys.modules, {"openvino": fake_ov, "nncf": fake_nncf}):
            result = converter.quantize_model(
                model_path=model_path,
                calibration_data=calibration_data,
                model_config=model_config,
                preset="performance",
            )

        output_folder = tmp_path / "test_model-int8-ov"
        assert result == output_folder / "test_model.xml"
        assert result.read_text() == "<int8/>"
        assert json.loads((output_folder / "config.json").read_text()) == {"model_type": "Classification"}
        assert (output_folder / ".gitattributes").read_text() == (template_dir / ".gitattributes").read_text()
        readme = (output_folder / "README.md").read_text()
        assert "# test_model" in readme
        assert "Variant: int8" in readme
        assert "  - vision" in readme
        assert "Docs: https://docs.example.com" in readme
        dataset_factory.assert_called_once()
        quantize.assert_called_once_with(
            "ov_model",
            calibration_dataset=calibration_data,
            preset="performance",
            subset_size=2,
        )
        save_model.assert_called_once()

    def test_passes_transformer_model_type_when_configured(
        self,
        converter,
        sample_model_config,
        template_dir,
        tmp_path,
        monkeypatch,
    ):
        """Transformer models forward NNCF's TRANSFORMER model type to quantization."""
        _set_template_root(monkeypatch, template_dir)
        (template_dir / "README-torchvision-int8.md").write_text("# <<model_name>>")
        fp16_dir = tmp_path / "test_model-fp16-ov"
        fp16_dir.mkdir(parents=True)
        model_path = fp16_dir / "test_model_fp32.xml"
        model_path.write_text("<net/>")
        calibration_data = [np.zeros((1, 3, 224, 224), dtype=np.float32)]

        quantized_model = MagicMock()
        quantized_model.get_rt_info.return_value = SimpleNamespace(value={"model_type": "Classification"})
        core = SimpleNamespace(read_model=MagicMock(return_value="ov_model"))
        fake_ov = ModuleType("openvino")
        fake_ov.Core = MagicMock(return_value=core)
        fake_ov.save_model = MagicMock(
            side_effect=lambda _model, path, compress_to_fp16=True: Path(path).write_text("<int8/>") or None,
        )
        quantize = MagicMock(return_value=quantized_model)
        fake_nncf = ModuleType("nncf")
        fake_nncf.Dataset = MagicMock(side_effect=lambda generator: list(generator))
        fake_nncf.quantize = quantize
        fake_nncf.QuantizationPreset = SimpleNamespace(PERFORMANCE="performance", MIXED="mixed")
        fake_nncf.ModelType = SimpleNamespace(TRANSFORMER="transformer")

        model_config = sample_model_config | {"quantization_model_type": "transformer"}

        with patch.dict(sys.modules, {"openvino": fake_ov, "nncf": fake_nncf}):
            converter.quantize_model(
                model_path=model_path,
                calibration_data=calibration_data,
                model_config=model_config,
                preset="mixed",
            )

        assert quantize.call_args.kwargs["model_type"] == "transformer"

    def test_validates_fp32_and_int8_outputs_when_requested(
        self,
        converter,
        sample_model_config,
        template_dir,
        tmp_path,
        monkeypatch,
    ):
        """Quantization validates both source and INT8 models when validation data is provided."""
        _set_template_root(monkeypatch, template_dir)
        (template_dir / "README-torchvision-int8.md").write_text("# <<model_name>>")
        fp16_dir = tmp_path / "test_model-fp16-ov"
        fp16_dir.mkdir(parents=True)
        model_path = fp16_dir / "test_model_fp32.xml"
        model_path.write_text("<net/>")

        quantized_model = MagicMock()
        quantized_model.get_rt_info.return_value = SimpleNamespace(value={"model_type": "Classification"})
        fake_ov = ModuleType("openvino")
        fake_ov.Core = MagicMock(return_value=SimpleNamespace(read_model=MagicMock(return_value="ov_model")))
        fake_ov.save_model = MagicMock()
        fake_nncf = ModuleType("nncf")
        fake_nncf.Dataset = MagicMock(side_effect=lambda generator: list(generator))
        fake_nncf.quantize = MagicMock(return_value=quantized_model)
        fake_nncf.QuantizationPreset = SimpleNamespace(PERFORMANCE="performance", MIXED="mixed")

        calibration_data = [np.zeros((1, 3, 224, 224), dtype=np.float32)]
        validation_data = [np.zeros((1, 3, 224, 224), dtype=np.float32)]
        validation_labels = [0]
        converter.validate_model = MagicMock(side_effect=[0.97, 0.95])

        with patch.dict(sys.modules, {"openvino": fake_ov, "nncf": fake_nncf}):
            converter.quantize_model(
                model_path=model_path,
                calibration_data=calibration_data,
                model_config=sample_model_config,
                validation_data=validation_data,
                validation_labels=validation_labels,
            )

        call_args = converter.validate_model.call_args_list
        assert call_args[0].args == (model_path, validation_data, validation_labels)
        assert call_args[1].args == (
            tmp_path / "test_model-int8-ov" / "test_model.xml",
            validation_data,
            validation_labels,
        )
        assert converter.validate_model.call_count == 2

    def test_measures_fp16_accuracy_and_populates_results(
        self,
        converter,
        sample_model_config,
        tmp_path,
        template_dir,
        monkeypatch,
    ):
        """Quantization measures FP16 accuracy and fills the AccuracyResults collector."""
        from model_converter.reporting import AccuracyResults

        _set_template_root(monkeypatch, template_dir)
        (template_dir / "README-torchvision-int8.md").write_text("# <<model_name>>")
        fp16_dir = tmp_path / "test_model-fp16-ov"
        fp16_dir.mkdir(parents=True)
        model_path = fp16_dir / "test_model_fp32.xml"
        model_path.write_text("<net/>")
        (fp16_dir / "test_model.xml").write_text("<net/>")

        quantized_model = MagicMock()
        quantized_model.get_rt_info.return_value = SimpleNamespace(value={"model_type": "Classification"})
        fake_ov = ModuleType("openvino")
        fake_ov.Core = MagicMock(return_value=SimpleNamespace(read_model=MagicMock(return_value="ov_model")))
        fake_ov.save_model = MagicMock()
        fake_nncf = ModuleType("nncf")
        fake_nncf.Dataset = MagicMock(side_effect=lambda generator: list(generator))
        fake_nncf.quantize = MagicMock(return_value=quantized_model)
        fake_nncf.QuantizationPreset = SimpleNamespace(PERFORMANCE="performance", MIXED="mixed")

        calibration_data = [np.zeros((1, 3, 224, 224), dtype=np.float32)]
        validation_data = [np.zeros((1, 3, 224, 224), dtype=np.float32)]
        validation_labels = [0]
        converter.validate_model = MagicMock(side_effect=[0.97, 0.96, 0.95])
        accuracy = AccuracyResults()

        with patch.dict(sys.modules, {"openvino": fake_ov, "nncf": fake_nncf}):
            converter.quantize_model(
                model_path=model_path,
                calibration_data=calibration_data,
                model_config=sample_model_config,
                validation_data=validation_data,
                validation_labels=validation_labels,
                accuracy_results=accuracy,
            )

        assert converter.validate_model.call_count == 3
        assert accuracy.measured is True
        assert accuracy.int8_succeeded is True
        assert accuracy.fp32_accuracy == pytest.approx(0.97)
        assert accuracy.fp16_accuracy == pytest.approx(0.96)
        assert accuracy.int8_accuracy == pytest.approx(0.95)

    def test_record_result_copies_measured_accuracy(self, converter, sample_model_config):
        """_record_result copies measured accuracies onto the conversion result."""
        from model_converter.reporting import STATUS_OK, AccuracyResults

        accuracy = AccuracyResults(
            fp32_accuracy=0.9,
            fp16_accuracy=0.89,
            int8_accuracy=0.88,
            int8_succeeded=True,
            measured=True,
        )
        result = converter._record_result(
            converter._build_result(sample_model_config),
            converted=True,
            quantized=True,
            accuracy=accuracy,
        )

        assert result.fp32_accuracy == pytest.approx(0.9)
        assert result.fp16_accuracy == pytest.approx(0.89)
        assert result.int8_accuracy == pytest.approx(0.88)
        assert result.status == STATUS_OK
        assert converter.results[-1] is result

    def test_record_result_calls_upsert_when_report_path_set(
        self,
        tmp_output_dir,
        tmp_cache_dir,
        sample_model_config,
        tmp_path,
    ):
        """_record_result calls upsert_result when report_path is set and not skipped."""
        from unittest.mock import patch

        from model_converter.converters.torchvision import TorchvisionConverter

        report_path = tmp_path / "report.md"
        conv = TorchvisionConverter(output_dir=tmp_output_dir, cache_dir=tmp_cache_dir, report_path=report_path)

        with patch("model_converter.converters.base.upsert_result") as mock_upsert:
            conv._record_result(conv._build_result(sample_model_config), converted=True, quantized=True)

        mock_upsert.assert_called_once()
        assert mock_upsert.call_args.args[1] == report_path

    def test_record_result_skips_upsert_when_skipped(
        self,
        tmp_output_dir,
        tmp_cache_dir,
        sample_model_config,
        tmp_path,
    ):
        """_record_result does not call upsert_result when the model export is skipped."""
        from unittest.mock import patch

        from model_converter.converters.torchvision import TorchvisionConverter

        report_path = tmp_path / "report.md"
        conv = TorchvisionConverter(output_dir=tmp_output_dir, cache_dir=tmp_cache_dir, report_path=report_path)

        with patch("model_converter.converters.base.upsert_result") as mock_upsert:
            conv._record_result(conv._build_result(sample_model_config), converted=False, quantized=False, skipped=True)

        mock_upsert.assert_not_called()

    def test_record_result_skips_upsert_when_no_report_path(self, converter, sample_model_config, tmp_path):
        """_record_result does not call upsert_result when report_path is None."""
        from unittest.mock import patch

        with patch("model_converter.converters.base.upsert_result") as mock_upsert:
            converter._record_result(converter._build_result(sample_model_config), converted=True, quantized=True)

        mock_upsert.assert_not_called()

    def test_returns_original_path_when_nncf_not_installed(self, converter, sample_model_config, tmp_path):
        """Quantization returns original path when nncf is not installed."""
        model_path = tmp_path / "model.xml"
        model_path.write_text("<net/>")
        calibration_data = [np.zeros((1, 3, 224, 224), dtype=np.float32)]

        with patch.dict(sys.modules, {"nncf": None, "openvino": None}):
            result = converter.quantize_model(model_path, calibration_data, sample_model_config)

        assert result == model_path

    def test_returns_original_path_when_quantize_raises_runtime_error(
        self,
        converter,
        sample_model_config,
        tmp_path,
    ):
        """Quantization returns original path when a runtime error occurs."""
        model_path = tmp_path / "test_model-fp16-ov" / "test_model_fp32.xml"
        model_path.parent.mkdir(parents=True)
        model_path.write_text("<net/>")
        calibration_data = [np.zeros((1, 3, 224, 224), dtype=np.float32)]

        fake_ov = ModuleType("openvino")
        fake_ov.Core = MagicMock(return_value=SimpleNamespace(read_model=MagicMock(return_value="ov_model")))
        fake_nncf = ModuleType("nncf")
        fake_nncf.Dataset = MagicMock(side_effect=lambda generator: list(generator))
        fake_nncf.quantize = MagicMock(side_effect=RuntimeError("quantization failed"))
        fake_nncf.QuantizationPreset = SimpleNamespace(PERFORMANCE="performance", MIXED="mixed")

        with patch.dict(sys.modules, {"openvino": fake_ov, "nncf": fake_nncf}):
            result = converter.quantize_model(model_path, calibration_data, sample_model_config)

        assert result == model_path
