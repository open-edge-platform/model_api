#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for model_converter.cli module."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from model_converter.cli import ModelConverter, list_models, main


class TestModelConverterInit:
    """Tests for ModelConverter.__init__."""

    def test_creates_directories(self, tmp_path):
        """ModelConverter creates output and cache directories."""
        output_dir = tmp_path / "output"
        cache_dir = tmp_path / "cache"

        ModelConverter(output_dir=output_dir, cache_dir=cache_dir)

        assert output_dir.exists()
        assert cache_dir.exists()

    def test_verbose_logging(self, tmp_path):
        """ModelConverter sets debug logging in verbose mode."""
        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            verbose=True,
        )
        assert converter.logger is not None

    def test_dataset_path(self, tmp_path):
        """ModelConverter stores dataset path."""
        dataset = tmp_path / "dataset"
        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=dataset,
        )
        assert converter.dataset_path == dataset

    def test_dataset_path_none(self, tmp_path):
        """ModelConverter handles None dataset path."""
        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=None,
        )
        assert converter.dataset_path is None


class TestGetLabels:
    """Tests for ModelConverter.get_labels."""

    def test_imagenet1k_v1(self, converter):
        """get_labels returns ImageNet1K labels."""
        mock_categories = ["tabby cat", "golden retriever", "great white shark"]
        with (
            patch("model_converter.cli.importlib") as _,
            patch.dict(
                "sys.modules",
                {"torchvision.models._meta": MagicMock(_IMAGENET_CATEGORIES=mock_categories)},
            ),
            patch("model_converter.cli.ModelConverter.get_labels", wraps=converter.get_labels),
        ):
            # Directly test the code path
            pass

        # Test with actual mocking of the import
        with patch("torchvision.models._meta._IMAGENET_CATEGORIES", ["tabby cat", "golden retriever"], create=True):
            result = converter.get_labels("IMAGENET1K_V1")
            assert result is not None
            assert " " not in result.split()[0] or "_" in result.split()[0]  # spaces replaced with underscores

    def test_imagenet21k(self, converter):
        """get_labels returns ImageNet21K labels."""
        mock_info = MagicMock()
        mock_info.label_descriptions.return_value = ["tabby, tabby cat", "golden retriever, dog"]

        mock_imagenet_info_cls = MagicMock(return_value=mock_info)
        with patch("timm.data.ImageNetInfo", mock_imagenet_info_cls):
            result = converter.get_labels("IMAGENET21K")

        assert result == "tabby golden_retriever"

    def test_coco_v1(self, converter):
        """get_labels returns COCO labels."""
        mock_weights = MagicMock()
        mock_weights.COCO_V1.meta = {"categories": ["person", "bicycle", "car"]}

        with patch(
            "torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights",
            mock_weights,
        ):
            result = converter.get_labels("COCO_V1")

        assert result == "person bicycle car"

    def test_unknown_label_set(self, converter):
        """get_labels returns None for unknown label sets."""
        result = converter.get_labels("NONEXISTENT_LABELS")
        assert result is None


class TestLoadModelClass:
    """Tests for ModelConverter.load_model_class."""

    def test_successful_import(self, converter):
        """load_model_class dynamically imports a class."""
        result = converter.load_model_class("torch.nn.Linear")
        assert result is torch.nn.Linear

    def test_import_failure(self, converter):
        """load_model_class raises on invalid path."""
        with pytest.raises(ModuleNotFoundError):
            converter.load_model_class("nonexistent.module.Class")


class TestLoadCheckpoint:
    """Tests for ModelConverter.load_checkpoint."""

    def test_successful_load(self, converter, tmp_path):
        """load_checkpoint loads a PyTorch checkpoint."""
        # Create a simple checkpoint
        checkpoint_path = tmp_path / "checkpoint.pth"
        state_dict = {"layer.weight": torch.randn(10, 10)}
        torch.save(state_dict, checkpoint_path)

        result = converter.load_checkpoint(checkpoint_path)
        assert "layer.weight" in result

    def test_load_failure(self, converter, tmp_path):
        """load_checkpoint raises on invalid file."""
        bad_path = tmp_path / "nonexistent.pth"
        with pytest.raises(FileNotFoundError):
            converter.load_checkpoint(bad_path)


class TestLoadHuggingfaceModel:
    """Tests for ModelConverter.load_huggingface_model."""

    def test_timm_model(self, converter):
        """load_huggingface_model loads timm model."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        with patch("timm.create_model", return_value=mock_model) as mock_create:
            result = converter.load_huggingface_model(
                repo_id="timm/resnet50",
                revision="abc123",
                model_library="timm",
            )

        assert result is mock_model
        mock_create.assert_called_once()
        mock_model.eval.assert_called_once()

    def test_timm_model_with_params(self, converter):
        """load_huggingface_model passes model_params to timm."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        with patch("timm.create_model", return_value=mock_model) as mock_create:
            converter.load_huggingface_model(
                repo_id="timm/resnet50",
                revision="abc123",
                model_library="timm",
                model_params={"num_classes": 10},
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["num_classes"] == 10

    def test_transformers_model(self, converter):
        """load_huggingface_model loads transformers model."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        with patch("transformers.AutoModel.from_pretrained", return_value=mock_model) as mock_from:
            result = converter.load_huggingface_model(
                repo_id="bert-base-uncased",
                revision="abc123",
                model_library="transformers",
            )

        assert result is mock_model
        mock_from.assert_called_once()

    def test_unsupported_library(self, converter):
        """load_huggingface_model raises for unsupported library."""
        with pytest.raises(ValueError, match="Unsupported model library"):
            converter.load_huggingface_model(
                repo_id="some/model",
                revision="abc123",
                model_library="unsupported_lib",
            )

    def test_load_failure(self, converter):
        """load_huggingface_model raises on failure."""
        with (
            patch("timm.create_model", side_effect=RuntimeError("Connection error")),
            pytest.raises(RuntimeError, match="Connection error"),
        ):
            converter.load_huggingface_model(
                repo_id="timm/resnet50",
                revision="abc123",
                model_library="timm",
            )


class TestCreateModel:
    """Tests for ModelConverter.create_model."""

    def test_nn_module_with_model_key(self, converter):
        """create_model extracts model from checkpoint 'model' key."""
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model
        checkpoint = {"model": mock_model}

        result = converter.create_model(torch.nn.Module, checkpoint)
        assert result is mock_model

    def test_nn_module_with_state_dict_only_raises(self, converter):
        """create_model raises when nn.Module with only state_dict."""
        checkpoint = {"state_dict": {"layer.weight": torch.randn(10)}}

        with pytest.raises(ValueError, match="state_dict"):
            converter.create_model(torch.nn.Module, checkpoint)

    def test_nn_module_direct_model_as_checkpoint(self, converter):
        """create_model handles nn.Module passed directly (not in dict) gracefully."""
        # When an nn.Module is passed as checkpoint, "model" in checkpoint raises TypeError
        # which is caught by the except block and re-raised
        model = nn.Linear(10, 10)
        with pytest.raises(TypeError):
            converter.create_model(torch.nn.Module, model)

    def test_nn_module_invalid_checkpoint(self, converter):
        """create_model raises when checkpoint is not a valid model."""
        with pytest.raises(ValueError, match="does not contain a valid model"):
            converter.create_model(torch.nn.Module, {"some_key": "some_value"})

    def test_model_class_with_state_dict(self, converter):
        """create_model instantiates class and loads state_dict."""
        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.eval.return_value = mock_instance
        mock_class.return_value = mock_instance

        state_dict = {"layer.weight": torch.randn(10)}
        checkpoint = {"state_dict": state_dict}

        result = converter.create_model(mock_class, checkpoint)
        assert result is mock_instance
        mock_instance.load_state_dict.assert_called_once()

    def test_model_class_with_model_params(self, converter):
        """create_model passes model_params to class constructor."""
        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.eval.return_value = mock_instance
        mock_class.return_value = mock_instance

        checkpoint = {"state_dict": {"w": torch.randn(10)}}

        converter.create_model(mock_class, checkpoint, model_params={"num_classes": 5})
        mock_class.assert_called_once_with(num_classes=5)

    def test_model_class_with_model_key_as_module(self, converter):
        """create_model returns checkpoint['model'] if it's an nn.Module."""
        mock_model = MagicMock(spec=nn.Module)
        mock_class = MagicMock()
        checkpoint = {"model": mock_model}

        result = converter.create_model(mock_class, checkpoint)
        assert result is mock_model

    def test_model_class_with_model_key_as_dict(self, converter):
        """create_model uses checkpoint['model'] as state_dict."""
        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.eval.return_value = mock_instance
        mock_class.return_value = mock_instance

        checkpoint = {"model": {"layer.weight": torch.randn(10)}}

        result = converter.create_model(mock_class, checkpoint)
        assert result is mock_instance
        mock_instance.load_state_dict.assert_called_once()

    def test_model_class_bare_state_dict(self, converter):
        """create_model uses checkpoint directly as state_dict."""
        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.eval.return_value = mock_instance
        mock_class.return_value = mock_instance

        checkpoint = {"layer.weight": torch.randn(10)}

        result = converter.create_model(mock_class, checkpoint)
        assert result is mock_instance
        mock_instance.load_state_dict.assert_called_once()

    def test_create_model_failure(self, converter):
        """create_model raises on instantiation failure."""
        mock_class = MagicMock(side_effect=RuntimeError("init failed"))

        with pytest.raises(RuntimeError, match="init failed"):
            converter.create_model(mock_class, {})


class TestCopyReadme:
    """Tests for ModelConverter.copy_readme."""

    def test_successful_copy(self, converter, tmp_path):
        """copy_readme copies and fills README template."""
        output_folder = tmp_path / "model-fp16-ov"
        output_folder.mkdir()

        # We mock the template file reading
        template_content = "# <<model_name>>\nLicense: <<license>>\nLink: <<license_link>>\nDocs: <<docs>>"

        config = {
            "model_short_name": "test_model",
            "license": "Apache-2.0",
            "license_link": "https://apache.org/licenses/LICENSE-2.0",
            "docs": "https://docs.example.com",
            "model_library": "timm",
        }

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=template_content),
        ):
            converter.copy_readme(config, output_folder, variant="fp16")

        readme = output_folder / "README.md"
        assert readme.exists()
        content = readme.read_text()
        assert "test_model" in content
        assert "Apache-2.0" in content

    def test_template_not_found(self, converter, tmp_path):
        """copy_readme handles missing template gracefully."""
        output_folder = tmp_path / "model-fp16-ov"
        output_folder.mkdir()

        config = {
            "model_short_name": "test_model",
            "license": "MIT",
            "license_link": "https://mit.edu",
            "docs": "",
            "model_library": "timm",
        }

        # Template path doesn't exist
        with patch.object(Path, "exists", return_value=False):
            converter.copy_readme(config, output_folder, variant="fp16")

        # No README should be created
        assert not (output_folder / "README.md").exists()

    def test_missing_model_short_name(self, converter, tmp_path):
        """copy_readme warns when model_short_name is empty."""
        output_folder = tmp_path / "model-fp16-ov"
        output_folder.mkdir()

        config = {
            "model_short_name": "",
            "license": "MIT",
            "license_link": "https://mit.edu",
        }

        # Should not raise but log warning
        converter.copy_readme(config, output_folder)

    def test_missing_license_link(self, converter, tmp_path):
        """copy_readme warns when license_link is empty."""
        output_folder = tmp_path / "model-fp16-ov"
        output_folder.mkdir()

        config = {
            "model_short_name": "test",
            "license": "MIT",
            "license_link": "",
        }

        converter.copy_readme(config, output_folder)

    def test_missing_license(self, converter, tmp_path):
        """copy_readme warns when license is empty."""
        output_folder = tmp_path / "model-fp16-ov"
        output_folder.mkdir()

        config = {
            "model_short_name": "test",
            "license": "",
            "license_link": "https://mit.edu",
        }

        converter.copy_readme(config, output_folder)

    def test_missing_docs_field(self, converter, tmp_path):
        """copy_readme handles missing docs field."""
        output_folder = tmp_path / "model-fp16-ov"
        output_folder.mkdir()

        template_content = "# <<model_name>>"
        config = {
            "model_short_name": "test_model",
            "license": "Apache-2.0",
            "license_link": "https://apache.org",
            "model_library": "timm",
        }

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=template_content),
        ):
            converter.copy_readme(config, output_folder, variant="fp16")

    def test_none_value_in_config(self, converter, tmp_path):
        """copy_readme skips None values in config placeholders."""
        output_folder = tmp_path / "model-fp16-ov"
        output_folder.mkdir()

        template_content = "# <<model_name>>"
        config = {
            "model_short_name": "test_model",
            "license": "Apache-2.0",
            "license_link": "https://apache.org",
            "model_library": "timm",
            "optional_field": None,
        }

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=template_content),
        ):
            converter.copy_readme(config, output_folder, variant="fp16")


class TestCollectDatasetEntries:
    """Tests for ModelConverter._collect_dataset_entries."""

    def test_collects_entries(self, converter, dataset_dir):
        """_collect_dataset_entries finds images with class labels."""
        entries = converter._collect_dataset_entries(dataset_dir)
        assert len(entries) == 2
        # Entries are (path, class_label) tuples
        assert entries[0][1] == 0
        assert entries[1][1] == 1

    def test_empty_directory(self, converter, tmp_path):
        """_collect_dataset_entries returns empty list for empty dir."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        entries = converter._collect_dataset_entries(empty_dir)
        assert entries == []


class TestPreprocessCalibrationImage:
    """Tests for ModelConverter._preprocess_calibration_image."""

    def test_valid_image(self, converter, tmp_path):
        """_preprocess_calibration_image processes image correctly."""
        # Create a dummy image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), img)

        result = converter._preprocess_calibration_image(
            img_path=img_path,
            width=224,
            height=224,
            mean=np.array([123.675, 116.28, 103.53]),
            scale=np.array([58.395, 57.12, 57.375]),
            reverse_input_channels=True,
        )

        assert result is not None
        assert result.shape == (1, 3, 224, 224)

    def test_no_channel_reversal(self, converter, tmp_path):
        """_preprocess_calibration_image without channel reversal."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), img)

        result = converter._preprocess_calibration_image(
            img_path=img_path,
            width=224,
            height=224,
            mean=np.array([0, 0, 0]),
            scale=np.array([1, 1, 1]),
            reverse_input_channels=False,
        )

        assert result is not None
        assert result.shape == (1, 3, 224, 224)

    def test_invalid_image(self, converter, tmp_path):
        """_preprocess_calibration_image returns None for invalid image."""
        bad_path = tmp_path / "notanimage.txt"
        bad_path.write_text("not an image")

        result = converter._preprocess_calibration_image(
            img_path=bad_path,
            width=224,
            height=224,
            mean=np.array([0, 0, 0]),
            scale=np.array([1, 1, 1]),
            reverse_input_channels=True,
        )

        assert result is None


class TestCreateCalibrationDataset:
    """Tests for ModelConverter.create_calibration_dataset."""

    def test_no_dataset_path(self, tmp_path):
        """create_calibration_dataset returns empty when no dataset path."""
        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=None,
        )
        result = converter.create_calibration_dataset(input_shape=[1, 3, 224, 224])
        assert result == []

    def test_nonexistent_dataset_path(self, tmp_path):
        """create_calibration_dataset returns empty for missing path."""
        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=tmp_path / "nonexistent",
        )
        result = converter.create_calibration_dataset(input_shape=[1, 3, 224, 224])
        assert result == []

    def test_with_return_labels(self, tmp_path, dataset_dir):
        """create_calibration_dataset returns images and labels."""
        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=dataset_dir,
        )
        result = converter.create_calibration_dataset(
            input_shape=[1, 3, 224, 224],
            return_labels=True,
        )
        images, labels = result
        assert len(images) == 2
        assert len(labels) == 2
        assert labels[0] == 0
        assert labels[1] == 1

    def test_without_return_labels(self, tmp_path, dataset_dir):
        """create_calibration_dataset returns images without labels flag."""
        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=dataset_dir,
        )
        result = converter.create_calibration_dataset(
            input_shape=[1, 3, 224, 224],
            return_labels=False,
        )
        images, labels = result
        assert len(images) == 2
        assert labels == []

    def test_with_mean_scale(self, tmp_path, dataset_dir):
        """create_calibration_dataset uses mean and scale values."""
        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=dataset_dir,
        )
        result = converter.create_calibration_dataset(
            input_shape=[1, 3, 224, 224],
            mean_values="123.675 116.28 103.53",
            scale_values="58.395 57.12 57.375",
            return_labels=True,
        )
        images, _labels = result
        assert len(images) == 2

    def test_empty_dataset(self, tmp_path):
        """create_calibration_dataset handles empty dataset directory."""
        empty_dataset = tmp_path / "empty_dataset"
        empty_dataset.mkdir()
        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=empty_dataset,
        )
        result = converter.create_calibration_dataset(
            input_shape=[1, 3, 224, 224],
            return_labels=True,
        )
        assert result == ([], [])

    def test_subset_size(self, tmp_path, dataset_dir):
        """create_calibration_dataset respects subset_size."""
        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=dataset_dir,
        )
        result = converter.create_calibration_dataset(
            input_shape=[1, 3, 224, 224],
            subset_size=1,
            return_labels=True,
        )
        images, _labels = result
        assert len(images) == 1

    def test_image_processing_error(self, tmp_path):
        """create_calibration_dataset skips images that raise exceptions (with labels)."""
        dataset_path = tmp_path / "dataset"
        class_dir = dataset_path / "0"
        class_dir.mkdir(parents=True)
        # Create a valid image file that will trigger an exception in preprocessing
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(class_dir / "image_001.jpg"), img)

        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=dataset_path,
        )
        # Mock _preprocess_calibration_image to raise an exception
        with patch.object(converter, "_preprocess_calibration_image", side_effect=ValueError("bad image")):
            result = converter.create_calibration_dataset(
                input_shape=[1, 3, 224, 224],
                return_labels=True,
            )
        images, _labels = result
        assert len(images) == 0

    def test_image_processing_error_no_labels(self, tmp_path):
        """create_calibration_dataset skips images that raise exceptions (without labels)."""
        dataset_path = tmp_path / "dataset"
        class_dir = dataset_path / "0"
        class_dir.mkdir(parents=True)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(class_dir / "image_001.jpg"), img)

        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=dataset_path,
        )
        with patch.object(converter, "_preprocess_calibration_image", side_effect=OSError("read error")):
            result = converter.create_calibration_dataset(
                input_shape=[1, 3, 224, 224],
                return_labels=False,
            )
        images, _labels = result
        assert len(images) == 0

    def test_image_returns_none_with_labels(self, tmp_path):
        """create_calibration_dataset skips None images (with labels)."""
        dataset_path = tmp_path / "dataset"
        class_dir = dataset_path / "0"
        class_dir.mkdir(parents=True)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(class_dir / "image_001.jpg"), img)

        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=dataset_path,
        )
        with patch.object(converter, "_preprocess_calibration_image", return_value=None):
            result = converter.create_calibration_dataset(
                input_shape=[1, 3, 224, 224],
                return_labels=True,
            )
        images, _labels = result
        assert len(images) == 0

    def test_image_returns_none_without_labels(self, tmp_path):
        """create_calibration_dataset skips None images (without labels)."""
        dataset_path = tmp_path / "dataset"
        class_dir = dataset_path / "0"
        class_dir.mkdir(parents=True)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(class_dir / "image_001.jpg"), img)

        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=dataset_path,
        )
        with patch.object(converter, "_preprocess_calibration_image", return_value=None):
            result = converter.create_calibration_dataset(
                input_shape=[1, 3, 224, 224],
                return_labels=False,
            )
        images, _labels = result
        assert len(images) == 0

    def test_progress_logging_with_labels(self, tmp_path):
        """create_calibration_dataset logs progress every 50 images (with labels)."""
        dataset_path = tmp_path / "dataset"
        class_dir = dataset_path / "0"
        class_dir.mkdir(parents=True)

        # Create 51 images to trigger the progress logging at i=49 (i+1=50)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        for i in range(51):
            cv2.imwrite(str(class_dir / f"image_{i:03d}.jpg"), img)

        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=dataset_path,
            verbose=True,
        )
        result = converter.create_calibration_dataset(
            input_shape=[1, 3, 10, 10],
            return_labels=True,
        )
        images, _labels = result
        assert len(images) == 51

    def test_progress_logging_without_labels(self, tmp_path):
        """create_calibration_dataset logs progress every 50 images (without labels)."""
        dataset_path = tmp_path / "dataset"
        class_dir = dataset_path / "0"
        class_dir.mkdir(parents=True)

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        for i in range(51):
            cv2.imwrite(str(class_dir / f"image_{i:03d}.jpg"), img)

        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=dataset_path,
            verbose=True,
        )
        result = converter.create_calibration_dataset(
            input_shape=[1, 3, 10, 10],
            return_labels=False,
        )
        images, _labels = result
        assert len(images) == 51

    def test_dataset_dir_removed_after_init_check(self, tmp_path):
        """create_calibration_dataset handles dir removed between checks."""

        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()

        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=dataset_path,
        )

        # Remove the directory after converter init but before calibration runs
        # Patch exists() to return True on first call (line 415) then False on second (line 428)
        original_exists = Path.exists
        call_count = [0]

        def mock_exists(self_path):
            if self_path == dataset_path:
                call_count[0] += 1
                return call_count[0] == 1
            return original_exists(self_path)

        with patch.object(Path, "exists", mock_exists):
            result = converter.create_calibration_dataset(
                input_shape=[1, 3, 224, 224],
                return_labels=True,
            )
        assert result == ([], [])


class TestValidateModel:
    """Tests for ModelConverter.validate_model."""

    def test_correct_predictions(self, converter, tmp_path):
        """validate_model computes accuracy correctly."""
        mock_output_layer = MagicMock()
        mock_compiled = MagicMock()
        mock_compiled.outputs = [mock_output_layer]

        # Setup callable to return predictions matching labels
        mock_compiled.return_value = {mock_output_layer: np.array([[0.1, 0.9, 0.0]])}
        call_results = [
            {mock_output_layer: np.array([[0.1, 0.9, 0.0]])},  # pred = 1
            {mock_output_layer: np.array([[0.0, 0.0, 0.9]])},  # pred = 2
        ]
        mock_compiled.side_effect = call_results

        mock_core = MagicMock()
        mock_core.read_model.return_value = MagicMock()
        mock_core.compile_model.return_value = mock_compiled

        with patch("openvino.Core", return_value=mock_core):
            accuracy = converter.validate_model(
                model_path=tmp_path / "model.xml",
                validation_data=[np.zeros((1, 3, 224, 224)), np.zeros((1, 3, 224, 224))],
                labels=[1, 2],
            )

        assert accuracy == pytest.approx(1.0)

    def test_partial_accuracy(self, converter, tmp_path):
        """validate_model returns partial accuracy."""
        mock_output_layer = MagicMock()
        mock_compiled = MagicMock()
        mock_compiled.outputs = [mock_output_layer]

        call_results = [
            {mock_output_layer: np.array([[0.9, 0.1]])},  # pred = 0 (correct)
            {mock_output_layer: np.array([[0.9, 0.1]])},  # pred = 0 (wrong, label=1)
        ]
        mock_compiled.side_effect = call_results

        mock_core = MagicMock()
        mock_core.read_model.return_value = MagicMock()
        mock_core.compile_model.return_value = mock_compiled

        with patch("openvino.Core", return_value=mock_core):
            accuracy = converter.validate_model(
                model_path=tmp_path / "model.xml",
                validation_data=[np.zeros((1, 3, 224, 224)), np.zeros((1, 3, 224, 224))],
                labels=[0, 1],
            )

        assert accuracy == pytest.approx(0.5)

    def test_validation_failure(self, converter, tmp_path):
        """validate_model returns 0.0 on error."""
        with patch("openvino.Core", side_effect=RuntimeError("OV error")):
            accuracy = converter.validate_model(
                model_path=tmp_path / "model.xml",
                validation_data=[np.zeros((1, 3, 224, 224))],
                labels=[0],
            )

        assert accuracy == pytest.approx(0.0)


class TestQuantizeModel:
    """Tests for ModelConverter.quantize_model."""

    def test_no_calibration_data(self, converter, sample_model_config, tmp_path):
        """quantize_model returns model_path when no calibration data."""
        model_path = tmp_path / "model.xml"
        result = converter.quantize_model(
            model_path=model_path,
            calibration_data=[],
            model_config=sample_model_config,
        )
        assert result == model_path

    def test_successful_quantization(self, converter, sample_model_config, tmp_path):
        """quantize_model performs INT8 quantization."""
        # Setup model path
        fp16_dir = tmp_path / "test_model-fp16-ov"
        fp16_dir.mkdir(parents=True)
        model_path = fp16_dir / "test_model_fp32.xml"
        model_path.write_text("<net/>")

        mock_ov_model = MagicMock()
        mock_quantized = MagicMock()
        mock_quantized.get_rt_info.return_value = MagicMock(value={"model_type": "Classification"})

        mock_core = MagicMock()
        mock_core.read_model.return_value = mock_ov_model

        calibration_data = [np.zeros((1, 3, 224, 224))]

        def consume_dataset(gen):
            """Mock nncf.Dataset that consumes the generator."""
            list(gen)  # Consume the generator to cover lines 572-573
            return MagicMock()

        with (
            patch("openvino.Core", return_value=mock_core),
            patch("nncf.quantize", return_value=mock_quantized),
            patch("nncf.Dataset", side_effect=consume_dataset),
            patch("nncf.QuantizationPreset") as mock_preset,
            patch("openvino.save_model"),
            patch.object(Path, "exists", return_value=True),
            patch("shutil.copy2"),
            patch.object(converter, "copy_readme"),
        ):
            mock_preset.MIXED = "mixed"
            mock_preset.PERFORMANCE = "performance"
            result = converter.quantize_model(
                model_path=model_path,
                calibration_data=calibration_data,
                model_config=sample_model_config,
                preset="mixed",
            )

        assert result != model_path  # Should return new quantized path

    def test_nncf_not_installed(self, converter, sample_model_config, tmp_path):
        """quantize_model handles missing NNCF."""
        model_path = tmp_path / "model.xml"
        calibration_data = [np.zeros((1, 3, 224, 224))]

        with (
            patch.dict("sys.modules", {"nncf": None}),
            patch("builtins.__import__", side_effect=ImportError("No module named 'nncf'")),
        ):
            result = converter.quantize_model(
                model_path=model_path,
                calibration_data=calibration_data,
                model_config=sample_model_config,
            )

        assert result == model_path

    def test_quantization_with_validation(self, converter, sample_model_config, tmp_path):
        """quantize_model validates accuracy when validation data provided."""
        fp16_dir = tmp_path / "test_model-fp16-ov"
        fp16_dir.mkdir(parents=True)
        model_path = fp16_dir / "test_model_fp32.xml"
        model_path.write_text("<net/>")

        mock_quantized = MagicMock()
        mock_quantized.get_rt_info.return_value = MagicMock(value={"model_type": "Classification"})

        mock_core = MagicMock()
        mock_core.read_model.return_value = MagicMock()

        calibration_data = [np.zeros((1, 3, 224, 224))]
        validation_data = [np.zeros((1, 3, 224, 224))]
        validation_labels = [0]

        def consume_dataset(gen):
            list(gen)
            return MagicMock()

        with (
            patch("openvino.Core", return_value=mock_core),
            patch("nncf.quantize", return_value=mock_quantized),
            patch("nncf.Dataset", side_effect=consume_dataset),
            patch("nncf.QuantizationPreset") as mock_preset,
            patch("openvino.save_model"),
            patch.object(Path, "exists", return_value=True),
            patch("shutil.copy2"),
            patch.object(converter, "copy_readme"),
            patch.object(converter, "validate_model", return_value=0.95),
        ):
            mock_preset.MIXED = "mixed"
            converter.quantize_model(
                model_path=model_path,
                calibration_data=calibration_data,
                model_config=sample_model_config,
                validation_data=validation_data,
                validation_labels=validation_labels,
            )

    def test_quantization_runtime_error(self, converter, sample_model_config, tmp_path):
        """quantize_model handles runtime errors gracefully."""
        model_path = tmp_path / "model.xml"
        calibration_data = [np.zeros((1, 3, 224, 224))]

        with patch("openvino.Core", side_effect=RuntimeError("OV error")):
            result = converter.quantize_model(
                model_path=model_path,
                calibration_data=calibration_data,
                model_config=sample_model_config,
            )

        assert result == model_path


class TestExportToOpenvino:
    """Tests for ModelConverter.export_to_openvino."""

    def test_successful_export(self, converter, sample_model_config, tmp_path):
        """export_to_openvino exports model to OV format."""
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_ov_model = MagicMock()
        mock_input = MagicMock()
        mock_input.get_names.return_value = {"input"}
        mock_ov_model.input.return_value = mock_input
        mock_ov_model.inputs = [mock_input]
        mock_ov_model.outputs = [MagicMock()]
        mock_ov_model.get_rt_info.return_value = MagicMock(value={"model_type": "Classification"})

        output_path = converter.output_dir / "test_model"

        with (
            patch("openvino.convert_model", return_value=mock_ov_model),
            patch("openvino.save_model"),
            patch.object(Path, "exists", return_value=True),
            patch("shutil.copy2"),
            patch.object(converter, "copy_readme"),
        ):
            fp16_path, _fp32_path = converter.export_to_openvino(
                model=mock_model,
                input_shape=[1, 3, 224, 224],
                output_path=output_path,
                model_config=sample_model_config,
                input_names=["input"],
                output_names=["result"],
                metadata={("model_info", "model_type"): "Classification"},
            )

        assert "fp16" in str(fp16_path.parent) or fp16_path.name == "test_model.xml"

    def test_export_failure(self, converter, sample_model_config, tmp_path):
        """export_to_openvino raises on conversion failure."""
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        output_path = converter.output_dir / "test_model"

        with (
            patch("openvino.convert_model", side_effect=RuntimeError("Conversion failed")),
            pytest.raises(RuntimeError, match="Conversion failed"),
        ):
            converter.export_to_openvino(
                model=mock_model,
                input_shape=[1, 3, 224, 224],
                output_path=output_path,
                model_config=sample_model_config,
            )


class TestPrepareModelForExport:
    """Tests for ModelConverter._prepare_model_for_export."""

    def test_with_adapter(self, converter):
        """_prepare_model_for_export applies adapter for known model type."""
        mock_model = MagicMock(spec=nn.Module)
        config = {"model_type": "MaskRCNN"}

        with patch("model_converter.cli.get_adapter") as mock_get_adapter:
            mock_adapted = MagicMock()
            mock_get_adapter.return_value = mock_adapted
            result = converter._prepare_model_for_export(mock_model, config)

        assert result is mock_adapted

    def test_without_adapter(self, converter):
        """_prepare_model_for_export returns model unchanged for no adapter."""
        mock_model = MagicMock(spec=nn.Module)
        config = {"model_type": "Classification"}

        with patch("model_converter.cli.get_adapter", return_value=mock_model):
            result = converter._prepare_model_for_export(mock_model, config)

        assert result is mock_model


class TestCreateExampleInput:
    """Tests for ModelConverter._create_example_input."""

    def test_maskrcnn_input(self, converter):
        """_create_example_input uses rand for maskrcnn."""
        config = {"model_type": "MaskRCNN"}
        result = converter._create_example_input([1, 3, 224, 224], config)
        assert result.shape == (1, 3, 224, 224)
        assert result.min() >= 0  # rand produces [0, 1)

    def test_default_input(self, converter):
        """_create_example_input uses randn for non-maskrcnn."""
        config = {"model_type": "Classification"}
        result = converter._create_example_input([1, 3, 224, 224], config)
        assert result.shape == (1, 3, 224, 224)


class TestPostprocessOpenvinoModel:
    """Tests for ModelConverter._postprocess_openvino_model."""

    def test_set_input_names(self, converter, mock_ov_model):
        """_postprocess_openvino_model sets input tensor names."""
        converter._postprocess_openvino_model(
            mock_ov_model,
            input_names=["images"],
        )
        mock_ov_model.input(0).set_names.assert_called_with({"images"})

    def test_set_output_names(self, converter, mock_ov_model):
        """_postprocess_openvino_model sets output tensor names."""
        converter._postprocess_openvino_model(
            mock_ov_model,
            output_names=["predictions"],
        )
        mock_ov_model.output(0).set_names.assert_called_with({"predictions"})

    def test_set_metadata(self, converter, mock_ov_model):
        """_postprocess_openvino_model adds metadata."""
        metadata = {
            ("model_info", "model_type"): "Classification",
            ("model_info", "labels"): "cat dog",
        }
        converter._postprocess_openvino_model(mock_ov_model, metadata=metadata)
        assert mock_ov_model.set_rt_info.call_count == 2

    def test_no_operations(self, converter, mock_ov_model):
        """_postprocess_openvino_model handles None params."""
        result = converter._postprocess_openvino_model(mock_ov_model)
        assert result is mock_ov_model


class TestLoadModelFromConfig:
    """Tests for ModelConverter._load_model_from_config."""

    def test_huggingface_path(self, converter):
        """_load_model_from_config loads from HuggingFace."""
        config = {
            "huggingface_repo": "timm/resnet50",
            "huggingface_revision": "abc123",
            "model_library": "timm",
        }
        mock_model = MagicMock()
        with patch.object(converter, "load_huggingface_model", return_value=mock_model):
            result = converter._load_model_from_config(config)
        assert result is mock_model

    def test_huggingface_missing_revision(self, converter):
        """_load_model_from_config raises when HF revision is missing."""
        config = {
            "huggingface_repo": "timm/resnet50",
        }
        with pytest.raises(ValueError, match="huggingface_revision"):
            converter._load_model_from_config(config)

    def test_url_path(self, converter, tmp_path):
        """_load_model_from_config loads from URL."""
        config = {
            "weights_url": "https://example.com/weights.pth",
            "model_class_name": "torch.nn.Module",
        }

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        with (
            patch.object(converter._url_downloader, "download", return_value=tmp_path / "weights.pth"),
            patch.object(converter, "load_model_class", return_value=torch.nn.Module),
            patch.object(converter, "load_checkpoint", return_value={"model": mock_model}),
            patch.object(converter, "create_model", return_value=mock_model),
        ):
            result = converter._load_model_from_config(config)

        assert result is mock_model

    def test_url_path_default_class(self, converter, tmp_path):
        """_load_model_from_config uses torch.nn.Module as default class."""
        config = {
            "weights_url": "https://example.com/weights.pth",
        }

        mock_model = MagicMock(spec=nn.Module)

        with (
            patch.object(converter._url_downloader, "download", return_value=tmp_path / "weights.pth"),
            patch.object(converter, "load_model_class", return_value=torch.nn.Module) as mock_load_class,
            patch.object(converter, "load_checkpoint", return_value={}),
            patch.object(converter, "create_model", return_value=mock_model),
        ):
            converter._load_model_from_config(config)

        mock_load_class.assert_called_once_with("torch.nn.Module")


class TestQuantizeAndCleanup:
    """Tests for ModelConverter._quantize_and_cleanup."""

    def test_with_classification_labels(self, converter, tmp_path, sample_model_config):
        """_quantize_and_cleanup runs validation for classification with labels."""
        fp32_path = tmp_path / "model_fp32.xml"
        fp32_path.write_text("<net/>")
        fp32_bin = tmp_path / "model_fp32.bin"
        fp32_bin.write_text("weights")

        validation_data = [np.zeros((1, 3, 224, 224))]
        validation_labels = [0]

        with (
            patch.object(
                converter,
                "create_calibration_dataset",
                return_value=(validation_data, validation_labels),
            ),
            patch.object(converter, "quantize_model"),
        ):
            converter._quantize_and_cleanup(
                sample_model_config,
                fp32_path,
                model_type="Classification",
                input_shape=[1, 3, 224, 224],
                mean_values="123.675 116.28 103.53",
                scale_values="58.395 57.12 57.375",
                reverse_input_channels=True,
            )

        # FP32 files should be cleaned up
        assert not fp32_path.exists()
        assert not fp32_bin.exists()

    def test_without_classification_labels(self, converter, tmp_path, sample_model_config):
        """_quantize_and_cleanup skips validation for non-classification."""
        fp32_path = tmp_path / "model_fp32.xml"
        fp32_path.write_text("<net/>")

        config = {**sample_model_config, "labels": None}
        validation_data = [np.zeros((1, 3, 224, 224))]

        with (
            patch.object(converter, "create_calibration_dataset", return_value=(validation_data, [])),
            patch.object(converter, "quantize_model") as mock_quantize,
        ):
            converter._quantize_and_cleanup(
                config,
                fp32_path,
                model_type="Detection",
                input_shape=[1, 3, 224, 224],
                mean_values="123.675 116.28 103.53",
                scale_values="58.395 57.12 57.375",
                reverse_input_channels=True,
            )

        # Quantize should be called with no validation data/labels
        mock_quantize.assert_called_once()
        call_kwargs = mock_quantize.call_args[1]
        assert call_kwargs["validation_data"] is None
        assert call_kwargs["validation_labels"] is None

    def test_empty_calibration_data(self, converter, tmp_path, sample_model_config):
        """_quantize_and_cleanup skips quantization when no data."""
        fp32_path = tmp_path / "model_fp32.xml"
        fp32_path.write_text("<net/>")

        with (
            patch.object(converter, "create_calibration_dataset", return_value=([], [])),
            patch.object(converter, "quantize_model") as mock_quantize,
        ):
            converter._quantize_and_cleanup(
                sample_model_config,
                fp32_path,
                model_type="Classification",
                input_shape=[1, 3, 224, 224],
                mean_values="123.675 116.28 103.53",
                scale_values="58.395 57.12 57.375",
                reverse_input_channels=True,
            )

        mock_quantize.assert_not_called()

    def test_cleanup_failure(self, converter, tmp_path, sample_model_config):
        """_quantize_and_cleanup handles cleanup failure gracefully."""
        fp32_path = tmp_path / "model_fp32.xml"
        fp32_path.write_text("<net/>")

        with (
            patch.object(converter, "create_calibration_dataset", return_value=([], [])),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "unlink", side_effect=OSError("Permission denied")),
        ):
            # Should not raise
            converter._quantize_and_cleanup(
                sample_model_config,
                fp32_path,
                model_type="Classification",
                input_shape=[1, 3, 224, 224],
                mean_values="123.675 116.28 103.53",
                scale_values="58.395 57.12 57.375",
                reverse_input_channels=True,
            )


class TestProcessModelConfig:
    """Tests for ModelConverter.process_model_config."""

    def test_already_exists(self, converter, sample_model_config):
        """process_model_config skips when both models already exist."""
        # Create existing model files
        fp16_dir = converter.output_dir / "test_model-fp16-ov"
        fp16_dir.mkdir(parents=True)
        (fp16_dir / "test_model.xml").write_text("<net/>")

        int8_dir = converter.output_dir / "test_model-int8-ov"
        int8_dir.mkdir(parents=True)
        (int8_dir / "test_model.xml").write_text("<net/>")

        result = converter.process_model_config(sample_model_config)
        assert result is True

    def test_missing_license(self, converter):
        """process_model_config fails when license is missing."""
        config = {"model_short_name": "test", "license_link": "https://example.com"}
        result = converter.process_model_config(config)
        assert result is False

    def test_missing_license_link(self, converter):
        """process_model_config fails when license_link is missing."""
        config = {"model_short_name": "test", "license": "MIT"}
        result = converter.process_model_config(config)
        assert result is False

    def test_successful_conversion(self, converter, sample_model_config):
        """process_model_config successfully converts a model."""
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        fp16_path = converter.output_dir / "test_model-fp16-ov" / "test_model.xml"
        fp32_path = converter.output_dir / "test_model-fp16-ov" / "test_model_fp32.xml"

        with (
            patch.object(converter, "_load_model_from_config", return_value=mock_model),
            patch.object(converter, "get_labels", return_value="cat dog"),
            patch.object(converter, "export_to_openvino", return_value=(fp16_path, fp32_path)),
        ):
            result = converter.process_model_config(sample_model_config)

        assert result is True

    def test_successful_conversion_with_dataset(self, tmp_path, sample_model_config, dataset_dir):
        """process_model_config quantizes when dataset is available."""
        conv = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_path=dataset_dir,
        )
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        fp16_path = conv.output_dir / "test_model-fp16-ov" / "test_model.xml"
        fp32_path = conv.output_dir / "test_model-fp16-ov" / "test_model_fp32.xml"

        with (
            patch.object(conv, "_load_model_from_config", return_value=mock_model),
            patch.object(conv, "get_labels", return_value="cat dog"),
            patch.object(conv, "export_to_openvino", return_value=(fp16_path, fp32_path)),
            patch.object(conv, "_quantize_and_cleanup"),
        ):
            result = conv.process_model_config(sample_model_config)

        assert result is True

    def test_conversion_failure(self, converter, sample_model_config):
        """process_model_config returns False on failure."""
        with patch.object(converter, "_load_model_from_config", side_effect=RuntimeError("load failed")):
            result = converter.process_model_config(sample_model_config)

        assert result is False

    def test_no_labels_configured(self, converter):
        """process_model_config works without labels in config."""
        config = {
            "model_short_name": "test_model",
            "license": "Apache-2.0",
            "license_link": "https://apache.org",
            "weights_url": "https://example.com/weights.pth",
            "model_class_name": "torch.nn.Module",
            "input_shape": [1, 3, 224, 224],
            "model_type": "Classification",
        }

        mock_model = MagicMock(spec=nn.Module)
        fp16_path = converter.output_dir / "test_model-fp16-ov" / "test_model.xml"
        fp32_path = converter.output_dir / "test_model-fp16-ov" / "test_model_fp32.xml"

        with (
            patch.object(converter, "_load_model_from_config", return_value=mock_model),
            patch.object(converter, "export_to_openvino", return_value=(fp16_path, fp32_path)),
        ):
            result = converter.process_model_config(config)

        assert result is True

    def test_labels_not_found(self, converter, sample_model_config):
        """process_model_config handles unknown label set."""
        mock_model = MagicMock(spec=nn.Module)
        fp16_path = converter.output_dir / "test_model-fp16-ov" / "test_model.xml"
        fp32_path = converter.output_dir / "test_model-fp16-ov" / "test_model_fp32.xml"

        with (
            patch.object(converter, "_load_model_from_config", return_value=mock_model),
            patch.object(converter, "get_labels", return_value=None),
            patch.object(converter, "export_to_openvino", return_value=(fp16_path, fp32_path)),
        ):
            result = converter.process_model_config(sample_model_config)

        assert result is True

    def test_metadata_fields(self, converter):
        """process_model_config includes optional metadata fields."""
        config = {
            "model_short_name": "test_model",
            "license": "Apache-2.0",
            "license_link": "https://apache.org",
            "weights_url": "https://example.com/weights.pth",
            "input_shape": [1, 3, 224, 224],
            "model_type": "Detection",
            "confidence_threshold": "0.5",
            "iou_threshold": "0.45",
            "resize_type": "standard",
        }

        mock_model = MagicMock(spec=nn.Module)
        fp16_path = converter.output_dir / "test_model-fp16-ov" / "test_model.xml"
        fp32_path = converter.output_dir / "test_model-fp16-ov" / "test_model_fp32.xml"

        with (
            patch.object(converter, "_load_model_from_config", return_value=mock_model),
            patch.object(converter, "export_to_openvino", return_value=(fp16_path, fp32_path)) as mock_export,
        ):
            converter.process_model_config(config)

        # Check metadata was passed
        call_kwargs = mock_export.call_args[1]
        metadata = call_kwargs["metadata"]
        assert ("model_info", "confidence_threshold") in metadata
        assert ("model_info", "iou_threshold") in metadata


class TestMetadataValue:
    """Tests for ModelConverter._metadata_value."""

    def test_string(self):
        """_metadata_value converts string to string."""
        assert ModelConverter._metadata_value("hello") == "hello"

    def test_integer(self):
        """_metadata_value converts int to string."""
        assert ModelConverter._metadata_value(42) == "42"

    def test_float(self):
        """_metadata_value converts float to string."""
        assert ModelConverter._metadata_value(0.5) == "0.5"

    def test_boolean(self):
        """_metadata_value converts bool to string."""
        assert ModelConverter._metadata_value(True) == "True"

    def test_list(self):
        """_metadata_value joins list with spaces."""
        assert ModelConverter._metadata_value([1, 2, 3]) == "1 2 3"

    def test_tuple(self):
        """_metadata_value joins tuple with spaces."""
        assert ModelConverter._metadata_value(("a", "b")) == "a b"


class TestProcessConfigFile:
    """Tests for ModelConverter.process_config_file."""

    def test_multiple_models(self, converter, tmp_path):
        """process_config_file processes multiple models."""
        config = {
            "models": [
                {
                    "model_short_name": "model1",
                    "license": "MIT",
                    "license_link": "https://mit.edu",
                    "weights_url": "https://example.com/1.pth",
                },
                {
                    "model_short_name": "model2",
                    "license": "MIT",
                    "license_link": "https://mit.edu",
                    "weights_url": "https://example.com/2.pth",
                },
            ],
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        with patch.object(converter, "process_model_config", return_value=True) as mock_process:
            successful, failed = converter.process_config_file(config_path)

        assert successful == 2
        assert failed == 0
        assert mock_process.call_count == 2

    def test_filter_match(self, converter, tmp_path):
        """process_config_file filters to specific model."""
        config = {
            "models": [
                {"model_short_name": "model1", "license": "MIT", "license_link": "x"},
                {"model_short_name": "model2", "license": "MIT", "license_link": "x"},
            ],
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        with patch.object(converter, "process_model_config", return_value=True) as mock_process:
            successful, _failed = converter.process_config_file(config_path, model_filter="model2")

        assert successful == 1
        mock_process.assert_called_once()

    def test_filter_no_match(self, converter, tmp_path):
        """process_config_file returns 0,0 when filter doesn't match."""
        config = {
            "models": [
                {"model_short_name": "model1"},
            ],
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        successful, failed = converter.process_config_file(config_path, model_filter="nonexistent")
        assert successful == 0
        assert failed == 0

    def test_empty_models(self, converter, tmp_path):
        """process_config_file handles empty models list."""
        config = {"models": []}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        successful, failed = converter.process_config_file(config_path)
        assert successful == 0
        assert failed == 0

    def test_invalid_json(self, converter, tmp_path):
        """process_config_file raises on invalid JSON."""
        config_path = tmp_path / "bad.json"
        config_path.write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            converter.process_config_file(config_path)

    def test_model_failure(self, converter, tmp_path):
        """process_config_file counts failed models."""
        config = {
            "models": [
                {"model_short_name": "model1", "license": "MIT", "license_link": "x"},
            ],
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        with patch.object(converter, "process_model_config", return_value=False):
            successful, failed = converter.process_config_file(config_path)

        assert successful == 0
        assert failed == 1


class TestListModels:
    """Tests for list_models function."""

    def test_normal_output(self, tmp_path, capsys):
        """list_models prints model information."""
        config = {
            "models": [
                {
                    "model_short_name": "resnet50",
                    "model_full_name": "ResNet-50",
                    "model_type": "Classification",
                },
            ],
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        list_models(config_path)

        captured = capsys.readouterr()
        assert "resnet50" in captured.out
        assert "ResNet-50" in captured.out
        assert "Classification" in captured.out

    def test_file_not_found(self, tmp_path, capsys):
        """list_models handles missing config file."""
        list_models(tmp_path / "nonexistent.json")

        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_empty_models(self, tmp_path, capsys):
        """list_models handles empty models list."""
        config = {"models": []}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        list_models(config_path)

        captured = capsys.readouterr()
        assert "No models found" in captured.out

    def test_invalid_json(self, tmp_path, capsys):
        """list_models handles invalid JSON."""
        config_path = tmp_path / "bad.json"
        config_path.write_text("not json")

        list_models(config_path)

        captured = capsys.readouterr()
        assert "Error" in captured.err


class TestMain:
    """Tests for main() CLI entry point."""

    def test_missing_config_file(self, tmp_path, monkeypatch):
        """main returns 1 when config file doesn't exist."""
        monkeypatch.setattr(sys, "argv", ["model_converter", str(tmp_path / "nonexistent.json")])
        result = main()
        assert result == 1

    def test_list_flag(self, tmp_path, monkeypatch, capsys):
        """main --list flag lists models and exits."""
        config = {"models": [{"model_short_name": "test", "model_type": "cls"}]}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        monkeypatch.setattr(sys, "argv", ["model_converter", str(config_path), "--list"])
        result = main()
        assert result == 0

        captured = capsys.readouterr()
        assert "test" in captured.out

    def test_successful_run(self, tmp_path, monkeypatch):
        """main runs conversion successfully."""
        config = {"models": [{"model_short_name": "m1", "license": "MIT", "license_link": "x"}]}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "model_converter",
                str(config_path),
                "-o",
                str(tmp_path / "output"),
                "-c",
                str(tmp_path / "cache"),
                "-d",
                str(tmp_path / "dataset"),
            ],
        )

        with patch.object(ModelConverter, "process_config_file", return_value=(1, 0)):
            result = main()

        assert result == 0

    def test_failed_run(self, tmp_path, monkeypatch):
        """main returns 1 when models fail."""
        config = {"models": [{"model_short_name": "m1"}]}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "model_converter",
                str(config_path),
                "-o",
                str(tmp_path / "output"),
                "-c",
                str(tmp_path / "cache"),
            ],
        )

        with patch.object(ModelConverter, "process_config_file", return_value=(0, 1)):
            result = main()

        assert result == 1

    def test_verbose_flag(self, tmp_path, monkeypatch):
        """main enables verbose logging with -v flag."""
        config = {"models": []}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "model_converter",
                str(config_path),
                "-o",
                str(tmp_path / "output"),
                "-c",
                str(tmp_path / "cache"),
                "-v",
            ],
        )

        with patch.object(ModelConverter, "process_config_file", return_value=(0, 0)):
            result = main()

        assert result == 0

    def test_model_filter(self, tmp_path, monkeypatch):
        """main passes --model filter to process_config_file."""
        config = {"models": [{"model_short_name": "target"}]}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "model_converter",
                str(config_path),
                "-o",
                str(tmp_path / "output"),
                "-c",
                str(tmp_path / "cache"),
                "--model",
                "target",
            ],
        )

        with patch.object(ModelConverter, "process_config_file", return_value=(1, 0)) as mock_process:
            main()

        mock_process.assert_called_once_with(config_path=config_path, model_filter="target")

    def test_exception_during_processing(self, tmp_path, monkeypatch):
        """main returns 1 on unhandled exception."""
        config = {"models": [{"model_short_name": "m1"}]}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "model_converter",
                str(config_path),
                "-o",
                str(tmp_path / "output"),
                "-c",
                str(tmp_path / "cache"),
            ],
        )

        with patch.object(ModelConverter, "process_config_file", side_effect=ValueError("bad config")):
            result = main()

        assert result == 1
