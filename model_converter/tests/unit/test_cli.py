#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for the model_converter CLI facade and converter hierarchy."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from model_converter.cli import ModelConverter, list_models, main
from model_converter.converters.getitune import GetituneConverter
from model_converter.converters.timm import TimmConverter
from model_converter.converters.torchvision import TorchvisionConverter
from model_converter.reporting import AccuracyResults


class TestModelConverterInit:
    """Tests for facade ModelConverter.__init__."""

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
        """ModelConverter stores dataset registry."""
        dataset = tmp_path / "dataset"
        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_registry=dataset,
        )
        assert converter.dataset_registry == dataset

    def test_dataset_path_none(self, tmp_path):
        """ModelConverter handles None dataset registry."""
        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_registry=None,
        )
        assert converter.dataset_registry is None


class TestModelConverterDispatch:
    """Tests for facade dispatching logic."""

    def test_get_converter_returns_torchvision_and_caches(self, facade_converter):
        """_get_converter caches converter instances per library."""
        first = facade_converter._get_converter("torchvision")
        second = facade_converter._get_converter("torchvision")

        assert isinstance(first, TorchvisionConverter)
        assert first is second

    def test_get_converter_passes_training_extensions_dir(self, tmp_path):
        """_get_converter wires training_extensions_dir for getitune."""
        training_extensions_dir = tmp_path / "training_extensions"
        converter = ModelConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            training_extensions_dir=training_extensions_dir,
        )

        getitune_converter = converter._get_converter("getitune")

        assert isinstance(getitune_converter, GetituneConverter)
        assert getitune_converter.training_extensions_dir == training_extensions_dir

    def test_get_converter_rejects_unknown_library(self, facade_converter):
        """_get_converter raises for unsupported libraries."""
        with pytest.raises(ValueError, match="Unsupported model_library"):
            facade_converter._get_converter("unknown")

    def test_process_model_config_defaults_to_torchvision(self, facade_converter):
        """process_model_config uses torchvision when model_library is omitted."""
        config = {"model_short_name": "test_model"}

        with patch.object(TorchvisionConverter, "process_model_config", return_value=True) as mock_process:
            result = facade_converter.process_model_config(config)

        assert result is True
        mock_process.assert_called_once_with(config)

    def test_process_model_config_routes_to_timm(self, facade_converter):
        """process_model_config dispatches by model_library."""
        config = {"model_short_name": "test_model", "model_library": "timm"}

        with patch.object(TimmConverter, "process_model_config", return_value=True) as mock_process:
            result = facade_converter.process_model_config(config)

        assert result is True
        mock_process.assert_called_once_with(config)

    def test_process_model_config_returns_false_for_unknown_library(self, facade_converter):
        """process_model_config returns False when dispatch fails."""
        assert facade_converter.process_model_config({"model_library": "unsupported"}) is False


class TestGetLabels:
    """Tests for PyTorchConverter.get_labels via TorchvisionConverter."""

    def test_imagenet1k_v1(self, converter):
        """get_labels returns ImageNet1K labels with underscores."""
        with patch(
            "torchvision.models._meta._IMAGENET_CATEGORIES",
            ["tabby cat", "golden retriever"],
            create=True,
        ):
            result = converter.get_labels("IMAGENET1K_V1")

        assert result == "tabby_cat golden_retriever"

    def test_imagenet21k(self, converter):
        """get_labels returns ImageNet21K labels."""
        mock_info = MagicMock()
        mock_info.label_descriptions.return_value = ["tabby, tabby cat", "golden retriever, dog"]

        with patch("timm.data.ImageNetInfo", return_value=mock_info):
            result = converter.get_labels("IMAGENET21K")

        assert result == "tabby golden_retriever"

    def test_coco_v1(self, converter):
        """get_labels returns COCO labels."""
        mock_weights = MagicMock()
        mock_weights.COCO_V1.meta = {"categories": ["person", "bicycle", "car"]}

        with patch("torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights", mock_weights):
            result = converter.get_labels("COCO_V1")

        assert result == "person bicycle car"

    def test_unknown_label_set(self, converter):
        """get_labels returns None for unknown label sets."""
        assert converter.get_labels("NONEXISTENT_LABELS") is None


class TestLoadModelClass:
    """Tests for PyTorchConverter.load_model_class via TorchvisionConverter."""

    def test_successful_import(self, converter):
        """load_model_class dynamically imports a class."""
        assert converter.load_model_class("torch.nn.Linear") is torch.nn.Linear

    def test_import_failure(self, converter):
        """load_model_class raises on invalid path."""
        with pytest.raises(ModuleNotFoundError):
            converter.load_model_class("nonexistent.module.Class")


class TestLoadCheckpoint:
    """Tests for PyTorchConverter.load_checkpoint via TorchvisionConverter."""

    def test_successful_load(self, converter, tmp_path):
        """load_checkpoint loads a PyTorch checkpoint."""
        checkpoint_path = tmp_path / "checkpoint.pth"
        checkpoint_path.touch()
        state_dict = {"layer.weight": torch.randn(10, 10)}

        with patch("model_converter.converters.pytorch.torch.load", return_value=state_dict) as mock_load:
            result = converter.load_checkpoint(checkpoint_path)

        assert result == state_dict
        mock_load.assert_called_once_with(checkpoint_path, map_location="cpu", weights_only=True)

    def test_load_failure(self, converter, tmp_path):
        """load_checkpoint re-raises underlying torch loading errors."""
        with (
            patch(
                "model_converter.converters.pytorch.torch.load",
                side_effect=FileNotFoundError("missing"),
            ),
            pytest.raises(FileNotFoundError, match="missing"),
        ):
            converter.load_checkpoint(tmp_path / "nonexistent.pth")


class TestLoadHuggingfaceModel:
    """Tests for TimmConverter.load_huggingface_model."""

    def test_timm_model(self, timm_converter):
        """load_huggingface_model loads timm model."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        with patch("timm.create_model", return_value=mock_model) as mock_create:
            result = timm_converter.load_huggingface_model(
                repo_id="timm/resnet50",
                revision="abc123",
                model_library="timm",
            )

        assert result is mock_model
        mock_create.assert_called_once_with(
            "hf-hub:timm/resnet50@abc123",
            pretrained=True,
            cache_dir=timm_converter.cache_dir,
        )
        mock_model.eval.assert_called_once()

    def test_timm_model_with_params(self, timm_converter):
        """load_huggingface_model passes model_params to timm."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        with patch("timm.create_model", return_value=mock_model) as mock_create:
            timm_converter.load_huggingface_model(
                repo_id="timm/resnet50",
                revision="abc123",
                model_library="timm",
                model_params={"num_classes": 10},
            )

        assert mock_create.call_args.kwargs["num_classes"] == 10

    def test_transformers_model(self, timm_converter):
        """load_huggingface_model loads transformers model."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        with patch("transformers.AutoModel.from_pretrained", return_value=mock_model) as mock_from:
            result = timm_converter.load_huggingface_model(
                repo_id="bert-base-uncased",
                revision="abc123",
                model_library="transformers",
            )

        assert result is mock_model
        mock_from.assert_called_once_with(
            "bert-base-uncased",
            revision="abc123",
            cache_dir=timm_converter.cache_dir,
        )

    def test_unsupported_library(self, timm_converter):
        """load_huggingface_model raises for unsupported library."""
        with pytest.raises(ValueError, match="Unsupported model library"):
            timm_converter.load_huggingface_model(
                repo_id="some/model",
                revision="abc123",
                model_library="unsupported_lib",
            )

    def test_load_failure(self, timm_converter):
        """load_huggingface_model re-raises loading errors."""
        with (
            patch("timm.create_model", side_effect=RuntimeError("Connection error")),
            pytest.raises(RuntimeError, match="Connection error"),
        ):
            timm_converter.load_huggingface_model(
                repo_id="timm/resnet50",
                revision="abc123",
                model_library="timm",
            )


class TestApplyTimmDataConfig:
    """Tests for TimmConverter._apply_timm_data_config."""

    def test_overrides_mean_scale_and_shape(self, timm_converter):
        """Resolved timm mean/std (0..1) are scaled to 0..255 and shape is applied."""
        model = MagicMock()
        config = {
            "mean_values": "123.675 116.28 103.53",
            "scale_values": "58.395 57.12 57.375",
            "input_shape": [1, 3, 224, 224],
        }
        resolved = {
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
            "input_size": (3, 256, 256),
        }

        with patch("timm.data.resolve_data_config", return_value=resolved):
            timm_converter._apply_timm_data_config(model, config)

        assert config["mean_values"] == "127.5 127.5 127.5"
        assert config["scale_values"] == "127.5 127.5 127.5"
        assert config["input_shape"] == [1, 3, 256, 256]
        assert config["reverse_input_channels"] is True

    def test_keeps_imagenet_values_when_model_matches(self, timm_converter):
        """Standard ImageNet normalization round-trips to the canonical strings."""
        model = MagicMock()
        config = {}
        resolved = {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "input_size": (3, 224, 224),
        }

        with patch("timm.data.resolve_data_config", return_value=resolved):
            timm_converter._apply_timm_data_config(model, config)

        assert config["mean_values"] == "123.675 116.28 103.53"
        assert config["scale_values"] == "58.395 57.12 57.375"
        assert config["input_shape"] == [1, 3, 224, 224]

    def test_ignores_missing_fields(self, timm_converter):
        """Fields timm does not provide leave the existing config untouched."""
        model = MagicMock()
        config = {"mean_values": "1 2 3", "scale_values": "4 5 6", "input_shape": [1, 3, 8, 8]}

        with patch("timm.data.resolve_data_config", return_value={}):
            timm_converter._apply_timm_data_config(model, config)

        assert config == {
            "mean_values": "1 2 3",
            "scale_values": "4 5 6",
            "input_shape": [1, 3, 8, 8],
            "reverse_input_channels": True,
        }

    def test_forces_reverse_input_channels(self, timm_converter):
        """reverse_input_channels is always forced to True for timm (RGB) models."""
        model = MagicMock()
        config = {"reverse_input_channels": False}

        with patch("timm.data.resolve_data_config", return_value={}):
            timm_converter._apply_timm_data_config(model, config)

        assert config["reverse_input_channels"] is True

    def test_ignores_malformed_input_size(self, timm_converter):
        """A non 3-tuple input_size is ignored, leaving input_shape untouched."""
        model = MagicMock()
        config = {"input_shape": [1, 3, 8, 8]}

        with patch("timm.data.resolve_data_config", return_value={"input_size": (224, 224)}):
            timm_converter._apply_timm_data_config(model, config)

        assert config["input_shape"] == [1, 3, 8, 8]

    def test_handles_resolution_failure(self, timm_converter):
        """A failure inside resolve_data_config keeps the configured values."""
        model = MagicMock()
        config = {"mean_values": "1 2 3"}

        with patch("timm.data.resolve_data_config", side_effect=RuntimeError("boom")):
            timm_converter._apply_timm_data_config(model, config)

        assert config["mean_values"] == "1 2 3"

    def test_handles_missing_timm(self, timm_converter):
        """If timm.data cannot be imported, configured values are kept."""
        model = MagicMock()
        config = {"mean_values": "1 2 3"}

        with patch.dict(sys.modules, {"timm.data": None}):
            timm_converter._apply_timm_data_config(model, config)

        assert config["mean_values"] == "1 2 3"
        assert config["reverse_input_channels"] is True


class TestCreateModel:
    """Tests for PyTorchConverter.create_model via TorchvisionConverter."""

    def test_nn_module_with_model_key(self, converter):
        """create_model extracts model from checkpoint model key."""
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        result = converter.create_model(torch.nn.Module, {"model": mock_model})
        assert result is mock_model

    def test_nn_module_with_state_dict_only_raises(self, converter):
        """create_model raises when nn.Module is used with only state_dict."""
        with pytest.raises(ValueError, match="state_dict"):
            converter.create_model(torch.nn.Module, {"state_dict": {"layer.weight": torch.randn(10)}})

    def test_nn_module_direct_model_as_checkpoint(self, converter):
        """create_model surfaces invalid checkpoint types."""
        with pytest.raises(TypeError):
            converter.create_model(torch.nn.Module, nn.Linear(10, 10))

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

        result = converter.create_model(mock_class, {"state_dict": state_dict})
        assert result is mock_instance
        mock_instance.load_state_dict.assert_called_once_with(state_dict, strict=False)

    def test_model_class_with_model_params(self, converter):
        """create_model passes model_params to class constructor."""
        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.eval.return_value = mock_instance
        mock_class.return_value = mock_instance

        converter.create_model(
            mock_class,
            {"state_dict": {"w": torch.randn(10)}},
            model_params={"num_classes": 5},
        )
        mock_class.assert_called_once_with(num_classes=5)

    def test_model_class_with_model_key_as_module(self, converter):
        """create_model returns checkpoint model when it already is an nn.Module."""
        mock_model = MagicMock(spec=nn.Module)
        mock_class = MagicMock()

        result = converter.create_model(mock_class, {"model": mock_model})
        assert result is mock_model

    def test_model_class_with_model_key_as_dict(self, converter):
        """create_model uses checkpoint model dict as state_dict."""
        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.eval.return_value = mock_instance
        mock_class.return_value = mock_instance

        result = converter.create_model(mock_class, {"model": {"layer.weight": torch.randn(10)}})
        assert result is mock_instance
        mock_instance.load_state_dict.assert_called_once()

    def test_model_class_bare_state_dict(self, converter):
        """create_model uses checkpoint directly as state_dict."""
        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.eval.return_value = mock_instance
        mock_class.return_value = mock_instance

        result = converter.create_model(mock_class, {"layer.weight": torch.randn(10)})
        assert result is mock_instance
        mock_instance.load_state_dict.assert_called_once()

    def test_create_model_failure(self, converter):
        """create_model raises on instantiation failure."""
        with pytest.raises(RuntimeError, match="init failed"):
            converter.create_model(MagicMock(side_effect=RuntimeError("init failed")), {})


class TestExportToOpenvino:
    """Tests for PyTorchConverter.export_to_openvino via TorchvisionConverter."""

    def test_successful_export(self, converter, sample_model_config):
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

        with (
            patch("openvino.convert_model", return_value=mock_ov_model),
            patch("openvino.save_model") as mock_save,
            patch.object(Path, "exists", return_value=True),
            patch("model_converter.converters.pytorch.shutil.copy2"),
            patch.object(converter, "copy_readme"),
        ):
            fp16_path, fp32_path = converter.export_to_openvino(
                model=mock_model,
                input_shape=[1, 3, 224, 224],
                output_path=converter.output_dir / "test_model",
                model_config=sample_model_config,
                input_names=["input"],
                output_names=["result"],
                metadata={("model_info", "model_type"): "Classification"},
            )

        assert fp16_path == converter.output_dir / "test_model-fp16-ov" / "test_model.xml"
        assert fp32_path == converter.output_dir / "test_model-fp16-ov" / "test_model_fp32.xml"
        assert mock_save.call_count == 2

    def test_export_failure(self, converter, sample_model_config):
        """export_to_openvino raises on conversion failure."""
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        with (
            patch("openvino.convert_model", side_effect=RuntimeError("Conversion failed")),
            pytest.raises(RuntimeError, match="Conversion failed"),
        ):
            converter.export_to_openvino(
                model=mock_model,
                input_shape=[1, 3, 224, 224],
                output_path=converter.output_dir / "test_model",
                model_config=sample_model_config,
            )


class TestPrepareModelForExport:
    """Tests for PyTorchConverter._prepare_model_for_export via TorchvisionConverter."""

    def test_with_adapter(self, converter):
        """_prepare_model_for_export applies adapter for known model type."""
        mock_model = MagicMock(spec=nn.Module)

        with patch("model_converter.converters.pytorch.get_adapter") as mock_get_adapter:
            mock_adapted = MagicMock()
            mock_get_adapter.return_value = mock_adapted
            result = converter._prepare_model_for_export(mock_model, {"model_type": "MaskRCNN"})

        assert result is mock_adapted

    def test_without_adapter(self, converter):
        """_prepare_model_for_export returns model unchanged without adapter."""
        mock_model = MagicMock(spec=nn.Module)

        with patch("model_converter.converters.pytorch.get_adapter", return_value=mock_model):
            result = converter._prepare_model_for_export(mock_model, {"model_type": "Classification"})

        assert result is mock_model


class TestCreateExampleInput:
    """Tests for PyTorchConverter._create_example_input via TorchvisionConverter."""

    def test_maskrcnn_input(self, converter):
        """_create_example_input uses rand for MaskRCNN."""
        result = converter._create_example_input([1, 3, 224, 224], {"model_type": "MaskRCNN"})
        assert result.shape == (1, 3, 224, 224)
        assert result.min() >= 0

    def test_default_input(self, converter):
        """_create_example_input uses randn for other model types."""
        assert converter._create_example_input([1, 3, 224, 224], {"model_type": "Classification"}).shape == (
            1,
            3,
            224,
            224,
        )


class TestPostprocessOpenvinoModel:
    """Tests for PyTorchConverter._postprocess_openvino_model via TorchvisionConverter."""

    def test_set_input_names(self, converter, mock_ov_model):
        """_postprocess_openvino_model sets input tensor names."""
        converter._postprocess_openvino_model(mock_ov_model, input_names=["images"])
        mock_ov_model.input(0).set_names.assert_called_with({"images"})

    def test_set_output_names(self, converter, mock_ov_model):
        """_postprocess_openvino_model sets output tensor names."""
        converter._postprocess_openvino_model(mock_ov_model, output_names=["predictions"])
        mock_ov_model.output(0).set_names.assert_called_with({"predictions"})

    def test_set_metadata(self, converter, mock_ov_model):
        """_postprocess_openvino_model adds metadata."""
        metadata = {("model_info", "model_type"): "Classification", ("model_info", "labels"): "cat dog"}
        converter._postprocess_openvino_model(mock_ov_model, metadata=metadata)
        assert mock_ov_model.set_rt_info.call_count == 2

    def test_no_operations(self, converter, mock_ov_model):
        """_postprocess_openvino_model handles None params."""
        assert converter._postprocess_openvino_model(mock_ov_model) is mock_ov_model


class TestBuildMetadata:
    """Tests for PyTorchConverter._build_metadata via TorchvisionConverter."""

    def test_builds_default_fields(self, converter, sample_model_config):
        """_build_metadata includes shared Model API metadata."""
        metadata = converter._build_metadata({**sample_model_config, "labels": None})

        assert metadata["model_info", "model_type"] == "Classification"
        assert metadata["model_info", "model_short_name"] == "test_model"
        assert metadata["model_info", "reverse_input_channels"] == "True"
        assert metadata["model_info", "mean_values"] == "123.675 116.28 103.53"
        assert metadata["model_info", "scale_values"] == "58.395 57.12 57.375"

    def test_includes_optional_fields(self, converter, sample_model_config):
        """_build_metadata includes optional metadata fields when configured."""
        metadata = converter._build_metadata(
            {
                **sample_model_config,
                "labels": None,
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "resize_type": "standard",
            },
        )

        assert metadata["model_info", "confidence_threshold"] == "0.5"
        assert metadata["model_info", "iou_threshold"] == "0.45"
        assert metadata["model_info", "resize_type"] == "standard"

    def test_adds_resolved_labels(self, converter, sample_model_config):
        """_build_metadata adds labels when they can be resolved."""
        with patch.object(converter, "get_labels", return_value="cat dog"):
            metadata = converter._build_metadata(sample_model_config)

        assert metadata["model_info", "labels"] == "cat dog"

    def test_skips_unknown_label_set(self, converter, sample_model_config):
        """_build_metadata omits labels when lookup fails."""
        with patch.object(converter, "get_labels", return_value=None):
            metadata = converter._build_metadata(sample_model_config)

        assert ("model_info", "labels") not in metadata


class TestQuantizeAndCleanup:
    """Tests for PyTorchConverter._quantize_and_cleanup via TorchvisionConverter."""

    def test_with_classification_labels(self, converter, tmp_path, sample_model_config):
        """_quantize_and_cleanup validates classification models with labels."""
        fp32_path = tmp_path / "model_fp32.xml"
        fp32_path.write_text("<net/>")
        fp32_bin = tmp_path / "model_fp32.bin"
        fp32_bin.write_text("weights")
        validation_data = [np.zeros((1, 3, 224, 224))]
        validation_labels = [0]

        with (
            patch.object(converter, "create_calibration_dataset", return_value=(validation_data, validation_labels)),
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

        call_kwargs = mock_quantize.call_args.kwargs
        assert call_kwargs["validation_data"] == validation_data
        assert call_kwargs["validation_labels"] == validation_labels
        assert not fp32_path.exists()
        assert not fp32_bin.exists()

    def test_measures_original_accuracy_with_torch_model(self, converter, tmp_path, sample_model_config):
        """_quantize_and_cleanup measures the original PyTorch model accuracy when a model is given."""
        from model_converter.reporting import AccuracyResults

        fp32_path = tmp_path / "model_fp32.xml"
        fp32_path.write_text("<net/>")
        validation_data = [np.zeros((1, 3, 224, 224))]
        validation_labels = [0]
        torch_model = MagicMock()

        with (
            patch.object(converter, "create_calibration_dataset", return_value=(validation_data, validation_labels)),
            patch.object(converter, "validate_torch_model", return_value=0.91) as mock_validate,
            patch.object(converter, "quantize_model") as mock_quantize,
        ):
            accuracy = converter._quantize_and_cleanup(
                sample_model_config,
                fp32_path,
                model_type="Classification",
                input_shape=[1, 3, 224, 224],
                mean_values="123.675 116.28 103.53",
                scale_values="58.395 57.12 57.375",
                reverse_input_channels=True,
                torch_model=torch_model,
            )

        mock_validate.assert_called_once_with(torch_model, validation_data, validation_labels)
        assert isinstance(accuracy, AccuracyResults)
        assert accuracy.original_accuracy == pytest.approx(0.91)
        assert accuracy.measured is True
        # quantize_model still receives the validation collector.
        assert mock_quantize.call_args.kwargs["accuracy_results"] is accuracy

    def test_skips_original_accuracy_when_torch_validation_fails(self, converter, tmp_path, sample_model_config):
        """A failed PyTorch validation leaves the original accuracy unset."""
        fp32_path = tmp_path / "model_fp32.xml"
        fp32_path.write_text("<net/>")
        validation_data = [np.zeros((1, 3, 224, 224))]
        validation_labels = [0]

        with (
            patch.object(converter, "create_calibration_dataset", return_value=(validation_data, validation_labels)),
            patch.object(converter, "validate_torch_model", return_value=None) as mock_validate,
            patch.object(converter, "quantize_model"),
        ):
            accuracy = converter._quantize_and_cleanup(
                sample_model_config,
                fp32_path,
                model_type="Classification",
                input_shape=[1, 3, 224, 224],
                mean_values="123.675 116.28 103.53",
                scale_values="58.395 57.12 57.375",
                reverse_input_channels=True,
                torch_model=MagicMock(),
            )

        mock_validate.assert_called_once()
        assert accuracy.original_accuracy is None

    def test_without_classification_labels(self, converter, tmp_path, sample_model_config):
        """_quantize_and_cleanup skips validation for non-classification models."""
        fp32_path = tmp_path / "model_fp32.xml"
        fp32_path.write_text("<net/>")

        with (
            patch.object(converter, "create_calibration_dataset", return_value=([np.zeros((1, 3, 224, 224))], [])),
            patch.object(converter, "quantize_model") as mock_quantize,
        ):
            converter._quantize_and_cleanup(
                {**sample_model_config, "labels": None},
                fp32_path,
                model_type="Detection",
                input_shape=[1, 3, 224, 224],
                mean_values="123.675 116.28 103.53",
                scale_values="58.395 57.12 57.375",
                reverse_input_channels=True,
            )

        call_kwargs = mock_quantize.call_args.kwargs
        assert call_kwargs["validation_data"] is None
        assert call_kwargs["validation_labels"] is None

    def test_empty_calibration_data(self, converter, tmp_path, sample_model_config):
        """_quantize_and_cleanup skips quantization when no calibration data is produced."""
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

    def test_collects_validation_samples_for_non_top1_metric(
        self,
        converter,
        tmp_path,
        sample_model_config,
    ):
        """When the dispatched metric is non-Top1, raw image samples are forwarded to quantize_model."""
        from model_converter.datasets import CalibrationSample
        from model_converter.metrics import MultilabelMAP

        fp32_path = tmp_path / "model_fp32.xml"
        fp32_path.write_text("<net/>")
        samples = [CalibrationSample(image_path=tmp_path / "x.jpg", label=0)]

        with (
            patch.object(
                converter,
                "_metric_for_config",
                return_value=MultilabelMAP(num_labels=3),
            ),
            patch.object(
                converter,
                "_collect_validation_samples",
                return_value=samples,
            ) as mock_collect,
            patch.object(
                converter,
                "create_calibration_dataset",
                return_value=([np.zeros((1, 3, 224, 224))], []),
            ),
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

        mock_collect.assert_called_once()
        assert mock_quantize.call_args.kwargs["validation_samples"] == samples
        assert isinstance(mock_quantize.call_args.kwargs["metric"], MultilabelMAP)

    def test_cleanup_failure(self, converter, tmp_path, sample_model_config):
        """_quantize_and_cleanup swallows cleanup errors."""
        fp32_path = tmp_path / "model_fp32.xml"
        fp32_path.write_text("<net/>")

        with (
            patch.object(converter, "create_calibration_dataset", return_value=([], [])),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "unlink", side_effect=OSError("Permission denied")),
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


class TestTorchvisionProcessModelConfig:
    """Tests for TorchvisionConverter.process_model_config."""

    def test_already_exists(self, converter, sample_model_config):
        """process_model_config skips when both models already exist."""
        fp16_dir = converter.output_dir / "test_model-fp16-ov"
        fp16_dir.mkdir(parents=True)
        (fp16_dir / "test_model.xml").write_text("<net/>")

        int8_dir = converter.output_dir / "test_model-int8-ov"
        int8_dir.mkdir(parents=True)
        (int8_dir / "test_model.xml").write_text("<net/>")

        assert converter.process_model_config(sample_model_config) is True

    def test_missing_license(self, converter):
        """process_model_config fails when license is missing."""
        assert (
            converter.process_model_config({"model_short_name": "test", "license_link": "https://example.com"}) is False
        )

    def test_missing_license_link(self, converter):
        """process_model_config fails when license_link is missing."""
        assert converter.process_model_config({"model_short_name": "test", "license": "MIT"}) is False

    def test_successful_conversion(self, converter, sample_model_config, tmp_path):
        """process_model_config runs the torchvision conversion workflow."""
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model
        fp16_path = converter.output_dir / "test_model-fp16-ov" / "test_model.xml"
        fp32_path = converter.output_dir / "test_model-fp16-ov" / "test_model_fp32.xml"

        with (
            patch.object(converter._url_downloader, "download", return_value=tmp_path / "weights.pth"),
            patch.object(converter, "load_model_class", return_value=torch.nn.Module),
            patch.object(converter, "load_checkpoint", return_value={"model": mock_model}),
            patch.object(converter, "create_model", return_value=mock_model),
            patch.object(converter, "_build_metadata", return_value={("model_info", "model_type"): "Classification"}),
            patch.object(converter, "export_to_openvino", return_value=(fp16_path, fp32_path)) as mock_export,
        ):
            result = converter.process_model_config(sample_model_config)

        assert result is True
        assert mock_export.call_args.kwargs["output_path"] == converter.output_dir / "test_model"

    def test_successful_conversion_with_dataset(self, tmp_path, sample_model_config, dataset_dir):
        """process_model_config quantizes when a dataset is available."""
        converter = TorchvisionConverter(
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            dataset_registry=dataset_dir,
        )
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model
        fp16_path = converter.output_dir / "test_model-fp16-ov" / "test_model.xml"
        fp32_path = converter.output_dir / "test_model-fp16-ov" / "test_model_fp32.xml"

        with (
            patch.object(converter._url_downloader, "download", return_value=tmp_path / "weights.pth"),
            patch.object(converter, "load_model_class", return_value=torch.nn.Module),
            patch.object(converter, "load_checkpoint", return_value={"model": mock_model}),
            patch.object(converter, "create_model", return_value=mock_model),
            patch.object(converter, "_build_metadata", return_value={("model_info", "model_type"): "Classification"}),
            patch.object(converter, "export_to_openvino", return_value=(fp16_path, fp32_path)),
            patch.object(converter, "_quantize_and_cleanup", return_value=AccuracyResults()) as mock_quantize,
        ):
            result = converter.process_model_config(sample_model_config)

        assert result is True
        mock_quantize.assert_called_once()

    def test_conversion_failure(self, converter, sample_model_config):
        """process_model_config returns False on failure."""
        with patch.object(converter._url_downloader, "download", side_effect=RuntimeError("download failed")):
            assert converter.process_model_config(sample_model_config) is False


class TestTimmProcessModelConfig:
    """Tests for TimmConverter.process_model_config."""

    def test_missing_huggingface_repo(self, timm_converter, sample_timm_config):
        """process_model_config fails when huggingface_repo is missing."""
        assert (
            timm_converter.process_model_config({
                k: v for k, v in sample_timm_config.items() if k != "huggingface_repo"
            })
            is False
        )

    def test_missing_huggingface_revision(self, timm_converter, sample_timm_config):
        """process_model_config fails when huggingface_revision is missing."""
        assert (
            timm_converter.process_model_config({
                k: v for k, v in sample_timm_config.items() if k != "huggingface_revision"
            })
            is False
        )

    def test_successful_conversion(self, timm_converter, sample_timm_config):
        """process_model_config runs the timm conversion workflow."""
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model
        fp16_path = timm_converter.output_dir / "test_timm_model-fp16-ov" / "test_timm_model.xml"
        fp32_path = timm_converter.output_dir / "test_timm_model-fp16-ov" / "test_timm_model_fp32.xml"

        with (
            patch.object(timm_converter, "load_huggingface_model", return_value=mock_model),
            patch.object(
                timm_converter,
                "_build_metadata",
                return_value={("model_info", "model_type"): "Classification"},
            ),
            patch.object(timm_converter, "export_to_openvino", return_value=(fp16_path, fp32_path)) as mock_export,
        ):
            result = timm_converter.process_model_config(sample_timm_config)

        assert result is True
        assert mock_export.call_args.kwargs["output_path"] == timm_converter.output_dir / "test_timm_model"


class TestMetadataValue:
    """Tests for BaseConverter._metadata_value via TorchvisionConverter."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("hello", "hello"), (42, "42"), (0.5, "0.5"), (True, "True"), ([1, 2, 3], "1 2 3"), (("a", "b"), "a b")],
    )
    def test_value_conversion(self, converter, value, expected):
        """_metadata_value converts scalars and iterables to metadata strings."""
        assert converter._metadata_value(value) == expected


class TestProcessConfigFile:
    """Tests for facade ModelConverter.process_config_file."""

    def test_collect_results_aggregates_converters(self, facade_converter):
        """collect_results concatenates results from every instantiated converter."""
        from model_converter.reporting import ConversionResult

        assert facade_converter.collect_results() == []

        converter = facade_converter._get_converter("torchvision")
        result = ConversionResult(
            model_short_name="m",
            model_full_name="M",
            model_type="Classification",
            model_library="torchvision",
        )
        converter.results.append(result)

        assert facade_converter.collect_results() == [result]

    def test_multiple_models(self, facade_converter, tmp_path):
        """process_config_file processes multiple models."""
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "models": [
                        {"model_short_name": "model1", "license": "MIT", "license_link": "https://mit.edu"},
                        {"model_short_name": "model2", "license": "MIT", "license_link": "https://mit.edu"},
                    ],
                },
            ),
        )

        with patch.object(facade_converter, "process_model_config", return_value=True) as mock_process:
            successful, failed = facade_converter.process_config_file(config_path)

        assert successful == 2
        assert failed == 0
        assert mock_process.call_count == 2

    def test_filter_match(self, facade_converter, tmp_path):
        """process_config_file filters to a specific model."""
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "models": [
                        {"model_short_name": "model1", "license": "MIT", "license_link": "x"},
                        {"model_short_name": "model2", "license": "MIT", "license_link": "x"},
                    ],
                },
            ),
        )

        with patch.object(facade_converter, "process_model_config", return_value=True) as mock_process:
            successful, _failed = facade_converter.process_config_file(config_path, model_filter="model2")

        assert successful == 1
        mock_process.assert_called_once()

    def test_filter_no_match(self, facade_converter, tmp_path):
        """process_config_file returns 0,0 when filter does not match."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": [{"model_short_name": "model1"}]}))

        successful, failed = facade_converter.process_config_file(config_path, model_filter="nonexistent")
        assert successful == 0
        assert failed == 0

    def test_empty_models(self, facade_converter, tmp_path):
        """process_config_file handles empty models list."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": []}))

        successful, failed = facade_converter.process_config_file(config_path)
        assert successful == 0
        assert failed == 0

    def test_invalid_json(self, facade_converter, tmp_path):
        """process_config_file raises on invalid JSON."""
        config_path = tmp_path / "bad.json"
        config_path.write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            facade_converter.process_config_file(config_path)

    def test_model_failure(self, facade_converter, tmp_path):
        """process_config_file counts failed models."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": [{"model_short_name": "model1"}]}))

        with patch.object(facade_converter, "process_model_config", return_value=False):
            successful, failed = facade_converter.process_config_file(config_path)

        assert successful == 0
        assert failed == 1


class TestListModels:
    """Tests for list_models function."""

    def test_normal_output(self, tmp_path, capsys):
        """list_models prints model information."""
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "model_short_name": "resnet50",
                            "model_full_name": "ResNet-50",
                            "model_library": "torchvision",
                            "model_type": "Classification",
                        },
                    ],
                },
            ),
        )

        list_models(config_path)

        captured = capsys.readouterr()
        assert "resnet50" in captured.out
        assert "ResNet-50" in captured.out
        assert "torchvision" in captured.out
        assert "Classification" in captured.out

    def test_file_not_found(self, tmp_path, capsys):
        """list_models handles missing config file."""
        list_models(tmp_path / "nonexistent.json")
        assert "Error" in capsys.readouterr().err

    def test_empty_models(self, tmp_path, capsys):
        """list_models handles empty models list."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": []}))

        list_models(config_path)
        assert "No models found" in capsys.readouterr().out

    def test_invalid_json(self, tmp_path, capsys):
        """list_models handles invalid JSON."""
        config_path = tmp_path / "bad.json"
        config_path.write_text("not json")

        list_models(config_path)
        assert "Error" in capsys.readouterr().err


class TestMain:
    """Tests for main() CLI entry point."""

    def test_missing_config_file(self, tmp_path, monkeypatch):
        """main returns 1 when config file does not exist."""
        monkeypatch.setattr(sys, "argv", ["model_converter", str(tmp_path / "nonexistent.json")])
        assert main() == 1

    def test_list_flag(self, tmp_path, monkeypatch, capsys):
        """main --list lists models and exits."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": [{"model_short_name": "test", "model_type": "cls"}]}))

        monkeypatch.setattr(sys, "argv", ["model_converter", str(config_path), "--list"])
        assert main() == 0
        assert "test" in capsys.readouterr().out

    def test_successful_run(self, tmp_path, monkeypatch):
        """main runs conversion successfully."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": [{"model_short_name": "m1"}]}))

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

        with patch.object(ModelConverter, "process_config_file", return_value=(1, 0)):
            assert main() == 0

    def test_datasets_config_not_found_warns(self, tmp_path, monkeypatch, caplog):
        """main warns and continues when --datasets-config path does not exist."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": [{"model_short_name": "m1"}]}))
        missing_datasets = tmp_path / "missing_datasets.json"

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
                "--datasets-config",
                str(missing_datasets),
            ],
        )

        with patch.object(ModelConverter, "process_config_file", return_value=(1, 0)):
            assert main() == 0
        assert "Dataset configuration not found" in caplog.text

    def test_datasets_config_invalid_warns(self, tmp_path, monkeypatch, caplog):
        """main warns and continues when --datasets-config contains invalid content."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": [{"model_short_name": "m1"}]}))
        bad_datasets = tmp_path / "bad_datasets.json"
        bad_datasets.write_text('{"not_datasets": {}}')

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
                "--datasets-config",
                str(bad_datasets),
            ],
        )

        with patch.object(ModelConverter, "process_config_file", return_value=(1, 0)):
            assert main() == 0
        assert "Failed to load dataset registry" in caplog.text

    def test_report_default_path(self, tmp_path, monkeypatch, capsys):
        """main prints console table even without --report (reporting is on by default)."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": [{"model_short_name": "m1"}]}))

        monkeypatch.setattr(
            sys,
            "argv",
            ["model_converter", str(config_path), "-o", str(tmp_path / "output"), "-c", str(tmp_path / "cache")],
        )

        from model_converter.reporting import STATUS_OK, ConversionResult

        result = ConversionResult(
            model_short_name="m1",
            model_full_name="M1",
            model_type="Classification",
            model_library="torchvision",
            status=STATUS_OK,
        )

        with (
            patch.object(ModelConverter, "process_config_file", return_value=(1, 0)),
            patch.object(ModelConverter, "collect_results", return_value=[result]),
        ):
            assert main() == 0

        assert "Conversion Summary Report" in capsys.readouterr().out

    def test_no_report_flag_disables_reporting(self, tmp_path, monkeypatch, capsys):
        """main --no-report suppresses the console table output."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": [{"model_short_name": "m1"}]}))

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
                "--no-report",
            ],
        )

        with patch.object(ModelConverter, "process_config_file", return_value=(1, 0)):
            assert main() == 0

        assert "Conversion Summary Report" not in capsys.readouterr().out

    def test_report_custom_path(self, tmp_path, monkeypatch, capsys):
        """main --report PATH passes the custom report path to ModelConverter."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": [{"model_short_name": "m1"}]}))
        report_path = tmp_path / "custom" / "my_report.md"

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
                "--report",
                str(report_path),
            ],
        )

        captured: dict = {}
        original_init = ModelConverter.__init__

        def spy_init(self, *args, **kwargs):
            captured.update(kwargs)
            return original_init(self, *args, **kwargs)

        with (
            patch.object(ModelConverter, "__init__", autospec=True, side_effect=spy_init),
            patch.object(ModelConverter, "process_config_file", return_value=(0, 0)),
            patch.object(ModelConverter, "collect_results", return_value=[]),
        ):
            assert main() == 0

        assert captured.get("report_path") == report_path

    def test_measure_accuracy_default_true(self, tmp_path, monkeypatch):
        """main passes measure_accuracy=True to ModelConverter by default."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": []}))

        monkeypatch.setattr(
            sys,
            "argv",
            ["model_converter", str(config_path), "-o", str(tmp_path / "output"), "-c", str(tmp_path / "cache")],
        )

        captured: dict = {}
        original_init = ModelConverter.__init__

        def spy_init(self, *args, **kwargs):
            captured.update(kwargs)
            return original_init(self, *args, **kwargs)

        with (
            patch.object(ModelConverter, "__init__", autospec=True, side_effect=spy_init),
            patch.object(ModelConverter, "process_config_file", return_value=(0, 0)),
            patch.object(ModelConverter, "collect_results", return_value=[]),
        ):
            assert main() == 0

        assert captured.get("measure_accuracy") is True

    def test_no_measure_accuracy_flag(self, tmp_path, monkeypatch):
        """main --no-measure-accuracy passes measure_accuracy=False."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": []}))

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
                "--no-measure-accuracy",
            ],
        )

        captured: dict = {}
        original_init = ModelConverter.__init__

        def spy_init(self, *args, **kwargs):
            captured.update(kwargs)
            return original_init(self, *args, **kwargs)

        with (
            patch.object(ModelConverter, "__init__", autospec=True, side_effect=spy_init),
            patch.object(ModelConverter, "process_config_file", return_value=(0, 0)),
            patch.object(ModelConverter, "collect_results", return_value=[]),
        ):
            assert main() == 0

        assert captured.get("measure_accuracy") is False

    def test_failed_run(self, tmp_path, monkeypatch):
        """main returns 1 when models fail."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": [{"model_short_name": "m1"}]}))

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
            assert main() == 1

    def test_verbose_flag(self, tmp_path, monkeypatch):
        """main enables verbose logging with -v."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": []}))

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
            assert main() == 0

    def test_model_filter(self, tmp_path, monkeypatch):
        """main passes --model filter through to process_config_file."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": [{"model_short_name": "target"}]}))

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

        mock_process.assert_called_once_with(config_path=config_path, model_filter="target", library_filter=None)

    def test_exception_during_processing(self, tmp_path, monkeypatch):
        """main returns 1 on processing exceptions."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": [{"model_short_name": "m1"}]}))

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
            assert main() == 1


class TestLibraryFilter:
    """Tests for --library filter functionality."""

    def test_single_library_filter(self, facade_converter, tmp_path):
        """process_config_file filters models by a single library."""
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "models": [
                        {"model_short_name": "m1", "model_library": "torchvision"},
                        {"model_short_name": "m2", "model_library": "getitune"},
                        {"model_short_name": "m3", "model_library": "timm"},
                    ],
                },
            ),
        )

        with patch.object(facade_converter, "process_model_config", return_value=True) as mock_process:
            successful, failed = facade_converter.process_config_file(
                config_path,
                library_filter=["getitune"],
            )

        assert successful == 1
        assert failed == 0
        mock_process.assert_called_once_with({"model_short_name": "m2", "model_library": "getitune"})

    def test_multiple_library_filter(self, facade_converter, tmp_path):
        """process_config_file filters models by multiple libraries."""
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "models": [
                        {"model_short_name": "m1", "model_library": "torchvision"},
                        {"model_short_name": "m2", "model_library": "getitune"},
                        {"model_short_name": "m3", "model_library": "timm"},
                        {"model_short_name": "m4", "model_library": "yolo"},
                    ],
                },
            ),
        )

        with patch.object(facade_converter, "process_model_config", return_value=True) as mock_process:
            successful, failed = facade_converter.process_config_file(
                config_path,
                library_filter=["getitune", "timm"],
            )

        assert successful == 2
        assert failed == 0
        assert mock_process.call_count == 2

    def test_library_filter_no_match(self, facade_converter, tmp_path):
        """process_config_file returns 0,0 when no models match library filter."""
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "models": [
                        {"model_short_name": "m1", "model_library": "torchvision"},
                    ],
                },
            ),
        )

        successful, failed = facade_converter.process_config_file(config_path, library_filter=["yolo"])
        assert successful == 0
        assert failed == 0

    def test_library_and_model_filter_combined(self, facade_converter, tmp_path):
        """process_config_file applies both library and model filters (AND logic)."""
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "models": [
                        {"model_short_name": "m1", "model_library": "getitune"},
                        {"model_short_name": "m2", "model_library": "getitune"},
                        {"model_short_name": "m1", "model_library": "torchvision"},
                    ],
                },
            ),
        )

        with patch.object(facade_converter, "process_model_config", return_value=True) as mock_process:
            successful, failed = facade_converter.process_config_file(
                config_path,
                model_filter="m1",
                library_filter=["getitune"],
            )

        assert successful == 1
        assert failed == 0
        mock_process.assert_called_once_with({"model_short_name": "m1", "model_library": "getitune"})

    def test_no_library_filter_processes_all(self, facade_converter, tmp_path):
        """process_config_file processes all models when no library filter is given."""
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "models": [
                        {"model_short_name": "m1", "model_library": "torchvision"},
                        {"model_short_name": "m2", "model_library": "getitune"},
                    ],
                },
            ),
        )

        with patch.object(facade_converter, "process_model_config", return_value=True) as mock_process:
            successful, failed = facade_converter.process_config_file(config_path, library_filter=None)

        assert successful == 2
        assert failed == 0
        assert mock_process.call_count == 2

    def test_list_models_with_library_filter(self, tmp_path, capsys):
        """list_models filters output by library."""
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "model_short_name": "resnet50",
                            "model_full_name": "ResNet-50",
                            "model_library": "torchvision",
                            "model_type": "Classification",
                        },
                        {
                            "model_short_name": "dino_v2",
                            "model_full_name": "DINOv2",
                            "model_library": "getitune",
                            "model_type": "Classification",
                        },
                    ],
                },
            ),
        )

        list_models(config_path, library_filter=["getitune"])

        captured = capsys.readouterr()
        assert "dino_v2" in captured.out
        assert "resnet50" not in captured.out

    def test_list_models_no_library_filter(self, tmp_path, capsys):
        """list_models shows all models when no library filter."""
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "model_short_name": "resnet50",
                            "model_full_name": "ResNet-50",
                            "model_library": "torchvision",
                            "model_type": "Classification",
                        },
                        {
                            "model_short_name": "dino_v2",
                            "model_full_name": "DINOv2",
                            "model_library": "getitune",
                            "model_type": "Classification",
                        },
                    ],
                },
            ),
        )

        list_models(config_path, library_filter=None)

        captured = capsys.readouterr()
        assert "dino_v2" in captured.out
        assert "resnet50" in captured.out

    def test_unknown_library_warns(self, facade_converter, tmp_path, caplog):
        """process_config_file warns on unknown library names."""
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "models": [
                        {"model_short_name": "m1", "model_library": "torchvision"},
                    ],
                },
            ),
        )

        with patch.object(facade_converter, "process_model_config", return_value=True):
            facade_converter.process_config_file(config_path, library_filter=["torchvision", "nonexistent"])

        assert "Unknown library 'nonexistent'" in caplog.text

    def test_list_models_unknown_library_warns(self, tmp_path, capsys):
        """list_models prints warning for unknown library names."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": [{"model_short_name": "m1", "model_library": "torchvision"}]}))

        list_models(config_path, library_filter=["nonexistent"])

        captured = capsys.readouterr()
        assert "Unknown library 'nonexistent'" in captured.err
        assert "No models found" in captured.out

    def test_main_library_flag(self, tmp_path, monkeypatch):
        """main passes --library filter through to process_config_file."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": [{"model_short_name": "m1", "model_library": "getitune"}]}))

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
                "--library",
                "getitune,timm",
            ],
        )

        with patch.object(ModelConverter, "process_config_file", return_value=(1, 0)) as mock_process:
            main()

        mock_process.assert_called_once_with(
            config_path=config_path,
            model_filter=None,
            library_filter=["getitune", "timm"],
        )

    def test_main_library_and_model_flags(self, tmp_path, monkeypatch):
        """main passes both --library and --model filters through."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models": [{"model_short_name": "m1", "model_library": "getitune"}]}))

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
                "--library",
                "getitune",
                "--model",
                "m1",
            ],
        )

        with patch.object(ModelConverter, "process_config_file", return_value=(1, 0)) as mock_process:
            main()

        mock_process.assert_called_once_with(
            config_path=config_path,
            model_filter="m1",
            library_filter=["getitune"],
        )

    def test_main_list_with_library_filter(self, tmp_path, monkeypatch, capsys):
        """main --list with --library filters listed models."""
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "models": [
                        {"model_short_name": "m1", "model_library": "torchvision", "model_type": "cls"},
                        {"model_short_name": "m2", "model_library": "getitune", "model_type": "cls"},
                    ],
                },
            ),
        )

        monkeypatch.setattr(sys, "argv", ["model_converter", str(config_path), "--list", "--library", "getitune"])
        assert main() == 0

        captured = capsys.readouterr()
        assert "m2" in captured.out
        assert "m1" not in captured.out
