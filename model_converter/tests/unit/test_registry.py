#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for converter registry and individual converter classes."""

import sys
import types
from unittest.mock import MagicMock, call, patch

import pytest
import torch.nn as nn
from model_converter.converters import (
    CONVERTER_REGISTRY,
    BaseConverter,
    GetituneConverter,
    PyTorchConverter,
    TimmConverter,
    TorchvisionConverter,
    YoloConverter,
)


class TestConverterRegistry:
    """Tests for the converter registry."""

    def test_contains_expected_keys(self):
        """CONVERTER_REGISTRY exposes all supported converter names."""
        assert set(CONVERTER_REGISTRY) == {"torchvision", "timm", "yolo", "getitune"}

    def test_maps_expected_classes(self):
        """CONVERTER_REGISTRY maps each name to the expected class."""
        assert {
            "torchvision": TorchvisionConverter,
            "timm": TimmConverter,
            "yolo": YoloConverter,
            "getitune": GetituneConverter,
        } == CONVERTER_REGISTRY

    @pytest.mark.parametrize(
        ("name", "expected_cls"),
        [
            ("BaseConverter", BaseConverter),
            ("PyTorchConverter", PyTorchConverter),
            ("TorchvisionConverter", TorchvisionConverter),
            ("TimmConverter", TimmConverter),
            ("YoloConverter", YoloConverter),
            ("GetituneConverter", GetituneConverter),
        ],
    )
    def test_package_exports_converter_classes(self, name, expected_cls):
        """model_converter.converters re-exports the converter classes."""
        import model_converter.converters as converters

        assert getattr(converters, name) is expected_cls


class TestTorchvisionConverter:
    """Tests for TorchvisionConverter-specific behavior."""

    def test_process_model_config_downloads_weights_and_loads_model(
        self,
        converter,
        sample_model_config,
        tmp_path,
        mock_torch_model,
    ):
        """process_model_config downloads weights and builds a model instance."""
        weights_path = tmp_path / "weights.pth"
        checkpoint = {"state_dict": {"layer.weight": "weights"}}
        model_class = MagicMock()
        fp16_path = converter.output_dir / "test_model-fp16-ov" / "test_model.xml"
        fp32_path = converter.output_dir / "test_model-fp16-ov" / "test_model_fp32.xml"

        with (
            patch.object(converter._url_downloader, "download", return_value=weights_path) as mock_download,
            patch.object(converter, "load_model_class", return_value=model_class) as mock_load_class,
            patch.object(converter, "load_checkpoint", return_value=checkpoint) as mock_load_checkpoint,
            patch.object(converter, "create_model", return_value=mock_torch_model) as mock_create_model,
            patch.object(converter, "_build_metadata", return_value={("model_info", "model_type"): "Classification"}),
            patch.object(converter, "export_to_openvino", return_value=(fp16_path, fp32_path)) as mock_export,
        ):
            result = converter.process_model_config(sample_model_config)

        assert result is True
        mock_download.assert_called_once_with(url=sample_model_config["weights_url"])
        mock_load_class.assert_called_once_with(sample_model_config["model_class_name"])
        mock_load_checkpoint.assert_called_once_with(weights_path)
        mock_create_model.assert_called_once_with(model_class, checkpoint, None)
        assert mock_export.call_args.kwargs["model"] is mock_torch_model
        assert mock_export.call_args.kwargs["output_path"] == converter.output_dir / "test_model"

    def test_process_model_config_skips_when_outputs_are_cached(self, converter, sample_model_config):
        """process_model_config skips work when both output models already exist."""
        fp16_dir = converter.output_dir / "test_model-fp16-ov"
        int8_dir = converter.output_dir / "test_model-int8-ov"
        fp16_dir.mkdir(parents=True)
        int8_dir.mkdir(parents=True)
        (fp16_dir / "test_model.xml").write_text("<net/>")
        (int8_dir / "test_model.xml").write_text("<net/>")

        with patch.object(converter._url_downloader, "download") as mock_download:
            result = converter.process_model_config(sample_model_config)

        assert result is True
        mock_download.assert_not_called()


class TestTimmConverter:
    """Tests for TimmConverter-specific behavior."""

    def test_load_huggingface_model_uses_timm_create_model(self, timm_converter):
        """load_huggingface_model builds the expected timm hub reference."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        timm_module = types.ModuleType("timm")
        timm_module.create_model = MagicMock(return_value=mock_model)

        with patch.dict(sys.modules, {"timm": timm_module}):
            result = timm_converter.load_huggingface_model(
                repo_id="timm/resnet50.a1_in1k",
                revision="abc123",
                model_library="timm",
                model_params={"num_classes": 10},
            )

        assert result is mock_model
        timm_module.create_model.assert_called_once_with(
            "hf-hub:timm/resnet50.a1_in1k@abc123",
            pretrained=True,
            cache_dir=timm_converter.cache_dir,
            num_classes=10,
        )
        mock_model.eval.assert_called_once_with()

    def test_process_model_config_uses_loader_and_export_pipeline(
        self,
        timm_converter,
        sample_timm_config,
        mock_torch_model,
    ):
        """process_model_config runs the timm conversion workflow."""
        fp16_path = timm_converter.output_dir / "test_timm_model-fp16-ov" / "test_timm_model.xml"
        fp32_path = timm_converter.output_dir / "test_timm_model-fp16-ov" / "test_timm_model_fp32.xml"

        with (
            patch.object(timm_converter, "load_huggingface_model", return_value=mock_torch_model) as mock_load,
            patch.object(
                timm_converter,
                "_build_metadata",
                return_value={("model_info", "model_type"): "Classification"},
            ) as mock_metadata,
            patch.object(timm_converter, "export_to_openvino", return_value=(fp16_path, fp32_path)) as mock_export,
        ):
            result = timm_converter.process_model_config(sample_timm_config)

        assert result is True
        mock_load.assert_called_once_with(
            repo_id=sample_timm_config["huggingface_repo"],
            revision=sample_timm_config["huggingface_revision"],
            model_library="timm",
            model_params=None,
        )
        mock_metadata.assert_called_once_with(sample_timm_config)
        assert mock_export.call_args.kwargs["model"] is mock_torch_model
        assert mock_export.call_args.kwargs["output_path"] == timm_converter.output_dir / "test_timm_model"


class TestYoloConverter:
    """Tests for YoloConverter-specific behavior."""

    def test_process_model_config_exports_fp16_and_int8(self, tmp_path, monkeypatch):
        """process_model_config exports both YOLO variants and repackages them."""
        converter = YoloConverter(output_dir=tmp_path / "output", cache_dir=tmp_path / "cache")
        config = {"model_short_name": "YOLO11n", "yolo_version": "yolo11n"}
        mock_model = MagicMock()
        yolo_module = types.ModuleType("ultralytics")
        yolo_module.YOLO = MagicMock(return_value=mock_model)
        export_calls: list[dict[str, object]] = []

        def export_side_effect(**kwargs):
            export_calls.append(kwargs)
            suffix = "_int8_openvino_model" if kwargs.get("int8") else "_openvino_model"
            export_dir = converter.cache_dir / f"yolo11n{suffix}"
            export_dir.mkdir(parents=True)
            (export_dir / "yolo11n.xml").write_text("<net/>")

        mock_model.export.side_effect = export_side_effect
        monkeypatch.chdir(tmp_path)

        with (
            patch.dict(sys.modules, {"ultralytics": yolo_module}),
            patch.object(converter, "_update_model_type_in_xml") as mock_update,
            patch.object(converter, "_copy_yolo_readme") as mock_copy_readme,
        ):
            result = converter.process_model_config(config)

        assert result is True
        yolo_module.YOLO.assert_called_once_with(str(converter.cache_dir / "yolo11n.pt"))
        assert export_calls == [
            {"format": "openvino", "half": True},
            {"format": "openvino", "int8": True, "data": "coco128.yaml"},
        ]
        assert (converter.output_dir / "YOLO11n-fp16-ov" / "yolo11n.xml").exists()
        assert (converter.output_dir / "YOLO11n-int8-ov" / "yolo11n.xml").exists()
        assert mock_update.call_count == 2
        mock_copy_readme.assert_has_calls(
            [
                call("README-yolo-fp16.md", converter.output_dir / "YOLO11n-fp16-ov", "n"),
                call("README-yolo-int8.md", converter.output_dir / "YOLO11n-int8-ov", "n"),
            ],
        )

    def test_process_model_config_returns_false_on_export_failure(self, tmp_path, monkeypatch):
        """process_model_config handles YOLO export errors gracefully."""
        converter = YoloConverter(output_dir=tmp_path / "output", cache_dir=tmp_path / "cache")
        yolo_module = types.ModuleType("ultralytics")
        failing_model = MagicMock()
        failing_model.export.side_effect = RuntimeError("export failed")
        yolo_module.YOLO = MagicMock(return_value=failing_model)
        monkeypatch.chdir(tmp_path)

        with patch.dict(sys.modules, {"ultralytics": yolo_module}):
            result = converter.process_model_config({"model_short_name": "YOLO11n", "yolo_version": "yolo11n"})

        assert result is False


class TestPyTorchConverterSharedLogic:
    """Tests for logic shared by PyTorch-based converters."""

    def test_export_to_openvino_converts_and_saves_models(
        self,
        converter,
        sample_model_config,
        mock_torch_model,
        mock_ov_model,
    ):
        """export_to_openvino converts the model and saves FP32 and FP16 artifacts."""
        dummy_input = object()
        openvino_module = types.ModuleType("openvino")
        openvino_module.convert_model = MagicMock(return_value=mock_ov_model)
        openvino_module.save_model = MagicMock()

        with (
            patch.dict(sys.modules, {"openvino": openvino_module}),
            patch.object(converter, "_prepare_model_for_export", return_value=mock_torch_model) as mock_prepare,
            patch.object(converter, "_create_example_input", return_value=dummy_input) as mock_example_input,
            patch.object(converter, "_postprocess_openvino_model", return_value=mock_ov_model) as mock_postprocess,
            patch("model_converter.converters.pytorch.shutil.copy2") as mock_copy,
            patch.object(converter, "copy_readme") as mock_copy_readme,
        ):
            fp16_path, fp32_path = converter.export_to_openvino(
                model=mock_torch_model,
                input_shape=[1, 3, 224, 224],
                output_path=converter.output_dir / "test_model",
                model_config=sample_model_config,
                input_names=["input"],
                output_names=["result"],
                metadata={("model_info", "model_type"): "Classification"},
            )

        assert fp16_path == converter.output_dir / "test_model-fp16-ov" / "test_model.xml"
        assert fp32_path == converter.output_dir / "test_model-fp16-ov" / "test_model_fp32.xml"
        mock_prepare.assert_called_once_with(mock_torch_model, sample_model_config)
        mock_example_input.assert_called_once_with([1, 3, 224, 224], sample_model_config)
        openvino_module.convert_model.assert_called_once_with(mock_torch_model, example_input=dummy_input)
        mock_ov_model.reshape.assert_called_once_with({"input": [1, 3, 224, 224]})
        mock_postprocess.assert_called_once_with(
            mock_ov_model,
            input_names=["input"],
            output_names=["result"],
            metadata={("model_info", "model_type"): "Classification"},
        )
        openvino_module.save_model.assert_has_calls(
            [
                call(mock_ov_model, fp32_path, compress_to_fp16=False),
                call(mock_ov_model, fp16_path, compress_to_fp16=True),
            ],
        )
        mock_copy.assert_called_once()
        mock_copy_readme.assert_called_once_with(sample_model_config, fp16_path.parent, variant="fp16")
        assert (fp16_path.parent / "config.json").exists()

    def test_build_metadata_includes_core_fields_and_labels(self, converter, sample_model_config):
        """_build_metadata adds required metadata fields and resolved labels."""
        with patch.object(converter, "get_labels", return_value="tabby_cat golden_retriever"):
            metadata = converter._build_metadata(
                {
                    **sample_model_config,
                    "confidence_threshold": 0.5,
                    "iou_threshold": 0.45,
                },
            )

        assert metadata["model_info", "model_type"] == "Classification"
        assert metadata["model_info", "model_short_name"] == "test_model"
        assert metadata["model_info", "reverse_input_channels"] == "True"
        assert metadata["model_info", "mean_values"] == "123.675 116.28 103.53"
        assert metadata["model_info", "scale_values"] == "58.395 57.12 57.375"
        assert metadata["model_info", "confidence_threshold"] == "0.5"
        assert metadata["model_info", "iou_threshold"] == "0.45"
        assert metadata["model_info", "labels"] == "tabby_cat golden_retriever"

    def test_get_labels_resolves_imagenet1k(self, converter):
        """get_labels resolves torchvision ImageNet-1K labels."""
        torchvision_module = types.ModuleType("torchvision")
        torchvision_module.__path__ = []
        torchvision_models_module = types.ModuleType("torchvision.models")
        torchvision_models_module.__path__ = []
        torchvision_meta_module = types.ModuleType("torchvision.models._meta")
        torchvision_meta_module._IMAGENET_CATEGORIES = ["tabby cat", "golden retriever"]

        with patch.dict(
            sys.modules,
            {
                "torchvision": torchvision_module,
                "torchvision.models": torchvision_models_module,
                "torchvision.models._meta": torchvision_meta_module,
            },
        ):
            assert converter.get_labels("IMAGENET1K_V1") == "tabby_cat golden_retriever"

    def test_get_labels_resolves_imagenet21k(self, converter):
        """get_labels resolves timm ImageNet-21K labels."""
        timm_module = types.ModuleType("timm")
        timm_module.__path__ = []
        timm_data_module = types.ModuleType("timm.data")
        image_net_info = MagicMock()
        image_net_info.label_descriptions.return_value = ["tabby, tabby cat", "golden retriever, dog"]
        timm_data_module.ImageNetInfo = MagicMock(return_value=image_net_info)

        with patch.dict(sys.modules, {"timm": timm_module, "timm.data": timm_data_module}):
            assert converter.get_labels("IMAGENET21K") == "tabby golden_retriever"

    def test_get_labels_resolves_coco(self, converter):
        """get_labels resolves torchvision COCO labels."""
        torchvision_module = types.ModuleType("torchvision")
        torchvision_module.__path__ = []
        torchvision_models_module = types.ModuleType("torchvision.models")
        torchvision_models_module.__path__ = []
        torchvision_detection_module = types.ModuleType("torchvision.models.detection")
        mock_weights = MagicMock()
        mock_weights.COCO_V1.meta = {"categories": ["person", "traffic light"]}
        torchvision_detection_module.MaskRCNN_ResNet50_FPN_Weights = mock_weights

        with patch.dict(
            sys.modules,
            {
                "torchvision": torchvision_module,
                "torchvision.models": torchvision_models_module,
                "torchvision.models.detection": torchvision_detection_module,
            },
        ):
            assert converter.get_labels("COCO_V1") == "person traffic_light"

    def test_create_model_loads_state_dict_into_instance(self, converter):
        """create_model instantiates a class and loads its state_dict."""
        model_class = MagicMock()
        model_instance = MagicMock(spec=nn.Module)
        model_instance.eval.return_value = model_instance
        model_class.return_value = model_instance
        checkpoint = {"state_dict": {"layer.weight": "weights"}}

        result = converter.create_model(model_class, checkpoint, model_params={"num_classes": 2})

        assert result is model_instance
        model_class.assert_called_once_with(num_classes=2)
        model_instance.load_state_dict.assert_called_once_with(checkpoint["state_dict"], strict=False)
        model_instance.eval.assert_called_once_with()
