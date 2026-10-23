#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for GetituneConverter."""

import contextlib
import json
import logging
import types
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
from model_converter.converters.getitune import GetituneConverter
from model_converter.reporting import AccuracyResults


def _write_openvino_model(xml_path: Path) -> None:
    """Create paired OpenVINO XML and BIN files."""
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text("<net/>")
    xml_path.with_suffix(".bin").write_bytes(b"bin")


def _make_openvino_module(model_info: dict[str, str] | None = None) -> types.ModuleType:
    """Create a fake openvino module for metadata extraction."""
    ov_module = types.ModuleType("openvino")
    core = MagicMock()
    model = MagicMock()
    model.get_rt_info.return_value = MagicMock(value=model_info or {"model_type": "Classification"})
    core.read_model.return_value = model
    setattr(ov_module, "Core", MagicMock(return_value=core))
    return ov_module


def _make_partial_shape(dims: list[int | None]) -> MagicMock:
    """Build a fake OpenVINO partial shape.

    Each ``int`` entry is a static dimension (``is_static`` True, ``get_length``
    returns it); ``None`` models a dynamic dimension (e.g. a dynamic batch).
    """

    def _make_dim(value: int | None) -> MagicMock:
        dim = MagicMock()
        dim.is_static = value is not None
        dim.get_length.return_value = value
        return dim

    partial_shape = MagicMock()
    partial_shape.__iter__.return_value = iter([_make_dim(v) for v in dims])
    return partial_shape


@pytest.fixture
def sample_getitune_config():
    """Sample getitune model configuration."""
    return {
        "model_short_name": "dino_v2_cls",
        "model_full_name": "DINOv2 Classification",
        "model_library": "getitune",
        "getitune_task": "MULTI_CLASS_CLS",
        "getitune_recipe": "dino_v2",
        "model_type": "Classification",
        "license": "apache-2.0",
        "license_link": "https://www.apache.org/licenses/LICENSE-2.0",
        "docs": "https://example.com",
        "tags": ["classification", "openvino"],
        "quantize": True,
        "dataset_type": "imagenet-1k",
    }


@pytest.fixture
def getitune_converter(tmp_output_dir, tmp_cache_dir, mock_dataset_registry):
    """GetituneConverter with temporary directories."""
    return GetituneConverter(
        output_dir=tmp_output_dir,
        cache_dir=tmp_cache_dir,
        verbose=True,
        dataset_registry=mock_dataset_registry,
    )


class TestPresetConfig:
    """Tests for getitune preset configuration."""

    @staticmethod
    def _load_config() -> dict:
        config_path = Path(__file__).resolve().parents[2] / "presets" / "config.json"
        return json.loads(config_path.read_text())

    def test_ssd_mobilenet_v2_has_export_matching_preprocessing_overrides(self):
        """SSD MobileNet-V2 keeps Model API metadata aligned with getitune export settings."""
        config = self._load_config()

        model_config = next(model for model in config["models"] if model["model_short_name"] == "ssd_mobilenet_v2")

        assert model_config["labels"] == "COCO_80"
        assert model_config["preprocessing_overrides"] == {
            "confidence_threshold": "0.02",
            "resize_type": "standard",
            "reverse_input_channels": "False",
            "mean_values": "0.0 0.0 0.0",
            "scale_values": "1.0 1.0 1.0",
        }

    def test_rfdetr_segmentation_configs_use_detr_instance_segmentation_wrapper(self):
        """RF-DETR segmentation presets must select full-image mask postprocessing."""
        config = self._load_config()
        expected_names = {
            "rfdetr_seg_large",
            "rfdetr_seg_medium",
            "rfdetr_seg_small",
            "rfdetr_seg_xlarge",
        }

        model_configs = {
            model["model_short_name"]: model
            for model in config["models"]
            if model.get("model_short_name") in expected_names
        }

        assert set(model_configs) == expected_names
        for model_config in model_configs.values():
            assert model_config["model_type"] == "DETRInstSeg"
            assert model_config["model_info_overrides"] == {"model_type": "DETRInstSeg"}


class TestProcessModelConfig:
    """Tests for GetituneConverter.process_model_config."""

    def test_process_model_config_succeeds(
        self,
        getitune_converter,
        sample_getitune_config,
        tmp_path,
    ):
        """process_model_config exports, repackages, and quantizes a valid model."""
        export_root = tmp_path / "getitune_export_process"
        expected_xml = export_root / "multi_class_cls" / "dino_v2" / "exported_model.xml"

        _write_openvino_model(expected_xml)
        fake_openvino = _make_openvino_module()

        with (
            patch("model_converter.converters.getitune.tempfile.mkdtemp", return_value=str(export_root)),
            patch.object(getitune_converter, "_run_export", return_value=expected_xml) as mock_run_export,
            patch.object(getitune_converter, "copy_readme") as mock_copy_readme,
            patch.object(
                getitune_converter,
                "_quantize_exported_model",
                return_value=AccuracyResults(),
            ) as mock_quantize,
            patch.dict("sys.modules", {"openvino": fake_openvino}),
        ):
            assert getitune_converter.process_model_config(sample_getitune_config) is True

        output_folder = getitune_converter.output_dir / "dino_v2_cls-fp16-ov"
        assert (output_folder / "dino_v2_cls.xml").exists()
        assert (output_folder / "dino_v2_cls.bin").exists()
        assert json.loads((output_folder / "config.json").read_text()) == {"model_type": "Classification"}
        mock_run_export.assert_called_once_with(sample_getitune_config)
        mock_copy_readme.assert_called_once_with(sample_getitune_config, output_folder, variant="fp16")
        mock_quantize.assert_called_once_with(sample_getitune_config)
        assert not export_root.exists()

    def test_process_model_config_skips_when_outputs_exist(self, getitune_converter, sample_getitune_config):
        """process_model_config skips work when FP16 and INT8 models already exist."""
        model_short_name = sample_getitune_config["model_short_name"]
        fp16_model = getitune_converter.output_dir / f"{model_short_name}-fp16-ov" / f"{model_short_name}.xml"
        int8_model = getitune_converter.output_dir / f"{model_short_name}-int8-ov" / f"{model_short_name}.xml"
        fp16_model.parent.mkdir(parents=True)
        int8_model.parent.mkdir(parents=True)
        fp16_model.write_text("<net/>")
        int8_model.write_text("<net/>")

        with patch.object(getitune_converter, "_run_export") as mock_run_export:
            assert getitune_converter.process_model_config(sample_getitune_config) is True

        mock_run_export.assert_not_called()

    def test_process_model_config_returns_false_when_export_fails(
        self,
        getitune_converter,
        sample_getitune_config,
        caplog,
    ):
        """process_model_config logs and returns False on export failures."""
        with (
            patch.object(getitune_converter, "_run_export", side_effect=RuntimeError("export failed")),
            patch.object(getitune_converter, "_repackage_model") as mock_repackage,
            caplog.at_level(logging.ERROR),
        ):
            result = getitune_converter.process_model_config(sample_getitune_config)

        assert result is False
        mock_repackage.assert_not_called()
        assert "export failed" in caplog.text

    def test_process_model_config_skips_quantization_when_disabled(
        self,
        getitune_converter,
        sample_getitune_config,
        tmp_path,
    ):
        """process_model_config does not quantize when quantize is disabled."""
        config = sample_getitune_config | {"quantize": False}
        exported_model_path = tmp_path / "exported_model.xml"

        with (
            patch.object(getitune_converter, "_run_export", return_value=exported_model_path),
            patch.object(getitune_converter, "_repackage_model") as mock_repackage,
            patch.object(getitune_converter, "_quantize_exported_model") as mock_quantize,
        ):
            assert getitune_converter.process_model_config(config) is True

        mock_repackage.assert_called_once_with(config, exported_model_path)
        mock_quantize.assert_not_called()

    def test_process_model_config_returns_false_when_license_is_missing(
        self,
        getitune_converter,
        sample_getitune_config,
        caplog,
    ):
        """process_model_config returns False when license is not defined."""
        config = dict(sample_getitune_config)
        config.pop("license")

        with caplog.at_level(logging.ERROR):
            result = getitune_converter.process_model_config(config)

        assert result is False
        assert "must define 'license'" in caplog.text

    def test_process_model_config_returns_false_when_license_link_is_missing(
        self,
        getitune_converter,
        sample_getitune_config,
        caplog,
    ):
        """process_model_config returns False when license_link is not defined."""
        config = dict(sample_getitune_config)
        config.pop("license_link")

        with caplog.at_level(logging.ERROR):
            result = getitune_converter.process_model_config(config)

        assert result is False
        assert "must define 'license_link'" in caplog.text

    def test_process_model_config_logs_description_when_present(
        self,
        getitune_converter,
        sample_getitune_config,
        tmp_path,
        caplog,
    ):
        """process_model_config logs model description when config includes it."""
        config = sample_getitune_config | {"description": "A test getitune model"}
        export_root = tmp_path / "getitune_export_desc"
        expected_xml = export_root / "multi_class_cls" / "dino_v2" / "exported_model.xml"

        expected_xml.parent.mkdir(parents=True, exist_ok=True)
        expected_xml.write_text("<net/>")
        expected_xml.with_suffix(".bin").write_bytes(b"bin")

        fake_openvino = _make_openvino_module()

        with (
            patch("model_converter.converters.getitune.tempfile.mkdtemp", return_value=str(export_root)),
            patch.object(getitune_converter, "_run_export", return_value=expected_xml),
            patch.object(getitune_converter, "copy_readme"),
            patch.object(getitune_converter, "_quantize_exported_model", return_value=AccuracyResults()),
            patch.dict("sys.modules", {"openvino": fake_openvino}),
            caplog.at_level(logging.INFO),
        ):
            result = getitune_converter.process_model_config(config)

        assert result is True
        assert "A test getitune model" in caplog.text


class TestRunExport:
    """Tests for GetituneConverter._run_export."""

    MODULE = "model_converter.converters.getitune"

    @staticmethod
    @contextlib.contextmanager
    def _patch_export_pipeline(
        tmp_path,
        *,
        recipe_paths=("/recipes/dino_v2.yaml",),
        resolved_config=None,
        label_info=1000,
        input_size=(224, 224),
        recover=("recovered", "loaded pre-trained head"),
    ):
        """Patch all static helpers used by ``_run_export`` and yield the mocks.

        Every collaborating static method is replaced so the test exercises only
        the orchestration logic inside ``_run_export``.
        """
        exported = tmp_path / "exported_model.xml"
        model = MagicMock(name="model")
        module = TestRunExport.MODULE
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch(f"{module}.tempfile.mkdtemp", return_value=str(tmp_path)))
            stack.enter_context(
                patch.object(GetituneConverter, "_get_recipe_paths", return_value=list(recipe_paths)),
            )
            stack.enter_context(
                patch.object(
                    GetituneConverter,
                    "_load_recipe_config",
                    return_value=resolved_config if resolved_config is not None else {"resolved": True},
                ),
            )
            stack.enter_context(
                patch.object(GetituneConverter, "_get_label_info_from_config", return_value=label_info),
            )
            stack.enter_context(
                patch.object(GetituneConverter, "_get_input_size_from_config", return_value=input_size),
            )
            mocks = types.SimpleNamespace(
                model=model,
                exported=exported,
                create=stack.enter_context(
                    patch.object(GetituneConverter, "_create_model_from_config", return_value=model),
                ),
                recover=stack.enter_context(
                    patch.object(GetituneConverter, "_recover_pretrained_head", return_value=recover),
                ),
                export=stack.enter_context(
                    patch.object(GetituneConverter, "_export_model", return_value=exported),
                ),
            )
            yield mocks

    def test_run_export_raises_when_task_is_missing(self, getitune_converter, sample_getitune_config):
        """_run_export rejects a config without getitune_task."""
        config = dict(sample_getitune_config)
        config.pop("getitune_task")

        with pytest.raises(ValueError, match="must define 'getitune_task'"):
            getitune_converter._run_export(config)

    def test_run_export_raises_when_recipe_is_missing(self, getitune_converter, sample_getitune_config):
        """_run_export rejects a config without getitune_recipe."""
        config = dict(sample_getitune_config)
        config.pop("getitune_recipe")

        with pytest.raises(ValueError, match="must define 'getitune_recipe'"):
            getitune_converter._run_export(config)

    def test_run_export_raises_when_recipe_not_found(
        self,
        getitune_converter,
        sample_getitune_config,
        tmp_path,
    ):
        """_run_export raises when no recipe path matches the configured recipe."""
        with (
            patch(f"{self.MODULE}.tempfile.mkdtemp", return_value=str(tmp_path)),
            patch.object(GetituneConverter, "_get_recipe_paths", return_value=["/recipes/other_model.yaml"]),
        ):
            with pytest.raises(ValueError, match="Recipe 'dino_v2' not found"):
                getitune_converter._run_export(sample_getitune_config)

    def test_run_export_raises_when_label_info_missing(
        self,
        getitune_converter,
        sample_getitune_config,
        tmp_path,
    ):
        """_run_export raises when the recipe resolves without a label_info (None check)."""
        with self._patch_export_pipeline(tmp_path, label_info=None) as mocks:
            with pytest.raises(ValueError, match="No label_info in recipe"):
                getitune_converter._run_export(sample_getitune_config)

        # The model is never created once label_info resolution fails.
        mocks.create.assert_not_called()
        mocks.export.assert_not_called()

    def test_run_export_recovers_head_for_classification(
        self,
        getitune_converter,
        sample_getitune_config,
        tmp_path,
        caplog,
    ):
        """_run_export restores the pre-trained head for a classification task and exports."""
        with (
            self._patch_export_pipeline(tmp_path) as mocks,
            caplog.at_level(logging.INFO),
        ):
            result = getitune_converter._run_export(sample_getitune_config)

        assert result == mocks.exported
        mocks.create.assert_called_once_with({"resolved": True}, label_info=1000, input_size=(224, 224))
        mocks.recover.assert_called_once_with(mocks.model)
        mocks.export.assert_called_once_with(mocks.model, output_dir=tmp_path, precision="FP16")
        assert "Pre-trained head: loaded pre-trained head" in caplog.text

    def test_run_export_logs_when_head_not_recovered(
        self,
        getitune_converter,
        sample_getitune_config,
        tmp_path,
        caplog,
    ):
        """_run_export logs the skipped/failed detail when the head cannot be recovered."""
        with (
            self._patch_export_pipeline(tmp_path, recover=("skipped", "no head-recovery handler")) as mocks,
            caplog.at_level(logging.INFO),
        ):
            result = getitune_converter._run_export(sample_getitune_config)

        assert result == mocks.exported
        mocks.recover.assert_called_once_with(mocks.model)
        assert "Pre-trained head NOT recovered (random head): no head-recovery handler" in caplog.text

    def test_run_export_skips_head_recovery_for_non_classification(
        self,
        getitune_converter,
        sample_getitune_config,
        tmp_path,
    ):
        """_run_export does not attempt head recovery for non-classification tasks."""
        config = {**sample_getitune_config, "getitune_task": "DETECTION"}

        with self._patch_export_pipeline(tmp_path) as mocks:
            result = getitune_converter._run_export(config)

        assert result == mocks.exported
        mocks.recover.assert_not_called()
        mocks.export.assert_called_once_with(mocks.model, output_dir=tmp_path, precision="FP16")

    def test_run_export_logs_input_size_when_present(
        self,
        getitune_converter,
        sample_getitune_config,
        tmp_path,
        caplog,
    ):
        """_run_export logs the input_size when the recipe provides one (not-None check)."""
        with (
            self._patch_export_pipeline(tmp_path, input_size=(336, 336)) as mocks,
            caplog.at_level(logging.INFO),
        ):
            getitune_converter._run_export(sample_getitune_config)

        mocks.create.assert_called_once_with({"resolved": True}, label_info=1000, input_size=(336, 336))
        assert "input_size: (336, 336)" in caplog.text

    def test_run_export_without_input_size_does_not_log_it(
        self,
        getitune_converter,
        sample_getitune_config,
        tmp_path,
        caplog,
    ):
        """_run_export omits the input_size log line when the recipe has none (None branch)."""
        with (
            self._patch_export_pipeline(tmp_path, input_size=None) as mocks,
            caplog.at_level(logging.INFO),
        ):
            getitune_converter._run_export(sample_getitune_config)

        mocks.create.assert_called_once_with({"resolved": True}, label_info=1000, input_size=None)
        assert "input_size:" not in caplog.text


class TestRepackageModel:
    """Tests for GetituneConverter._repackage_model."""

    def test_repackage_model_creates_layout_and_cleans_up(
        self,
        getitune_converter,
        sample_getitune_config,
        tmp_path,
    ):
        """_repackage_model copies files, writes metadata, and removes the export directory."""
        export_root = tmp_path / "getitune_export_repackage"
        exported_model_path = export_root / "multi_class_cls" / "dino_v2" / "exported_model.xml"
        _write_openvino_model(exported_model_path)
        fake_openvino = _make_openvino_module({"model_type": "Classification", "labels": "cat dog"})

        with (
            patch.object(getitune_converter, "copy_readme") as mock_copy_readme,
            patch.dict("sys.modules", {"openvino": fake_openvino}),
        ):
            getitune_converter._repackage_model(sample_getitune_config, exported_model_path)

        output_folder = getitune_converter.output_dir / "dino_v2_cls-fp16-ov"
        assert (output_folder / "dino_v2_cls.xml").exists()
        assert (output_folder / "dino_v2_cls.bin").exists()
        assert (output_folder / "dino_v2_cls_fp32.xml").exists()
        assert (output_folder / "dino_v2_cls_fp32.bin").exists()
        assert json.loads((output_folder / "config.json").read_text()) == {
            "model_type": "Classification",
            "labels": "cat dog",
        }
        assert (output_folder / ".gitattributes").exists()
        mock_copy_readme.assert_called_once_with(sample_getitune_config, output_folder, variant="fp16")
        assert not export_root.exists()

    def test_repackage_model_handles_metadata_extraction_failure(
        self,
        getitune_converter,
        sample_getitune_config,
        tmp_path,
        caplog,
    ):
        """_repackage_model continues when openvino metadata extraction fails."""
        export_root = tmp_path / "getitune_export_meta_fail"
        exported_model_path = export_root / "multi_class_cls" / "dino_v2" / "exported_model.xml"
        _write_openvino_model(exported_model_path)

        failing_ov = types.ModuleType("openvino")
        core = MagicMock()
        model = MagicMock()
        model.get_rt_info.side_effect = RuntimeError("No rt_info found")
        core.read_model.return_value = model
        setattr(failing_ov, "Core", MagicMock(return_value=core))

        with (
            patch.object(getitune_converter, "copy_readme"),
            patch.dict("sys.modules", {"openvino": failing_ov}),
            caplog.at_level(logging.WARNING),
        ):
            getitune_converter._repackage_model(sample_getitune_config, exported_model_path)

        output_folder = getitune_converter.output_dir / "dino_v2_cls-fp16-ov"
        assert (output_folder / "dino_v2_cls.xml").exists()
        assert "Could not extract model_info metadata" in caplog.text
        assert not (output_folder / "config.json").exists()

    def test_repackage_model_handles_non_matching_temp_path(
        self,
        getitune_converter,
        sample_getitune_config,
        tmp_path,
    ):
        """_repackage_model handles exported path without getitune_export_ prefix."""
        export_dir = tmp_path / "some_other_dir" / "sub"
        exported_model_path = export_dir / "exported_model.xml"
        _write_openvino_model(exported_model_path)

        fake_openvino = _make_openvino_module()

        with (
            patch.object(getitune_converter, "copy_readme"),
            patch.dict("sys.modules", {"openvino": fake_openvino}),
        ):
            getitune_converter._repackage_model(sample_getitune_config, exported_model_path)

        output_folder = getitune_converter.output_dir / "dino_v2_cls-fp16-ov"
        assert (output_folder / "dino_v2_cls.xml").exists()
        # The temp directory should still exist since it doesn't match the pattern
        assert export_dir.exists()


class TestQuantizeExportedModel:
    """Tests for GetituneConverter._quantize_exported_model."""

    def test_quantize_exported_model_uses_fp32_model_and_cleans_up(
        self,
        getitune_converter,
        sample_getitune_config,
    ):
        """_quantize_exported_model uses the FP32 copy and removes it afterward."""
        output_folder = getitune_converter.output_dir / "dino_v2_cls-fp16-ov"
        fp32_xml = output_folder / "dino_v2_cls_fp32.xml"
        fp32_bin = output_folder / "dino_v2_cls_fp32.bin"
        _write_openvino_model(fp32_xml)
        calibration_data = [np.zeros((1, 3, 224, 224), dtype=np.float32)]

        with (
            patch.object(
                getitune_converter,
                "_read_preprocessing_from_model",
                return_value=([1, 3, 224, 224], "123.675 116.28 103.53", "58.395 57.12 57.375", True, 255.0),
            ) as mock_read_preproc,
            patch.object(
                getitune_converter,
                "create_calibration_dataset",
                return_value=(calibration_data, []),
            ) as mock_dataset,
            patch.object(getitune_converter, "quantize_model") as mock_quantize,
        ):
            getitune_converter._quantize_exported_model(sample_getitune_config)

        mock_read_preproc.assert_called_once()
        mock_dataset.assert_called_once_with(
            input_shape=[1, 3, 224, 224],
            mean_values="123.675 116.28 103.53",
            scale_values="58.395 57.12 57.375",
            reverse_input_channels=True,
            subset_size=500,
            return_labels=True,
            dataset_path=ANY,
            dataset_type="imagenet-1k",
            intensity_scale=255.0,
        )
        mock_quantize.assert_called_once_with(
            model_path=fp32_xml,
            calibration_data=calibration_data,
            model_config=sample_getitune_config,
            preset="mixed",
            validation_data=None,
            validation_labels=None,
            validation_samples=None,
            metric=ANY,
            accuracy_results=ANY,
        )
        assert not fp32_xml.exists()
        assert not fp32_bin.exists()

    def test_quantize_exported_model_falls_back_to_fp16_when_fp32_is_missing(
        self,
        getitune_converter,
        sample_getitune_config,
    ):
        """_quantize_exported_model falls back to the FP16 model when no FP32 copy exists."""
        output_folder = getitune_converter.output_dir / "dino_v2_cls-fp16-ov"
        fp16_xml = output_folder / "dino_v2_cls.xml"
        _write_openvino_model(fp16_xml)
        calibration_data = [np.zeros((1, 3, 224, 224), dtype=np.float32)]

        with (
            patch.object(
                getitune_converter,
                "_read_preprocessing_from_model",
                return_value=([1, 3, 224, 224], "0 0 0", "1 1 1", True, 255.0),
            ),
            patch.object(
                getitune_converter,
                "create_calibration_dataset",
                return_value=(calibration_data, []),
            ),
            patch.object(getitune_converter, "quantize_model") as mock_quantize,
        ):
            getitune_converter._quantize_exported_model(sample_getitune_config)

        mock_quantize.assert_called_once_with(
            model_path=fp16_xml,
            calibration_data=calibration_data,
            model_config=sample_getitune_config,
            preset="mixed",
            validation_data=None,
            validation_labels=None,
            validation_samples=None,
            metric=ANY,
            accuracy_results=ANY,
        )

    def test_quantize_exported_model_handles_cleanup_oserror(
        self,
        getitune_converter,
        sample_getitune_config,
        caplog,
    ):
        """_quantize_exported_model logs a warning when FP32 file cleanup fails."""
        output_folder = getitune_converter.output_dir / "dino_v2_cls-fp16-ov"
        fp32_xml = output_folder / "dino_v2_cls_fp32.xml"
        _write_openvino_model(fp32_xml)
        calibration_data = [np.zeros((1, 3, 224, 224), dtype=np.float32)]

        with (
            patch.object(
                getitune_converter,
                "_read_preprocessing_from_model",
                return_value=([1, 3, 224, 224], "0 0 0", "1 1 1", True, 255.0),
            ),
            patch.object(
                getitune_converter,
                "create_calibration_dataset",
                return_value=(calibration_data, []),
            ),
            patch.object(getitune_converter, "quantize_model"),
            patch("pathlib.Path.unlink", side_effect=OSError("permission denied")),
            caplog.at_level(logging.WARNING),
        ):
            getitune_converter._quantize_exported_model(sample_getitune_config)

        assert "Failed to remove temporary FP32 files" in caplog.text

    def test_quantize_exported_model_measures_accuracy_for_imagenet1k(
        self,
        getitune_converter,
        sample_getitune_config,
    ):
        """Accuracy is measured for MULTI_CLASS_CLS IMAGENET1K_V1 models."""
        config = {**sample_getitune_config, "labels": "IMAGENET1K_V1"}
        output_folder = getitune_converter.output_dir / "dino_v2_cls-fp16-ov"
        fp32_xml = output_folder / "dino_v2_cls_fp32.xml"
        _write_openvino_model(fp32_xml)
        calibration_data = [np.zeros((1, 3, 224, 224), dtype=np.float32)]
        validation_labels = [7]

        with (
            patch.object(
                getitune_converter,
                "_read_preprocessing_from_model",
                return_value=([1, 3, 224, 224], "0 0 0", "1 1 1", True, 255.0),
            ),
            patch.object(
                getitune_converter,
                "create_calibration_dataset",
                return_value=(calibration_data, validation_labels),
            ) as mock_dataset,
            patch.object(getitune_converter, "quantize_model") as mock_quantize,
        ):
            getitune_converter._quantize_exported_model(config)

        mock_dataset.assert_called_once_with(
            input_shape=[1, 3, 224, 224],
            mean_values="0 0 0",
            scale_values="1 1 1",
            reverse_input_channels=True,
            subset_size=500,
            return_labels=True,
            dataset_path=ANY,
            dataset_type="imagenet-1k",
            intensity_scale=255.0,
        )
        mock_quantize.assert_called_once_with(
            model_path=fp32_xml,
            calibration_data=calibration_data,
            model_config=config,
            preset="mixed",
            validation_data=calibration_data,
            validation_labels=validation_labels,
            validation_samples=None,
            metric=ANY,
            accuracy_results=ANY,
        )

    def test_quantize_exported_model_skips_accuracy_for_non_classification(
        self,
        getitune_converter,
        sample_getitune_config,
    ):
        """Accuracy is not measured for non-classification model types."""
        config = {
            **sample_getitune_config,
            "getitune_task": "DETECTION",
            "model_type": "SSD",
            "dataset_type": "coco-detection",
            "labels": "IMAGENET1K_V1",
        }
        output_folder = getitune_converter.output_dir / "dino_v2_cls-fp16-ov"
        fp32_xml = output_folder / "dino_v2_cls_fp32.xml"
        _write_openvino_model(fp32_xml)
        calibration_data = [np.zeros((1, 3, 224, 224), dtype=np.float32)]

        with (
            patch.object(
                getitune_converter,
                "_read_preprocessing_from_model",
                return_value=([1, 3, 224, 224], "0 0 0", "1 1 1", True, 255.0),
            ),
            patch.object(
                getitune_converter,
                "create_calibration_dataset",
                return_value=(calibration_data, []),
            ) as mock_dataset,
            patch.object(getitune_converter, "quantize_model") as mock_quantize,
        ):
            getitune_converter._quantize_exported_model(config)

        assert mock_dataset.call_args.kwargs["return_labels"] is False
        mock_quantize.assert_called_once_with(
            model_path=fp32_xml,
            calibration_data=calibration_data,
            model_config=config,
            preset="mixed",
            validation_data=None,
            validation_labels=None,
            validation_samples=None,
            metric=ANY,
            accuracy_results=ANY,
        )

    def test_quantize_exported_model_honours_measure_accuracy_false(
        self,
        tmp_output_dir,
        tmp_cache_dir,
        mock_dataset_registry,
        sample_getitune_config,
    ):
        """measure_accuracy=False on the converter disables accuracy measurement entirely."""
        converter = GetituneConverter(
            output_dir=tmp_output_dir,
            cache_dir=tmp_cache_dir,
            verbose=True,
            dataset_registry=mock_dataset_registry,
            measure_accuracy=False,
        )
        config = {**sample_getitune_config, "labels": "IMAGENET1K_V1"}
        output_folder = converter.output_dir / "dino_v2_cls-fp16-ov"
        fp32_xml = output_folder / "dino_v2_cls_fp32.xml"
        _write_openvino_model(fp32_xml)
        calibration_data = [np.zeros((1, 3, 224, 224), dtype=np.float32)]

        with (
            patch.object(
                converter,
                "_read_preprocessing_from_model",
                return_value=([1, 3, 224, 224], "0 0 0", "1 1 1", True, 255.0),
            ),
            patch.object(
                converter,
                "create_calibration_dataset",
                return_value=(calibration_data, []),
            ) as mock_dataset,
            patch.object(converter, "quantize_model"),
        ):
            result = converter._quantize_exported_model(config)

        assert mock_dataset.call_args.kwargs["return_labels"] is False
        assert result.metric_name is None


class TestApplyConfigLabels:
    """Tests for GetituneConverter._apply_config_labels."""

    def test_no_labels_in_config_is_a_noop(self, getitune_converter, sample_getitune_config, tmp_path):
        """Without a labels config, no OpenVINO work is attempted."""
        with patch.object(getitune_converter, "get_labels") as mock_get_labels:
            getitune_converter._apply_config_labels(sample_getitune_config, tmp_path / "model.xml")
        mock_get_labels.assert_not_called()

    def test_unresolved_labels_logs_warning(self, getitune_converter, sample_getitune_config, tmp_path, caplog):
        """An unknown label set logs a warning and skips rewriting."""
        config = {**sample_getitune_config, "labels": "UNKNOWN_SET"}
        with (
            patch.object(getitune_converter, "get_labels", return_value=None),
            caplog.at_level(logging.WARNING),
        ):
            getitune_converter._apply_config_labels(config, tmp_path / "model.xml")
        assert "Could not load labels for: UNKNOWN_SET" in caplog.text

    def test_rewrites_labels_into_existing_models(self, getitune_converter, sample_getitune_config, tmp_path):
        """Labels are written to rt_info and the models are re-saved.

        The save must go to a temporary path that is then moved over the
        original, never writing back to the file currently being read (which
        OpenVINO memory-maps and would corrupt).
        """
        config = {**sample_getitune_config, "labels": "IMAGENET1K_V1"}
        fp16_xml = tmp_path / "dino_v2_cls.xml"
        fp32_xml = tmp_path / "dino_v2_cls_fp32.xml"
        _write_openvino_model(fp16_xml)
        _write_openvino_model(fp32_xml)
        missing_xml = tmp_path / "missing.xml"

        fake_ov = types.ModuleType("openvino")
        core = MagicMock()
        model = MagicMock()
        core.read_model.return_value = model
        setattr(fake_ov, "Core", MagicMock(return_value=core))

        def fake_save_model(_model, path, compress_to_fp16):
            # Never overwrite the source model directly.
            assert path != fp16_xml
            assert path != fp32_xml
            _write_openvino_model(Path(path))

        save_model = MagicMock(side_effect=fake_save_model)
        setattr(fake_ov, "save_model", save_model)

        with (
            patch.object(getitune_converter, "get_labels", return_value="cat dog"),
            patch.dict("sys.modules", {"openvino": fake_ov}),
        ):
            getitune_converter._apply_config_labels(config, fp16_xml, fp32_xml, missing_xml)

        model.set_rt_info.assert_called_with("cat dog", ["model_info", "labels"])
        assert save_model.call_count == 2
        assert save_model.call_args_list[0].kwargs["compress_to_fp16"] is True
        assert save_model.call_args_list[1].kwargs["compress_to_fp16"] is False
        # Final models exist and the temporary files were moved away.
        assert fp16_xml.exists()
        assert fp32_xml.exists()
        assert not (tmp_path / "dino_v2_cls_labeled_tmp.xml").exists()
        assert not (tmp_path / "dino_v2_cls_fp32_labeled_tmp.xml").exists()

    def test_handles_openvino_runtime_error(self, getitune_converter, sample_getitune_config, tmp_path, caplog):
        """A failure while rewriting labels is logged and swallowed."""
        config = {**sample_getitune_config, "labels": "IMAGENET1K_V1"}
        model_xml = tmp_path / "dino_v2_cls.xml"
        _write_openvino_model(model_xml)

        fake_ov = types.ModuleType("openvino")
        core = MagicMock()
        core.read_model.side_effect = RuntimeError("read failed")
        setattr(fake_ov, "Core", MagicMock(return_value=core))

        with (
            patch.object(getitune_converter, "get_labels", return_value="cat dog"),
            patch.dict("sys.modules", {"openvino": fake_ov}),
            caplog.at_level(logging.WARNING),
        ):
            getitune_converter._apply_config_labels(config, model_xml)

        assert "Could not apply labels to exported model" in caplog.text


class TestApplyPreprocessingOverrides:
    """Tests for GetituneConverter._apply_preprocessing_overrides."""

    def test_no_overrides_in_config_is_a_noop(self, getitune_converter, sample_getitune_config, tmp_path):
        """Without a preprocessing_overrides config, no OpenVINO work is attempted."""
        fake_ov = types.ModuleType("openvino")
        core = MagicMock()
        setattr(fake_ov, "Core", MagicMock(return_value=core))
        with patch.dict("sys.modules", {"openvino": fake_ov}):
            getitune_converter._apply_preprocessing_overrides(sample_getitune_config, tmp_path / "model.xml")
        core.read_model.assert_not_called()

    def test_rewrites_overrides_into_existing_models(self, getitune_converter, sample_getitune_config, tmp_path):
        """Override values are written to rt_info and the models are re-saved.

        The save must go to a temporary path that is then moved over the
        original, never writing back to the file currently being read (which
        OpenVINO memory-maps and would corrupt).
        """
        overrides = {
            "mean_values": "0.5 0.5 0.5",
            "scale_values": "0.5 0.5 0.5",
            "resize_type": "crop",
        }
        config = {**sample_getitune_config, "preprocessing_overrides": overrides}
        fp16_xml = tmp_path / "vit_tiny_cls.xml"
        fp32_xml = tmp_path / "vit_tiny_cls_fp32.xml"
        _write_openvino_model(fp16_xml)
        _write_openvino_model(fp32_xml)
        missing_xml = tmp_path / "missing.xml"

        fake_ov = types.ModuleType("openvino")
        core = MagicMock()
        model = MagicMock()
        core.read_model.return_value = model
        setattr(fake_ov, "Core", MagicMock(return_value=core))

        def fake_save_model(_model, path, compress_to_fp16):
            assert path != fp16_xml
            assert path != fp32_xml
            _write_openvino_model(Path(path))

        save_model = MagicMock(side_effect=fake_save_model)
        setattr(fake_ov, "save_model", save_model)

        with patch.dict("sys.modules", {"openvino": fake_ov}):
            getitune_converter._apply_preprocessing_overrides(config, fp16_xml, fp32_xml, missing_xml)

        for key, value in overrides.items():
            model.set_rt_info.assert_any_call(value, ["model_info", key])
        assert save_model.call_count == 2
        assert save_model.call_args_list[0].kwargs["compress_to_fp16"] is True
        assert save_model.call_args_list[1].kwargs["compress_to_fp16"] is False
        assert fp16_xml.exists()
        assert fp32_xml.exists()
        assert not (tmp_path / "vit_tiny_cls_preproc_tmp.xml").exists()
        assert not (tmp_path / "vit_tiny_cls_fp32_preproc_tmp.xml").exists()

    def test_handles_openvino_runtime_error(self, getitune_converter, sample_getitune_config, tmp_path, caplog):
        """A failure while rewriting overrides is logged and swallowed."""
        config = {**sample_getitune_config, "preprocessing_overrides": {"resize_type": "crop"}}
        model_xml = tmp_path / "vit_tiny_cls.xml"
        _write_openvino_model(model_xml)

        fake_ov = types.ModuleType("openvino")
        core = MagicMock()
        core.read_model.side_effect = RuntimeError("read failed")
        setattr(fake_ov, "Core", MagicMock(return_value=core))

        with (
            patch.dict("sys.modules", {"openvino": fake_ov}),
            caplog.at_level(logging.WARNING),
        ):
            getitune_converter._apply_preprocessing_overrides(config, model_xml)

        assert "Could not apply preprocessing overrides to exported model" in caplog.text


class TestApplyModelInfoOverrides:
    """Tests for GetituneConverter._apply_model_info_overrides."""

    def test_no_overrides_in_config_is_a_noop(self, getitune_converter, sample_getitune_config, tmp_path):
        """Without a model_info_overrides config, no OpenVINO work is attempted."""
        fake_ov = types.ModuleType("openvino")
        core = MagicMock()
        setattr(fake_ov, "Core", MagicMock(return_value=core))
        with patch.dict("sys.modules", {"openvino": fake_ov}):
            getitune_converter._apply_model_info_overrides(sample_getitune_config, tmp_path / "model.xml")
        core.read_model.assert_not_called()

    def test_rewrites_overrides_into_existing_models(self, getitune_converter, sample_getitune_config, tmp_path):
        """Model-info override values are written to rt_info and the models are re-saved."""
        overrides = {"model_type": "DETRInstSeg"}
        config = {**sample_getitune_config, "model_info_overrides": overrides}
        fp16_xml = tmp_path / "rfdetr_seg_small.xml"
        fp32_xml = tmp_path / "rfdetr_seg_small_fp32.xml"
        _write_openvino_model(fp16_xml)
        _write_openvino_model(fp32_xml)
        missing_xml = tmp_path / "missing.xml"

        fake_ov = types.ModuleType("openvino")
        core = MagicMock()
        model = MagicMock()
        core.read_model.return_value = model
        setattr(fake_ov, "Core", MagicMock(return_value=core))

        def fake_save_model(_model, path, compress_to_fp16):
            assert path != fp16_xml
            assert path != fp32_xml
            _write_openvino_model(Path(path))

        save_model = MagicMock(side_effect=fake_save_model)
        setattr(fake_ov, "save_model", save_model)

        with patch.dict("sys.modules", {"openvino": fake_ov}):
            getitune_converter._apply_model_info_overrides(config, fp16_xml, fp32_xml, missing_xml)

        model.set_rt_info.assert_called_with("DETRInstSeg", ["model_info", "model_type"])
        assert save_model.call_count == 2
        assert save_model.call_args_list[0].kwargs["compress_to_fp16"] is True
        assert save_model.call_args_list[1].kwargs["compress_to_fp16"] is False
        assert fp16_xml.exists()
        assert fp32_xml.exists()
        assert not (tmp_path / "rfdetr_seg_small_model_info_tmp.xml").exists()
        assert not (tmp_path / "rfdetr_seg_small_fp32_model_info_tmp.xml").exists()

    def test_handles_openvino_runtime_error(self, getitune_converter, sample_getitune_config, tmp_path, caplog):
        """A failure while rewriting model-info overrides is logged and swallowed."""
        config = {**sample_getitune_config, "model_info_overrides": {"model_type": "DETRInstSeg"}}
        model_xml = tmp_path / "rfdetr_seg_small.xml"
        _write_openvino_model(model_xml)

        fake_ov = types.ModuleType("openvino")
        core = MagicMock()
        core.read_model.side_effect = RuntimeError("read failed")
        setattr(fake_ov, "Core", MagicMock(return_value=core))

        with (
            patch.dict("sys.modules", {"openvino": fake_ov}),
            caplog.at_level(logging.WARNING),
        ):
            getitune_converter._apply_model_info_overrides(config, model_xml)

        assert "Could not apply model_info overrides to exported model" in caplog.text


class TestReadPreprocessingFromModel:
    """Tests for GetituneConverter._read_preprocessing_from_model."""

    def test_reads_all_params_from_rt_info(self, tmp_path):
        """_read_preprocessing_from_model extracts params from model rt_info."""
        model_path = tmp_path / "model.xml"
        model_path.write_text("<net/>")

        mock_model = MagicMock()
        mock_model.input.return_value = MagicMock(partial_shape=_make_partial_shape([1, 3, 640, 640]))

        def fake_get_rt_info(path):
            values = {
                "mean_values": "123.675 116.28 103.53",
                "scale_values": "58.395 57.12 57.375",
                "reverse_input_channels": "True",
            }
            key = path[1]
            if key in values:
                result = MagicMock()
                result.astype.return_value = values[key]
                return result
            msg = "Cannot get runtime attribute. Path to runtime attribute is incorrect."
            raise RuntimeError(msg)

        mock_model.get_rt_info.side_effect = fake_get_rt_info

        mock_ov = MagicMock()
        mock_ov.Core.return_value.read_model.return_value = mock_model

        input_shape, mean_values, scale_values, reverse, intensity_scale = (
            GetituneConverter._read_preprocessing_from_model(
                mock_ov,
                model_path,
            )
        )

        assert input_shape == [1, 3, 640, 640]
        assert mean_values == "123.675 116.28 103.53"
        assert scale_values == "58.395 57.12 57.375"
        assert reverse is True
        assert intensity_scale == pytest.approx(1.0)

    def test_reads_intensity_scale_for_scale_to_unit_u8(self, tmp_path):
        """A scale_to_unit/u8 model yields intensity_scale=255 (÷255 before mean/scale)."""
        model_path = tmp_path / "model.xml"
        model_path.write_text("<net/>")

        mock_model = MagicMock()
        mock_model.input.return_value = MagicMock(partial_shape=_make_partial_shape([1, 3, 224, 224]))

        def fake_get_rt_info(path):
            values = {
                "mean_values": "0.485 0.456 0.406",
                "scale_values": "0.229 0.224 0.225",
                "reverse_input_channels": "False",
                "intensity_mode": "scale_to_unit",
                "input_dtype": "u8",
            }
            key = path[1]
            if key in values:
                result = MagicMock()
                result.astype.return_value = values[key]
                return result
            msg = "Cannot get runtime attribute. Path to runtime attribute is incorrect."
            raise RuntimeError(msg)

        mock_model.get_rt_info.side_effect = fake_get_rt_info

        mock_ov = MagicMock()
        mock_ov.Core.return_value.read_model.return_value = mock_model

        _, _, _, reverse, intensity_scale = GetituneConverter._read_preprocessing_from_model(
            mock_ov,
            model_path,
        )

        assert reverse is False
        assert intensity_scale == pytest.approx(255.0)

    def test_reads_explicit_intensity_max_value(self, tmp_path):
        """An explicit intensity_max_value overrides the input_dtype default."""
        model_path = tmp_path / "model.xml"
        model_path.write_text("<net/>")

        mock_model = MagicMock()
        mock_model.input.return_value = MagicMock(partial_shape=_make_partial_shape([1, 3, 224, 224]))

        def fake_get_rt_info(path):
            values = {
                "intensity_mode": "scale_to_unit",
                "input_dtype": "u8",
                "intensity_max_value": "1023",
            }
            key = path[1]
            if key in values:
                result = MagicMock()
                result.astype.return_value = values[key]
                return result
            msg = "Cannot get runtime attribute. Path to runtime attribute is incorrect."
            raise RuntimeError(msg)

        mock_model.get_rt_info.side_effect = fake_get_rt_info

        mock_ov = MagicMock()
        mock_ov.Core.return_value.read_model.return_value = mock_model

        *_, intensity_scale = GetituneConverter._read_preprocessing_from_model(mock_ov, model_path)

        assert intensity_scale == pytest.approx(1023.0)

    def test_falls_back_to_defaults_when_rt_info_missing(self, tmp_path):
        """_read_preprocessing_from_model uses defaults when rt_info keys are absent."""
        model_path = tmp_path / "model.xml"
        model_path.write_text("<net/>")

        mock_model = MagicMock()
        mock_model.input.return_value = MagicMock(partial_shape=_make_partial_shape([1, 3, 224, 224]))
        mock_model.get_rt_info.side_effect = RuntimeError(
            "Cannot get runtime attribute. Path to runtime attribute is incorrect.",
        )

        mock_ov = MagicMock()
        mock_ov.Core.return_value.read_model.return_value = mock_model

        input_shape, mean_values, scale_values, reverse, intensity_scale = (
            GetituneConverter._read_preprocessing_from_model(
                mock_ov,
                model_path,
            )
        )

        assert input_shape == [1, 3, 224, 224]
        assert mean_values == "0 0 0"
        assert scale_values == "1 1 1"
        assert reverse is True
        assert intensity_scale == pytest.approx(1.0)

    def test_resolves_dynamic_batch_dimension_to_one(self, tmp_path):
        """A dynamic batch dimension is resolved to a concrete size of 1."""
        model_path = tmp_path / "model.xml"
        model_path.write_text("<net/>")

        mock_model = MagicMock()
        # Dynamic batch (None) like DINOv2/ViT exports; spatial dims are static.
        mock_model.input.return_value = MagicMock(partial_shape=_make_partial_shape([None, 3, 224, 224]))
        mock_model.get_rt_info.side_effect = RuntimeError(
            "Cannot get runtime attribute. Path to runtime attribute is incorrect.",
        )

        mock_ov = MagicMock()
        mock_ov.Core.return_value.read_model.return_value = mock_model

        input_shape, *_ = GetituneConverter._read_preprocessing_from_model(mock_ov, model_path)

        assert input_shape == [1, 3, 224, 224]
