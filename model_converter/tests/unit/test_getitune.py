#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for GetituneConverter."""

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
def training_extensions_dir(tmp_path):
    """Temporary training_extensions checkout with export script."""
    repo_dir = tmp_path / "training_extensions"
    repo_dir.mkdir()
    (repo_dir / "export_pretrained_models.py").write_text("print('export')\n")
    library_dir = repo_dir / "library"
    library_dir.mkdir()
    (library_dir / "pyproject.toml").write_text('[project]\nname = "getitune"\n')
    return repo_dir


@pytest.fixture
def getitune_converter(tmp_output_dir, tmp_cache_dir, training_extensions_dir, mock_dataset_registry):
    """GetituneConverter with temporary directories."""
    return GetituneConverter(
        output_dir=tmp_output_dir,
        cache_dir=tmp_cache_dir,
        verbose=True,
        dataset_registry=mock_dataset_registry,
        training_extensions_dir=training_extensions_dir,
    )


class TestProcessModelConfig:
    """Tests for GetituneConverter.process_model_config."""

    def test_process_model_config_succeeds(
        self,
        getitune_converter,
        sample_getitune_config,
        training_extensions_dir,
        tmp_path,
    ):
        """process_model_config exports, repackages, and quantizes a valid model."""
        export_root = tmp_path / "getitune_export_process"
        expected_xml = export_root / "multi_class_cls" / "dino_v2" / "exported_model.xml"

        def fake_run(*args, **kwargs):
            _write_openvino_model(expected_xml)
            return MagicMock(returncode=0, stdout="ok", stderr="")

        fake_openvino = _make_openvino_module()

        with (
            patch("model_converter.converters.getitune.tempfile.mkdtemp", return_value=str(export_root)),
            patch("model_converter.converters.getitune.subprocess.run", side_effect=fake_run),
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
        mock_copy_readme.assert_called_once_with(sample_getitune_config, output_folder, variant="fp16")
        mock_quantize.assert_called_once_with(sample_getitune_config)
        assert not export_root.exists()
        assert training_extensions_dir.exists()

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

    def test_process_model_config_returns_false_without_training_extensions_dir(
        self,
        tmp_output_dir,
        tmp_cache_dir,
        dataset_dir,
        sample_getitune_config,
        caplog,
    ):
        """process_model_config returns False when training_extensions_dir is not provided."""
        converter = GetituneConverter(
            output_dir=tmp_output_dir,
            cache_dir=tmp_cache_dir,
            verbose=True,
            dataset_registry=dataset_dir,
            training_extensions_dir=None,
        )

        with caplog.at_level(logging.ERROR):
            result = converter.process_model_config(sample_getitune_config)

        assert result is False
        assert "training_extensions_dir is required" in caplog.text

    def test_process_model_config_returns_false_when_training_extensions_dir_missing(
        self,
        tmp_output_dir,
        tmp_cache_dir,
        dataset_dir,
        sample_getitune_config,
        tmp_path,
        caplog,
    ):
        """process_model_config returns False when training_extensions_dir is missing."""
        converter = GetituneConverter(
            output_dir=tmp_output_dir,
            cache_dir=tmp_cache_dir,
            verbose=True,
            dataset_registry=dataset_dir,
            training_extensions_dir=tmp_path / "missing_training_extensions",
        )

        with caplog.at_level(logging.ERROR):
            result = converter.process_model_config(sample_getitune_config)

        assert result is False
        assert "training_extensions directory not found" in caplog.text

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
        training_extensions_dir,
        tmp_path,
        caplog,
    ):
        """process_model_config logs model description when config includes it."""
        config = sample_getitune_config | {"description": "A test getitune model"}
        export_root = tmp_path / "getitune_export_desc"
        expected_xml = export_root / "multi_class_cls" / "dino_v2" / "exported_model.xml"

        def fake_run(*args, **kwargs):
            expected_xml.parent.mkdir(parents=True, exist_ok=True)
            expected_xml.write_text("<net/>")
            expected_xml.with_suffix(".bin").write_bytes(b"bin")
            return MagicMock(returncode=0, stdout="ok", stderr="")

        fake_openvino = _make_openvino_module()

        with (
            patch("model_converter.converters.getitune.tempfile.mkdtemp", return_value=str(export_root)),
            patch("model_converter.converters.getitune.subprocess.run", side_effect=fake_run),
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

    def test_run_export_builds_expected_command_and_returns_standard_path(
        self,
        getitune_converter,
        sample_getitune_config,
        tmp_path,
    ):
        """_run_export invokes export_pretrained_models.py via uv run and returns the standard output path."""
        export_root = tmp_path / "getitune_export_standard"
        expected_xml = export_root / "multi_class_cls" / "dino_v2" / "exported_model.xml"

        def fake_run(*args, **kwargs):
            _write_openvino_model(expected_xml)
            return MagicMock(returncode=0, stdout="ok", stderr="")

        with (
            patch("model_converter.converters.getitune.tempfile.mkdtemp", return_value=str(export_root)),
            patch("model_converter.converters.getitune.subprocess.run", side_effect=fake_run) as mock_run,
        ):
            result = getitune_converter._run_export(sample_getitune_config)

        assert result == expected_xml
        command = mock_run.call_args.args[0]
        library_dir = getitune_converter.training_extensions_dir / "library"
        assert command == [
            "uv",
            "run",
            "--project",
            str(library_dir),
            "--extra",
            "cpu",
            "python",
            str(getitune_converter.training_extensions_dir / "export_pretrained_models.py"),
            "--task",
            "MULTI_CLASS_CLS",
            "--model",
            "dino_v2",
            "--output-dir",
            str(export_root),
            "--format",
            "OPENVINO",
            "--precision",
            "FP16",
        ]
        assert mock_run.call_args.kwargs == {
            "cwd": str(getitune_converter.training_extensions_dir),
            "capture_output": True,
            "text": True,
            "check": False,
        }

    def test_run_export_falls_back_to_first_found_xml(self, getitune_converter, sample_getitune_config, tmp_path):
        """_run_export falls back to searching for any exported XML file."""
        export_root = tmp_path / "getitune_export_fallback"
        fallback_xml = export_root / "other" / "nested" / "model.xml"

        def fake_run(*args, **kwargs):
            _write_openvino_model(fallback_xml)
            return MagicMock(returncode=0, stdout="ok", stderr="")

        with (
            patch("model_converter.converters.getitune.tempfile.mkdtemp", return_value=str(export_root)),
            patch("model_converter.converters.getitune.subprocess.run", side_effect=fake_run),
        ):
            result = getitune_converter._run_export(sample_getitune_config)

        assert result == fallback_xml

    def test_run_export_raises_on_non_zero_return_code(self, getitune_converter, sample_getitune_config, tmp_path):
        """_run_export raises RuntimeError when the export script fails."""
        export_root = tmp_path / "getitune_export_error"

        with (
            patch("model_converter.converters.getitune.tempfile.mkdtemp", return_value=str(export_root)),
            patch(
                "model_converter.converters.getitune.subprocess.run",
                return_value=MagicMock(returncode=1, stdout="", stderr="boom"),
            ),
            pytest.raises(RuntimeError, match="return code 1"),
        ):
            getitune_converter._run_export(sample_getitune_config)

    def test_run_export_raises_when_task_is_missing(self, getitune_converter, sample_getitune_config):
        """_run_export requires getitune_task."""
        config = dict(sample_getitune_config)
        config.pop("getitune_task")

        with pytest.raises(ValueError, match="must define 'getitune_task'"):
            getitune_converter._run_export(config)

    def test_run_export_raises_when_recipe_is_missing(self, getitune_converter, sample_getitune_config):
        """_run_export requires getitune_recipe."""
        config = dict(sample_getitune_config)
        config.pop("getitune_recipe")

        with pytest.raises(ValueError, match="must define 'getitune_recipe'"):
            getitune_converter._run_export(config)

    def test_run_export_raises_when_export_script_is_missing(
        self,
        tmp_output_dir,
        tmp_cache_dir,
        dataset_dir,
        sample_getitune_config,
        tmp_path,
    ):
        """_run_export raises FileNotFoundError when export_pretrained_models.py is absent."""
        training_extensions_dir = tmp_path / "training_extensions"
        training_extensions_dir.mkdir()
        converter = GetituneConverter(
            output_dir=tmp_output_dir,
            cache_dir=tmp_cache_dir,
            verbose=True,
            dataset_registry=dataset_dir,
            training_extensions_dir=training_extensions_dir,
        )

        with pytest.raises(FileNotFoundError, match="Export script not found"):
            converter._run_export(sample_getitune_config)

    def test_run_export_raises_when_library_pyproject_is_missing(
        self,
        tmp_output_dir,
        tmp_cache_dir,
        dataset_dir,
        sample_getitune_config,
        tmp_path,
    ):
        """_run_export raises FileNotFoundError when library/pyproject.toml is absent."""
        training_extensions_dir = tmp_path / "training_extensions"
        training_extensions_dir.mkdir()
        (training_extensions_dir / "export_pretrained_models.py").write_text("print('export')\n")
        converter = GetituneConverter(
            output_dir=tmp_output_dir,
            cache_dir=tmp_cache_dir,
            verbose=True,
            dataset_registry=dataset_dir,
            training_extensions_dir=training_extensions_dir,
        )

        with pytest.raises(FileNotFoundError, match="getitune library project not found"):
            converter._run_export(sample_getitune_config)

    def test_run_export_raises_when_no_xml_files_are_found(
        self,
        getitune_converter,
        sample_getitune_config,
        tmp_path,
    ):
        """_run_export raises FileNotFoundError when the export output contains no XML files."""
        export_root = tmp_path / "getitune_export_empty"

        with (
            patch("model_converter.converters.getitune.tempfile.mkdtemp", return_value=str(export_root)),
            patch(
                "model_converter.converters.getitune.subprocess.run",
                return_value=MagicMock(returncode=0, stdout="ok", stderr=""),
            ),
            pytest.raises(FileNotFoundError, match="No exported model found"),
        ):
            getitune_converter._run_export(sample_getitune_config)


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
                return_value=([1, 3, 224, 224], "123.675 116.28 103.53", "58.395 57.12 57.375", True),
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
                return_value=([1, 3, 224, 224], "0 0 0", "1 1 1", True),
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
                return_value=([1, 3, 224, 224], "0 0 0", "1 1 1", True),
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
                return_value=([1, 3, 224, 224], "0 0 0", "1 1 1", True),
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
                return_value=([1, 3, 224, 224], "0 0 0", "1 1 1", True),
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
        training_extensions_dir,
        mock_dataset_registry,
        sample_getitune_config,
    ):
        """measure_accuracy=False on the converter disables accuracy measurement entirely."""
        converter = GetituneConverter(
            output_dir=tmp_output_dir,
            cache_dir=tmp_cache_dir,
            verbose=True,
            dataset_registry=mock_dataset_registry,
            training_extensions_dir=training_extensions_dir,
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
                return_value=([1, 3, 224, 224], "0 0 0", "1 1 1", True),
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


class TestReadPreprocessingFromModel:
    """Tests for GetituneConverter._read_preprocessing_from_model."""

    def test_reads_all_params_from_rt_info(self, tmp_path):
        """_read_preprocessing_from_model extracts params from model rt_info."""
        model_path = tmp_path / "model.xml"
        model_path.write_text("<net/>")

        mock_model = MagicMock()
        mock_model.input.return_value = MagicMock(shape=[1, 3, 640, 640])

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

        input_shape, mean_values, scale_values, reverse = GetituneConverter._read_preprocessing_from_model(
            mock_ov,
            model_path,
        )

        assert input_shape == [1, 3, 640, 640]
        assert mean_values == "123.675 116.28 103.53"
        assert scale_values == "58.395 57.12 57.375"
        assert reverse is True

    def test_falls_back_to_defaults_when_rt_info_missing(self, tmp_path):
        """_read_preprocessing_from_model uses defaults when rt_info keys are absent."""
        model_path = tmp_path / "model.xml"
        model_path.write_text("<net/>")

        mock_model = MagicMock()
        mock_model.input.return_value = MagicMock(shape=[1, 3, 224, 224])
        mock_model.get_rt_info.side_effect = RuntimeError(
            "Cannot get runtime attribute. Path to runtime attribute is incorrect.",
        )

        mock_ov = MagicMock()
        mock_ov.Core.return_value.read_model.return_value = mock_model

        input_shape, mean_values, scale_values, reverse = GetituneConverter._read_preprocessing_from_model(
            mock_ov,
            model_path,
        )

        assert input_shape == [1, 3, 224, 224]
        assert mean_values == "0 0 0"
        assert scale_values == "1 1 1"
        assert reverse is True
