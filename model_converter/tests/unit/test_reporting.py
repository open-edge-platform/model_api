#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for the conversion summary reporting module."""

from typing import Any

from model_converter.reporting import (
    STATUS_ACCURACY_DROP,
    STATUS_FAILED_CONVERSION,
    STATUS_FAILED_QUANTIZATION,
    STATUS_OK,
    STATUS_OK_NO_ACCURACY,
    STATUS_SKIPPED,
    AccuracyResults,
    ConversionResult,
    determine_status,
    format_console_table,
    format_markdown_report,
    original_url_for_config,
    upsert_result,
    write_markdown_report,
)


def _result(**kwargs) -> ConversionResult:
    base: dict[str, Any] = {
        "model_short_name": "m",
        "model_full_name": "Model",
        "model_type": "Classification",
        "model_library": "timm",
    }
    base.update(kwargs)
    return ConversionResult(**base)


class TestAccuracyResults:
    """Tests for the AccuracyResults dataclass defaults."""

    def test_defaults(self):
        acc = AccuracyResults()
        assert acc.original_accuracy is None
        assert acc.fp32_accuracy is None
        assert acc.fp16_accuracy is None
        assert acc.int8_accuracy is None
        assert acc.int8_succeeded is False
        assert acc.measured is False


class TestOriginalUrlForConfig:
    """Tests for original_url_for_config."""

    def test_weights_url_takes_precedence(self):
        config = {"weights_url": "https://example.com/w.pth", "huggingface_repo": "org/model"}
        assert original_url_for_config(config) == "https://example.com/w.pth"

    def test_huggingface_repo(self):
        assert original_url_for_config({"huggingface_repo": "org/model"}) == "https://huggingface.co/org/model"

    def test_none_when_unavailable(self):
        assert original_url_for_config({}) is None


class TestDetermineStatus:
    """Tests for every branch of determine_status."""

    def test_skipped(self):
        status, detail = determine_status(_result(), converted=False, quantized=False, skipped=True)
        assert status == STATUS_SKIPPED
        assert detail

    def test_failed_conversion(self):
        status, _ = determine_status(_result(), converted=False, quantized=False, skipped=False)
        assert status == STATUS_FAILED_CONVERSION

    def test_failed_quantization(self):
        status, _ = determine_status(_result(), converted=True, quantized=False, skipped=False)
        assert status == STATUS_FAILED_QUANTIZATION

    def test_no_accuracy_data(self):
        status, _ = determine_status(_result(fp32_accuracy=None), converted=True, quantized=True, skipped=False)
        assert status == STATUS_OK_NO_ACCURACY

    def test_ok_within_threshold(self):
        result = _result(fp32_accuracy=0.90, fp16_accuracy=0.89, int8_accuracy=0.88)
        status, detail = determine_status(result, converted=True, quantized=True, skipped=False)
        assert status == STATUS_OK
        assert detail == ""

    def test_boundary_exactly_5_percent_is_ok(self):
        result = _result(fp32_accuracy=0.90, int8_accuracy=0.85)
        status, _ = determine_status(result, converted=True, quantized=True, skipped=False)
        assert status == STATUS_OK

    def test_fp16_drop_flagged(self):
        result = _result(fp32_accuracy=0.90, fp16_accuracy=0.80, int8_accuracy=0.89)
        status, detail = determine_status(result, converted=True, quantized=True, skipped=False)
        assert status == STATUS_ACCURACY_DROP
        assert "FP16 drop" in detail

    def test_int8_drop_flagged(self):
        result = _result(fp32_accuracy=0.90, fp16_accuracy=0.89, int8_accuracy=0.70)
        status, detail = determine_status(result, converted=True, quantized=True, skipped=False)
        assert status == STATUS_ACCURACY_DROP
        assert "INT8 drop" in detail

    def test_fp32_conversion_drop_flagged(self):
        result = _result(original_accuracy=0.90, fp32_accuracy=0.80, fp16_accuracy=0.80, int8_accuracy=0.79)
        status, detail = determine_status(result, converted=True, quantized=True, skipped=False)
        assert status == STATUS_ACCURACY_DROP
        assert "FP32 drop" in detail

    def test_original_within_threshold_is_ok(self):
        result = _result(original_accuracy=0.90, fp32_accuracy=0.89, fp16_accuracy=0.89, int8_accuracy=0.88)
        status, detail = determine_status(result, converted=True, quantized=True, skipped=False)
        assert status == STATUS_OK
        assert detail == ""


class TestRendering:
    """Tests for console and markdown rendering."""

    def test_console_table_includes_values_and_na(self):
        results = [
            _result(
                original_accuracy=0.9123,
                fp32_accuracy=0.9012,
                fp16_accuracy=0.9,
                int8_accuracy=0.88,
                status=STATUS_OK,
            ),
            _result(model_full_name="NoAcc", original_url=None, status=STATUS_OK_NO_ACCURACY),
        ]
        table = format_console_table(results)
        assert "Conversion Summary Report" in table
        assert "Original Accuracy" in table
        assert "91.23%" in table
        assert "90.12%" in table
        assert "N/A" in table
        assert "Status" in table

    def test_markdown_report_escapes_pipes_and_renders_rows(self):
        results = [_result(model_full_name="A|B", fp32_accuracy=0.5, status=STATUS_OK)]
        md = format_markdown_report(results)
        assert md.startswith("# Conversion Summary Report")
        assert "A\\|B" in md
        assert "50.00%" in md
        assert md.endswith("\n")

    def test_markdown_report_empty(self):
        md = format_markdown_report([])
        assert "No models were processed." in md

    def test_console_table_uses_short_name_when_full_name_missing(self):
        result = ConversionResult(
            model_short_name="short",
            model_full_name="",
            model_type="",
            model_library="",
        )
        table = format_console_table([result])
        assert "short" in table

    def test_write_markdown_report_creates_file(self, tmp_path):
        path = tmp_path / "nested" / "report.md"
        write_markdown_report([_result(status=STATUS_OK)], path)
        assert path.exists()
        assert "# Conversion Summary Report" in path.read_text()


class TestUpsertResult:
    """Tests for upsert_result."""

    def test_creates_files_on_first_write(self, tmp_path):
        """upsert_result creates both the Markdown and JSON sidecar."""
        path = tmp_path / "report.md"
        upsert_result(_result(model_short_name="m1", model_full_name="Model One", status=STATUS_OK), path)
        assert path.exists()
        assert path.with_suffix(".json").exists()
        assert "# Conversion Summary Report" in path.read_text()

    def test_appends_new_entry(self, tmp_path):
        """upsert_result appends a new entry when the model is not yet tracked."""
        path = tmp_path / "report.md"
        upsert_result(_result(model_short_name="m1", model_full_name="Model One", status=STATUS_OK), path)
        upsert_result(_result(model_short_name="m2", model_full_name="Model Two", status=STATUS_OK), path)
        md = path.read_text()
        assert "Model One" in md
        assert "Model Two" in md

    def test_replaces_existing_entry(self, tmp_path):
        """upsert_result replaces an existing entry for the same model_short_name."""
        path = tmp_path / "report.md"
        upsert_result(
            _result(model_short_name="m1", model_full_name="Model One", status=STATUS_FAILED_CONVERSION),
            path,
        )
        upsert_result(_result(model_short_name="m1", model_full_name="Model One", status=STATUS_OK), path)
        md = path.read_text()
        assert STATUS_OK in md
        assert STATUS_FAILED_CONVERSION not in md
        # Only one row for Model One
        assert md.count("Model One") == 1

    def test_preserves_other_entries_on_replace(self, tmp_path):
        """upsert_result keeps all other entries when replacing one."""
        path = tmp_path / "report.md"
        upsert_result(_result(model_short_name="m1", model_full_name="Model One", status=STATUS_OK), path)
        upsert_result(_result(model_short_name="m2", model_full_name="Model Two", status=STATUS_OK), path)
        upsert_result(
            _result(model_short_name="m1", model_full_name="Model One", status=STATUS_FAILED_CONVERSION),
            path,
        )
        md = path.read_text()
        assert "Model Two" in md
        assert STATUS_FAILED_CONVERSION in md

    def test_creates_parent_directories(self, tmp_path):
        """upsert_result creates missing parent directories."""
        path = tmp_path / "nested" / "deep" / "report.md"
        upsert_result(_result(status=STATUS_OK), path)
        assert path.exists()

    def test_json_sidecar_is_valid(self, tmp_path):
        """upsert_result writes valid JSON to the sidecar file."""
        import json

        path = tmp_path / "report.md"
        upsert_result(_result(model_short_name="m1", status=STATUS_OK), path)
        data = json.loads(path.with_suffix(".json").read_text())
        assert isinstance(data, list)
        assert data[0]["model_short_name"] == "m1"

    def test_handles_corrupt_json_sidecar(self, tmp_path):
        """upsert_result gracefully handles a corrupt JSON sidecar by treating it as empty."""
        import json

        path = tmp_path / "report.md"
        json_path = path.with_suffix(".json")
        json_path.write_text("{ not valid json }")

        upsert_result(_result(model_short_name="m1", status=STATUS_OK), path)

        data = json.loads(json_path.read_text())
        assert len(data) == 1
        assert data[0]["model_short_name"] == "m1"
