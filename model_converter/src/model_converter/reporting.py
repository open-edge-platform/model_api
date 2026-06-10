#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Conversion summary report generation.

Collects per-model conversion outcomes (accuracies and status) in a structured
form and renders them as a console table and a Markdown report.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Status string constants
STATUS_OK = "OK"
STATUS_OK_NO_ACCURACY = "OK (no accuracy data)"
STATUS_ACCURACY_DROP = "ACCURACY DROP >5%"
STATUS_FAILED_CONVERSION = "FAILED: conversion"
STATUS_FAILED_QUANTIZATION = "FAILED: quantization"
STATUS_SKIPPED = "SKIPPED"

#: Maximum tolerated top-1 accuracy drop (percentage points) versus FP32.
DEFAULT_ACCURACY_DROP_THRESHOLD = 5.0

_REPORT_COLUMNS = (
    "Model Full Name",
    "Model Type",
    "Model Library",
    "Original URL",
    "Original Accuracy",
    "FP32 Accuracy",
    "FP16 Accuracy",
    "INT8 Accuracy",
    "Status",
)


@dataclass
class ConversionResult:
    """Structured outcome for a single model conversion.

    Accuracies are top-1 accuracy fractions in the ``[0.0, 1.0]`` range, or
    ``None`` when not measured.
    """

    model_short_name: str
    model_full_name: str
    model_type: str
    model_library: str
    original_url: str | None = None
    original_accuracy: float | None = None
    fp32_accuracy: float | None = None
    fp16_accuracy: float | None = None
    int8_accuracy: float | None = None
    status: str = STATUS_OK
    status_detail: str = ""


@dataclass
class AccuracyResults:
    """Container for accuracies measured during quantization."""

    original_accuracy: float | None = None
    fp32_accuracy: float | None = None
    fp16_accuracy: float | None = None
    int8_accuracy: float | None = None
    int8_succeeded: bool = False
    measured: bool = field(default=False)
    metric_name: str | None = None


def original_url_for_config(config: dict[str, Any]) -> str | None:
    """Resolve the exact download URL for a model configuration.

    Only exact download locations are returned: ``weights_url`` for PyTorch
    checkpoints and the Hugging Face repository URL for ``huggingface_repo``.
    Returns ``None`` when no exact download URL is available.

    Args:
        config: Model configuration dictionary.

    Returns:
        The download URL, or ``None`` if unavailable.
    """
    weights_url = config.get("weights_url")
    if weights_url:
        return str(weights_url)

    hf_repo = config.get("huggingface_repo")
    if hf_repo:
        return f"https://huggingface.co/{hf_repo}"

    return None


def determine_status(
    result: ConversionResult,
    *,
    converted: bool,
    quantized: bool,
    skipped: bool,
    threshold: float = DEFAULT_ACCURACY_DROP_THRESHOLD,
) -> tuple[str, str]:
    """Compute the status and detail for a conversion result.

    FP32 is the preferred baseline for quantization drops. When FP32 accuracy
    is absent (e.g. for YOLO11 models where no FP32 OV artifact is produced),
    ``original_accuracy`` is used as baseline instead. A drop is abnormal when
    ``(baseline - FP16)`` or ``(baseline - INT8)`` exceeds ``threshold``
    percentage points.

    Args:
        result: The conversion result holding accuracy values.
        converted: Whether the FP16 model was produced.
        quantized: Whether the INT8 model was produced.
        skipped: Whether processing was skipped (model already existed).
        threshold: Maximum tolerated accuracy drop in percentage points.

    Returns:
        A tuple of ``(status, status_detail)``.
    """
    if skipped:
        return STATUS_SKIPPED, "FP16 and INT8 models already existed"

    if not converted:
        return STATUS_FAILED_CONVERSION, "Model load or export failed"

    if not quantized:
        return STATUS_FAILED_QUANTIZATION, "INT8 model was not produced"

    # Determine the baseline accuracy for drop checks.
    # Prefer FP32 (standard OV export); fall back to original (e.g. YOLO PT model).
    baseline = result.fp32_accuracy if result.fp32_accuracy is not None else result.original_accuracy
    if baseline is None:
        return STATUS_OK_NO_ACCURACY, "No accuracy measurement available"

    baseline_pct = baseline * 100
    drops: list[str] = []

    if result.fp32_accuracy is not None and result.original_accuracy is not None:
        fp32_drop = result.original_accuracy * 100 - baseline_pct
        if fp32_drop > threshold:
            drops.append(f"FP32 drop {fp32_drop:.2f}%")

    if result.fp16_accuracy is not None:
        fp16_drop = baseline_pct - result.fp16_accuracy * 100
        if fp16_drop > threshold:
            drops.append(f"FP16 drop {fp16_drop:.2f}%")

    if result.int8_accuracy is not None:
        int8_drop = baseline_pct - result.int8_accuracy * 100
        if int8_drop > threshold:
            drops.append(f"INT8 drop {int8_drop:.2f}%")

    if drops:
        return STATUS_ACCURACY_DROP, "; ".join(drops)

    return STATUS_OK, ""


def _format_accuracy(value: float | None) -> str:
    """Format an accuracy fraction as a percentage string or ``N/A``."""
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _row_values(result: ConversionResult) -> list[str]:
    """Build the ordered cell values for a single result row."""
    return [
        result.model_full_name or result.model_short_name,
        result.model_type or "N/A",
        result.model_library or "N/A",
        result.original_url or "N/A",
        _format_accuracy(result.original_accuracy),
        _format_accuracy(result.fp32_accuracy),
        _format_accuracy(result.fp16_accuracy),
        _format_accuracy(result.int8_accuracy),
        result.status,
    ]


def format_console_table(results: list[ConversionResult]) -> str:
    """Render results as an aligned plain-text table for the console.

    Args:
        results: Conversion results to render.

    Returns:
        The formatted table as a string.
    """
    rows = [list(_REPORT_COLUMNS)] + [_row_values(r) for r in results]
    widths = [max(len(row[col]) for row in rows) for col in range(len(_REPORT_COLUMNS))]

    def render_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    separator = "  ".join("-" * widths[i] for i in range(len(_REPORT_COLUMNS)))

    lines = ["Conversion Summary Report", "", render_row(rows[0]), separator]
    lines.extend(render_row(row) for row in rows[1:])
    return "\n".join(lines)


def format_markdown_report(results: list[ConversionResult]) -> str:
    """Render results as a Markdown document with a summary table.

    Args:
        results: Conversion results to render.

    Returns:
        The Markdown report as a string.
    """
    header = "| " + " | ".join(_REPORT_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in _REPORT_COLUMNS) + " |"

    lines = ["# Conversion Summary Report", ""]
    if not results:
        lines.append("No models were processed.")
        return "\n".join(lines) + "\n"

    lines.append(header)
    lines.append(separator)
    for result in results:
        cells = [cell.replace("|", "\\|") for cell in _row_values(result)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_report(results: list[ConversionResult], path: Path) -> None:
    """Write the Markdown report to ``path``, creating parent directories.

    Args:
        results: Conversion results to render.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_markdown_report(results))


def _load_results_from_json(path: Path) -> list[ConversionResult]:
    """Load persisted ConversionResult objects from a JSON sidecar file.

    Args:
        path: Path to the JSON sidecar file.

    Returns:
        List of ConversionResult objects, or an empty list if the file does not exist.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return [ConversionResult(**entry) for entry in data]
    except (json.JSONDecodeError, TypeError, KeyError):
        return []


def upsert_result(result: ConversionResult, path: Path) -> None:
    """Upsert a single result into the report files.

    Reads the existing JSON sidecar (``path.with_suffix('.json')``), replaces
    the entry whose ``model_short_name`` matches ``result``, or appends it if
    absent.  Then writes the updated JSON sidecar and regenerates the Markdown
    report at ``path``.

    Args:
        result: Conversion result to upsert.
        path: Destination Markdown file path.
    """
    path = Path(path)
    json_path = path.with_suffix(".json")

    existing = _load_results_from_json(json_path)
    updated = [r for r in existing if r.model_short_name != result.model_short_name]
    updated.append(result)

    path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps([dataclasses.asdict(r) for r in updated], indent=2))
    path.write_text(format_markdown_report(updated))
