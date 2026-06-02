#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Conversion summary report generation.

Collects per-model conversion outcomes (accuracies and status) in a structured
form and renders them as a console table and a Markdown report.
"""

from __future__ import annotations

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
    fp32_accuracy: float | None = None
    fp16_accuracy: float | None = None
    int8_accuracy: float | None = None
    status: str = STATUS_OK
    status_detail: str = ""


@dataclass
class AccuracyResults:
    """Container for accuracies measured during quantization."""

    fp32_accuracy: float | None = None
    fp16_accuracy: float | None = None
    int8_accuracy: float | None = None
    int8_succeeded: bool = False
    measured: bool = field(default=False)


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

    FP32 is the baseline. A drop is abnormal when ``(FP32 - FP16)`` or
    ``(FP32 - INT8)`` exceeds ``threshold`` percentage points.

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

    if result.fp32_accuracy is None:
        return STATUS_OK_NO_ACCURACY, "No accuracy measurement available"

    fp32_pct = result.fp32_accuracy * 100
    drops: list[str] = []

    if result.fp16_accuracy is not None:
        fp16_drop = fp32_pct - result.fp16_accuracy * 100
        if fp16_drop > threshold:
            drops.append(f"FP16 drop {fp16_drop:.2f}%")

    if result.int8_accuracy is not None:
        int8_drop = fp32_pct - result.int8_accuracy * 100
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
