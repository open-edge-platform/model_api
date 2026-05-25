"""Noise floor measurement utility for comparator tolerance handling."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

_NOISY_CV_THRESHOLD = 0.01


@dataclass
class FieldStats:
    mean: float
    std: float
    min: float
    max: float
    cv: float


@dataclass
class NoiseFloorReport:
    n_runs: int
    per_field: dict[str, FieldStats]
    summary: str


def _compute_field_stats(values: list[float]) -> FieldStats:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    amin = float(arr.min())
    amax = float(arr.max())
    cv = float(std / mean) if mean != 0.0 else 0.0
    return FieldStats(mean=mean, std=std, min=amin, max=amax, cv=cv)


def _make_summary(per_field: dict[str, FieldStats]) -> str:
    noisy = sorted(name for name, stats in per_field.items() if abs(stats.cv) > _NOISY_CV_THRESHOLD)
    if not noisy:
        return f"No noisy fields detected (cv threshold: {_NOISY_CV_THRESHOLD})."
    listed = ", ".join(noisy)
    return f"Noisy fields (cv > {_NOISY_CV_THRESHOLD}): {listed}"


def measure_noise_floor(
    model_fn: Callable[[object], dict[str, float]],
    input_data: object,
    *,
    n_runs: int = 10,
) -> NoiseFloorReport:
    """Run ``model_fn(input_data)`` ``n_runs`` times and compute per-field stats.

    ``model_fn`` must return a flat dict mapping field name to numeric value.
    All runs must produce the same set of field names.
    """
    if n_runs < 1:
        msg = f"n_runs must be >= 1, got {n_runs}"
        raise ValueError(msg)

    collected: dict[str, list[float]] = {}
    for i in range(n_runs):
        out = model_fn(input_data)
        if not isinstance(out, dict):
            msg = f"model_fn must return dict, got {type(out).__name__}"
            raise TypeError(msg)
        if i == 0:
            for key in out:
                collected[key] = []
        elif set(out.keys()) != set(collected.keys()):
            msg = f"Inconsistent field names on run {i}: expected {sorted(collected)}, got {sorted(out)}"
            raise ValueError(msg)
        for key, value in out.items():
            collected[key].append(float(value))

    per_field = {name: _compute_field_stats(values) for name, values in collected.items()}
    summary = _make_summary(per_field)
    return NoiseFloorReport(n_runs=n_runs, per_field=per_field, summary=summary)


def save_noise_floor_report(report: NoiseFloorReport, path: str | Path) -> None:
    """Serialize ``report`` to JSON at ``path``."""
    payload = {
        "n_runs": report.n_runs,
        "per_field": {name: asdict(stats) for name, stats in report.per_field.items()},
        "summary": report.summary,
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_noise_floor_report(path: str | Path) -> NoiseFloorReport:
    """Load a NoiseFloorReport from a JSON file written by ``save_noise_floor_report``."""
    payload = json.loads(Path(path).read_text())
    per_field = {name: FieldStats(**stats) for name, stats in payload["per_field"].items()}
    return NoiseFloorReport(
        n_runs=int(payload["n_runs"]),
        per_field=per_field,
        summary=str(payload["summary"]),
    )
