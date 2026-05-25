"""Fingerprint comparison helpers for future comparator policies."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np

_HIST_BINS = 16
_SAMPLE_COUNT = 16
_THUMB_SIZE = 16


def _round4(value: float) -> float:
    return round(float(value), 4)


def _round3(value: float) -> float:
    return round(float(value), 3)


def _compute_spatial_moments(plane: np.ndarray) -> dict[str, float] | None:
    if plane.ndim != 2 or plane.size == 0:
        return None
    # Use absolute values so negative entries don't cancel weights.
    weights = np.abs(plane.astype(np.float64))
    total = weights.sum()
    h, w = plane.shape
    if total <= 0:
        cy = (h - 1) / 2.0 if h > 0 else 0.0
        cx = (w - 1) / 2.0 if w > 0 else 0.0
        return {"cx": _round4(cx), "cy": _round4(cy)}
    indices = np.indices(plane.shape, dtype=np.float64)
    cy = float((indices[0] * weights).sum() / total)
    cx = float((indices[1] * weights).sum() / total)
    return {"cx": _round4(cx), "cy": _round4(cy)}


def _compute_thumbnail(array: np.ndarray) -> list[list[float]] | None:
    if array.ndim < 2:
        return None
    plane = array.mean(axis=tuple(range(array.ndim - 2))) if array.ndim > 2 else array
    if plane.size == 0:
        return None
    plane = np.ascontiguousarray(plane.astype(np.float32))
    resized = cv2.resize(plane, (_THUMB_SIZE, _THUMB_SIZE), interpolation=cv2.INTER_AREA)
    rows = cast(list[list[float]], resized.tolist())
    return [[_round3(v) for v in row] for row in rows]


def compute_fingerprint(array: np.ndarray | list[np.ndarray] | None) -> dict | list[dict] | None:
    """Compute a compact, JSON-serializable statistical fingerprint of ``array``.

    Returns ``None`` for ``None`` input. For 0-size arrays, numeric stats are
    ``None`` and list-typed stats are empty.

    For ``list[np.ndarray]`` input (e.g., per-instance saliency maps), returns
    a list of fingerprints, one per array.
    """
    if array is None:
        return None

    if isinstance(array, list):
        fps: list[dict[str, Any]] = []
        for arr in array:
            fp = compute_fingerprint(arr)
            if isinstance(fp, dict):
                fps.append(fp)
        return fps

    shape = [int(d) for d in array.shape]
    dtype = str(array.dtype)

    if array.size == 0:
        return {
            "shape": shape,
            "dtype": dtype,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "l2_norm": None,
            "percentiles": {"p1": None, "p50": None, "p99": None},
            "histogram": [],
            "spatial_moments": None,
            "argmax_index": [],
            "thumbnail": None,
            "sample_values": [],
        }

    flat = array.reshape(-1)
    flat_f64 = flat.astype(np.float64, copy=False)

    amin = float(flat_f64.min())
    amax = float(flat_f64.max())
    amean = float(flat_f64.mean())
    astd = float(flat_f64.std())
    al2 = float(np.sqrt(np.sum(flat_f64 * flat_f64)))

    p1, p50, p99 = (float(v) for v in np.percentile(flat_f64, [1, 50, 99]))

    if amin == amax:
        histogram = [int(flat_f64.size)] + [0] * (_HIST_BINS - 1)
    else:
        hist, _ = np.histogram(flat_f64, bins=_HIST_BINS, range=(amin, amax))
        histogram = [int(v) for v in hist.tolist()]

    if array.ndim == 1:
        spatial_moments: dict[str, float] | None = None
    elif array.ndim == 2:
        spatial_moments = _compute_spatial_moments(array)
    else:
        plane = array.mean(axis=tuple(range(array.ndim - 2)))
        spatial_moments = _compute_spatial_moments(plane)

    argmax_flat = int(np.argmax(flat_f64))
    argmax_index = [int(i) for i in np.unravel_index(argmax_flat, array.shape)]

    thumbnail = _compute_thumbnail(array) if array.ndim >= 2 else None

    sample_idx = np.linspace(0, array.size - 1, _SAMPLE_COUNT, dtype=int)
    sample_values = [_round4(flat_f64[i]) for i in sample_idx]

    return {
        "shape": shape,
        "dtype": dtype,
        "min": _round4(amin),
        "max": _round4(amax),
        "mean": _round4(amean),
        "std": _round4(astd),
        "l2_norm": _round4(al2),
        "percentiles": {
            "p1": _round4(p1),
            "p50": _round4(p50),
            "p99": _round4(p99),
        },
        "histogram": histogram,
        "spatial_moments": spatial_moments,
        "argmax_index": argmax_index,
        "thumbnail": thumbnail,
        "sample_values": sample_values,
    }
