"""Comparison policy types and report containers for comparator dispatch."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from model_api.models.result import Label


from tests.accuracy.comparator.fingerprint import compute_fingerprint
from tests.accuracy.comparator.matching import match_by_bbox_iou, match_by_mask_iou
from tests.accuracy.comparator.storage import (
    compute_class_map_iou,
    compute_mean_iou,
    load_binary_mask,
    load_class_map,
    load_instance_masks,
)

if TYPE_CHECKING:
    from pathlib import Path


class ComparisonPolicy(Enum):
    """Policies available for comparing a result field against reference data."""

    EXACT = "exact"
    NUMERIC_CLOSE = "numeric_close"
    MASK_IOU = "mask_iou"
    STAT_FINGERPRINT = "stat_fingerprint"
    LABEL_CLOSE = "label_close"


@dataclass
class FieldResult:
    """Outcome for comparing a single field."""

    policy: ComparisonPolicy
    passed: bool
    message: str
    actual_summary: str
    reference_summary: str
    delta: float | None
    tolerance: float | None
    pct_over_budget: float | None


@dataclass
class ComparisonReport:
    """Aggregate outcome for a full result comparison."""

    passed: bool
    field_results: dict[str, FieldResult]

    def pretty(self) -> str:
        """Render a multi-line human-readable failure report."""
        failing = [name for name, fr in self.field_results.items() if not fr.passed]
        header = (
            f"Comparison PASSED for {len(self.field_results)} fields"
            if self.passed
            else f"Comparison FAILED for {len(failing)} fields:"
        )
        lines = [header]
        for name, fr in self.field_results.items():
            status = "PASS" if fr.passed else "FAIL"
            msg = fr.message if not fr.passed else "OK"
            lines.append(f"  [{status}] {name} ({fr.policy.name}): {msg}")
        return "\n".join(lines)


def _summarize_value(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        preview = ", ".join(repr(x.item() if hasattr(x, "item") else x) for x in flat[:3])
        return f"shape={value.shape}, dtype={value.dtype}, values=[{preview}]"
    if isinstance(value, (list, tuple)):
        preview = ", ".join(repr(x) for x in value[:3])
        return f"len={len(value)}, values=[{preview}]"
    return repr(value)


def compare_exact(actual, reference, *, field_name: str) -> FieldResult:
    """Exact equality comparison. Handles scalar, list, and np.ndarray."""

    if isinstance(actual, np.ndarray) or isinstance(reference, np.ndarray):
        passed = np.array_equal(np.asarray(actual), np.asarray(reference))
    else:
        passed = actual == reference

    message = f"{field_name}: exact match passed" if passed else f"{field_name}: exact match failed"
    return FieldResult(
        policy=ComparisonPolicy.EXACT,
        passed=bool(passed),
        message=message,
        actual_summary=_summarize_value(actual),
        reference_summary=_summarize_value(reference),
        delta=None,
        tolerance=None,
        pct_over_budget=None,
    )


def compare_numeric_close(
    actual,
    reference,
    *,
    field_name: str,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> FieldResult:
    """Tolerance-based numeric comparison. Handles None, shape mismatch, np.allclose."""

    if actual is None and reference is None:
        return FieldResult(
            policy=ComparisonPolicy.NUMERIC_CLOSE,
            passed=True,
            message=f"{field_name}: both values are None",
            actual_summary="None",
            reference_summary="None",
            delta=None,
            tolerance=None,
            pct_over_budget=None,
        )
    if actual is None or reference is None:
        actual_summary = _summarize_value(actual)
        reference_summary = _summarize_value(reference)
        return FieldResult(
            policy=ComparisonPolicy.NUMERIC_CLOSE,
            passed=False,
            message=(f"{field_name}: one value is None (actual={actual_summary}, reference={reference_summary})"),
            actual_summary=actual_summary,
            reference_summary=reference_summary,
            delta=None,
            tolerance=None,
            pct_over_budget=None,
        )

    actual_arr = np.asarray(actual)
    reference_arr = np.asarray(reference)

    if actual_arr.shape != reference_arr.shape:
        return FieldResult(
            policy=ComparisonPolicy.NUMERIC_CLOSE,
            passed=False,
            message=(
                f"{field_name}: shape mismatch "
                f"actual_shape={actual_arr.shape}, reference_shape={reference_arr.shape}"
            ),
            actual_summary=_summarize_value(actual_arr),
            reference_summary=_summarize_value(reference_arr),
            delta=None,
            tolerance=None,
            pct_over_budget=None,
        )

    if not (
        (np.issubdtype(actual_arr.dtype, np.number) or actual_arr.dtype.kind == "b")
        and (np.issubdtype(reference_arr.dtype, np.number) or reference_arr.dtype.kind == "b")
    ):
        return FieldResult(
            policy=ComparisonPolicy.NUMERIC_CLOSE,
            passed=False,
            message=f"{field_name}: non-numeric values cannot be compared numerically",
            actual_summary=_summarize_value(actual_arr),
            reference_summary=_summarize_value(reference_arr),
            delta=None,
            tolerance=None,
            pct_over_budget=None,
        )

    passed = bool(np.allclose(actual_arr, reference_arr, atol=atol, rtol=rtol))
    abs_delta = np.abs(actual_arr - reference_arr)
    max_abs_delta = float(np.max(abs_delta)) if abs_delta.size else 0.0
    denom = np.maximum(np.abs(reference_arr), np.finfo(float).eps)
    max_rel_delta = float(np.max(abs_delta / denom)) if abs_delta.size else 0.0
    pct_over_budget = float(((max_abs_delta - atol) / atol) * 100.0) if atol > 0 and not passed else None

    if passed:
        message = f"{field_name}: values within tolerance"
    else:
        message = (
            f"{field_name}: values differ; actual_summary={_summarize_value(actual_arr)}; "
            f"reference_summary={_summarize_value(reference_arr)}; max_delta={max_abs_delta}; "
            f"max_rel_delta={max_rel_delta}; atol={atol}; rtol={rtol}; pct_over_budget={pct_over_budget}"
        )

    return FieldResult(
        policy=ComparisonPolicy.NUMERIC_CLOSE,
        passed=passed,
        message=message,
        actual_summary=_summarize_value(actual_arr),
        reference_summary=_summarize_value(reference_arr),
        delta=max_abs_delta,
        tolerance=max(atol, rtol),
        pct_over_budget=pct_over_budget,
    )


def compare_labels(
    actual: list[Label],
    reference: list[Label],
    *,
    field_name: str,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> FieldResult:
    """Compare list of Label objects: id and name exactly, confidence with tolerance."""
    if actual is None and reference is None:
        return FieldResult(
            policy=ComparisonPolicy.LABEL_CLOSE,
            passed=True,
            message=f"{field_name}: both values are None",
            actual_summary="None",
            reference_summary="None",
            delta=None,
            tolerance=None,
            pct_over_budget=None,
        )

    if actual is None or reference is None:
        return FieldResult(
            policy=ComparisonPolicy.LABEL_CLOSE,
            passed=False,
            message=f"{field_name}: one value is None",
            actual_summary=_summarize_value(actual),
            reference_summary=_summarize_value(reference),
            delta=None,
            tolerance=None,
            pct_over_budget=None,
        )

    if len(actual) != len(reference):
        return FieldResult(
            policy=ComparisonPolicy.LABEL_CLOSE,
            passed=False,
            message=f"{field_name}: length mismatch (actual={len(actual)}, reference={len(reference)})",
            actual_summary=_summarize_value(actual),
            reference_summary=_summarize_value(reference),
            delta=None,
            tolerance=None,
            pct_over_budget=None,
        )

    mismatches: list[str] = []
    max_conf_delta = 0.0

    for i, (a, r) in enumerate(zip(actual, reference)):
        if a.id != r.id:
            mismatches.append(f"[{i}].id: {a.id} != {r.id}")
        if a.name != r.name:
            mismatches.append(f"[{i}].name: {a.name!r} != {r.name!r}")

        if a.confidence is not None and r.confidence is not None:
            conf_delta = abs(float(a.confidence) - float(r.confidence))
            max_conf_delta = max(max_conf_delta, conf_delta)
            if not np.isclose(a.confidence, r.confidence, atol=atol, rtol=rtol):
                mismatches.append(f"[{i}].confidence: {a.confidence} vs {r.confidence} (delta={conf_delta:.6g})")
        elif a.confidence != r.confidence:
            mismatches.append(f"[{i}].confidence: {a.confidence} vs {r.confidence} (one is None)")

    passed = len(mismatches) == 0
    if passed:
        message = f"{field_name}: {len(actual)} labels match (max_conf_delta={max_conf_delta:.6g})"
    else:
        message = f"{field_name}: {len(mismatches)} mismatches: {'; '.join(mismatches[:5])}"
        if len(mismatches) > 5:
            message += f" ... and {len(mismatches) - 5} more"

    return FieldResult(
        policy=ComparisonPolicy.LABEL_CLOSE,
        passed=passed,
        message=message,
        actual_summary=_summarize_value(actual),
        reference_summary=_summarize_value(reference),
        delta=max_conf_delta if max_conf_delta > 0 else None,
        tolerance=max(atol, rtol),
        pct_over_budget=None,
    )


def _compare_single_fingerprint(  # noqa: C901
    actual_array: np.ndarray,
    reference_fingerprint: dict,
    *,
    field_name: str,
    atol: float,
    rtol: float,
    iou_threshold_thumbnail: float,
    sample_atol: float,
) -> FieldResult:
    """Compare a single array against a single reference fingerprint."""
    actual_fp = compute_fingerprint(actual_array)
    if actual_fp is None or isinstance(actual_fp, list):
        return FieldResult(
            policy=ComparisonPolicy.STAT_FINGERPRINT,
            passed=False,
            message=f"{field_name}: compute_fingerprint returned None for non-None array",
            actual_summary=_summarize_value(actual_array),
            reference_summary="fingerprint(dict)",
            delta=None,
            tolerance=None,
            pct_over_budget=None,
        )

    checks: list[tuple[str, bool, str]] = []

    a_shape = tuple(actual_fp.get("shape") or ())
    r_shape = tuple(reference_fingerprint.get("shape") or ())
    if a_shape != r_shape:
        return FieldResult(
            policy=ComparisonPolicy.STAT_FINGERPRINT,
            passed=False,
            message=(f"{field_name}: shape mismatch actual={a_shape} reference={r_shape}"),
            actual_summary=_summarize_value(actual_array),
            reference_summary=f"shape={r_shape}, dtype={reference_fingerprint.get('dtype')}",
            delta=None,
            tolerance=None,
            pct_over_budget=None,
        )
    checks.append(("shape", True, f"{a_shape}=={r_shape}"))

    a_dtype = actual_fp.get("dtype")
    r_dtype = reference_fingerprint.get("dtype")
    dtype_match = a_dtype == r_dtype
    checks.append(
        ("dtype", True, f"actual={a_dtype} ref={r_dtype}{' (warn:mismatch)' if not dtype_match else ''}"),
    )

    scalar_stats = ("mean", "std", "min", "max", "l2_norm")
    for stat in scalar_stats:
        a_val = actual_fp.get(stat)
        r_val = reference_fingerprint.get(stat)
        if a_val is None and r_val is None:
            checks.append((stat, True, "both None"))
            continue
        if a_val is None or r_val is None:
            checks.append((stat, False, f"one None: actual={a_val} ref={r_val}"))
            continue
        ok = bool(np.isclose(a_val, r_val, atol=atol, rtol=rtol))
        checks.append((stat, ok, f"actual={a_val} ref={r_val} delta={abs(a_val - r_val):.6g}"))

    a_pcts = actual_fp.get("percentiles") or {}
    r_pcts = reference_fingerprint.get("percentiles") or {}
    for pct in ("p1", "p50", "p99"):
        a_val = a_pcts.get(pct)
        r_val = r_pcts.get(pct)
        if a_val is None and r_val is None:
            checks.append((f"pct.{pct}", True, "both None"))
            continue
        if a_val is None or r_val is None:
            checks.append((f"pct.{pct}", False, f"one None: actual={a_val} ref={r_val}"))
            continue
        ok = bool(np.isclose(a_val, r_val, atol=atol, rtol=rtol))
        checks.append((f"pct.{pct}", ok, f"actual={a_val} ref={r_val} delta={abs(a_val - r_val):.6g}"))

    a_hist = np.asarray(actual_fp.get("histogram") or [], dtype=np.float64)
    r_hist = np.asarray(reference_fingerprint.get("histogram") or [], dtype=np.float64)
    if a_hist.shape != r_hist.shape:
        checks.append(("histogram", False, f"bin count mismatch a={a_hist.shape} r={r_hist.shape}"))
    elif a_hist.size == 0:
        checks.append(("histogram", True, "empty"))
    else:
        total = float(r_hist.sum()) if r_hist.sum() > 0 else float(a_hist.sum())
        l1 = float(np.abs(a_hist - r_hist).sum())
        budget = 0.1 * total
        ok = l1 <= budget
        checks.append(("histogram", ok, f"l1={l1:.4g} budget={budget:.4g}(=0.1*{total:.4g})"))

    a_sm = actual_fp.get("spatial_moments")
    r_sm = reference_fingerprint.get("spatial_moments")
    if a_sm is None and r_sm is None:
        checks.append(("spatial_moments", True, "both None"))
    elif a_sm is None or r_sm is None:
        checks.append(("spatial_moments", False, f"one None: actual={a_sm} ref={r_sm}"))
    else:
        if len(a_shape) >= 2:
            h, w = a_shape[-2], a_shape[-1]
            extent = max(h, w)
        else:
            extent = 1
        sm_atol = 0.05 * extent
        a_cx, a_cy = a_sm.get("cx"), a_sm.get("cy")
        r_cx, r_cy = r_sm.get("cx"), r_sm.get("cy")
        dx = abs(a_cx - r_cx)
        dy = abs(a_cy - r_cy)
        ok = dx <= sm_atol and dy <= sm_atol
        checks.append(
            (
                "spatial_moments",
                ok,
                (
                    f"actual=(cx={a_cx},cy={a_cy}) ref=(cx={r_cx},cy={r_cy}) "
                    f"dx={dx:.4g} dy={dy:.4g} atol={sm_atol:.4g}"
                ),
            ),
        )

    a_argmax = actual_fp.get("argmax_index") or []
    r_argmax = reference_fingerprint.get("argmax_index") or []
    if not a_argmax and not r_argmax:
        checks.append(("argmax_index", True, "both empty"))
    elif len(a_argmax) != len(r_argmax):
        checks.append(("argmax_index", False, f"dim mismatch a={a_argmax} r={r_argmax}"))
    else:
        manhattan = sum(abs(int(a) - int(b)) for a, b in zip(a_argmax, r_argmax))
        budget = 0.05 * sum(a_shape) if a_shape else 0.0
        ok = manhattan <= budget
        checks.append(
            ("argmax_index", ok, f"actual={a_argmax} ref={r_argmax} manhattan={manhattan} budget={budget:.4g}"),
        )

    a_thumb = actual_fp.get("thumbnail")
    r_thumb = reference_fingerprint.get("thumbnail")
    if a_thumb is None and r_thumb is None:
        checks.append(("thumbnail", True, "both None"))
    elif a_thumb is None or r_thumb is None:
        checks.append(("thumbnail", False, f"one None: actual_is_none={a_thumb is None}"))
    else:
        a_t = np.asarray(a_thumb, dtype=np.float64)
        r_t = np.asarray(r_thumb, dtype=np.float64)
        if a_t.shape != r_t.shape:
            checks.append(("thumbnail", False, f"shape mismatch a={a_t.shape} r={r_t.shape}"))
        else:
            max_delta = float(np.max(np.abs(a_t - r_t))) if a_t.size else 0.0
            ok = bool(np.allclose(a_t, r_t, atol=0.05))
            checks.append(("thumbnail", ok, f"max_delta={max_delta:.4g} atol=0.05"))

    a_sv = np.asarray(actual_fp.get("sample_values") or [], dtype=np.float64)
    r_sv = np.asarray(reference_fingerprint.get("sample_values") or [], dtype=np.float64)
    if a_sv.shape != r_sv.shape:
        checks.append(("sample_values", False, f"len mismatch a={a_sv.shape} r={r_sv.shape}"))
    elif a_sv.size == 0:
        checks.append(("sample_values", True, "empty"))
    else:
        max_delta = float(np.max(np.abs(a_sv - r_sv)))
        ok = bool(np.allclose(a_sv, r_sv, atol=sample_atol))
        checks.append(("sample_values", ok, f"max_delta={max_delta:.4g} atol={sample_atol}"))

    failing = [name for name, ok, _ in checks if not ok]
    overall_passed = len(failing) == 0

    parts = [f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}" for name, ok, detail in checks]
    summary_line = (
        f"{field_name}: fingerprint OK ({len(checks)} checks)"
        if overall_passed
        else f"{field_name}: fingerprint FAILED ({len(failing)}/{len(checks)} failed: {failing})"
    )
    message = summary_line + "\n  " + "\n  ".join(parts)

    r_dtype = reference_fingerprint.get("dtype")
    return FieldResult(
        policy=ComparisonPolicy.STAT_FINGERPRINT,
        passed=overall_passed,
        message=message,
        actual_summary=_summarize_value(actual_array),
        reference_summary=f"shape={r_shape}, dtype={r_dtype}",
        delta=None,
        tolerance=max(atol, rtol),
        pct_over_budget=None,
    )


def compare_fingerprint(
    actual_array: np.ndarray | list[np.ndarray] | None,
    reference_fingerprint: dict | list[dict] | None,
    *,
    field_name: str,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    iou_threshold_thumbnail: float = 0.85,
    sample_atol: float = 1e-2,
) -> FieldResult:
    """Compare an actual array (or list of arrays) against a precomputed reference fingerprint.

    For list inputs, compares element-wise and aggregates results.
    Reports each sub-stat in the message. Fails if ANY sub-check fails.
    Dtype mismatch is reported as a warning (does not cause failure).
    """
    # Both None → pass
    if actual_array is None and reference_fingerprint is None:
        return FieldResult(
            policy=ComparisonPolicy.STAT_FINGERPRINT,
            passed=True,
            message=f"{field_name}: both actual and reference are None",
            actual_summary="None",
            reference_summary="None",
            delta=None,
            tolerance=None,
            pct_over_budget=None,
        )
    # One None → fail
    if actual_array is None or reference_fingerprint is None:
        return FieldResult(
            policy=ComparisonPolicy.STAT_FINGERPRINT,
            passed=False,
            message=(
                f"{field_name}: one side is None "
                f"(actual_is_none={actual_array is None}, reference_is_none={reference_fingerprint is None})"
            ),
            actual_summary=_summarize_value(actual_array),
            reference_summary="None" if reference_fingerprint is None else "fingerprint(dict)",
            delta=None,
            tolerance=None,
            pct_over_budget=None,
        )

    actual_is_list = isinstance(actual_array, list)
    ref_is_list = isinstance(reference_fingerprint, list)

    if actual_is_list != ref_is_list:
        return FieldResult(
            policy=ComparisonPolicy.STAT_FINGERPRINT,
            passed=False,
            message=(f"{field_name}: type mismatch (actual_is_list={actual_is_list}, reference_is_list={ref_is_list})"),
            actual_summary=_summarize_value(actual_array),
            reference_summary="fingerprint(list)" if ref_is_list else "fingerprint(dict)",
            delta=None,
            tolerance=None,
            pct_over_budget=None,
        )

    if actual_is_list:
        if len(actual_array) != len(reference_fingerprint):
            return FieldResult(
                policy=ComparisonPolicy.STAT_FINGERPRINT,
                passed=False,
                message=(
                    f"{field_name}: list length mismatch "
                    f"(actual={len(actual_array)}, reference={len(reference_fingerprint)})"
                ),
                actual_summary=f"list[{len(actual_array)}]",
                reference_summary=f"list[{len(reference_fingerprint)}]",
                delta=None,
                tolerance=None,
                pct_over_budget=None,
            )

        sub_results: list[FieldResult] = []
        for i, (arr, ref_fp) in enumerate(zip(actual_array, reference_fingerprint)):
            sub_result = _compare_single_fingerprint(
                arr,
                ref_fp,
                field_name=f"{field_name}[{i}]",
                atol=atol,
                rtol=rtol,
                iou_threshold_thumbnail=iou_threshold_thumbnail,
                sample_atol=sample_atol,
            )
            sub_results.append(sub_result)

        all_passed = all(r.passed for r in sub_results)
        failing_indices = [i for i, r in enumerate(sub_results) if not r.passed]
        if all_passed:
            message = f"{field_name}: all {len(sub_results)} fingerprints OK"
        else:
            message = (
                f"{field_name}: {len(failing_indices)}/{len(sub_results)} fingerprints FAILED "
                f"(indices: {failing_indices})"
            )
            for i in failing_indices:
                message += f"\n  [{i}]: {sub_results[i].message}"

        return FieldResult(
            policy=ComparisonPolicy.STAT_FINGERPRINT,
            passed=all_passed,
            message=message,
            actual_summary=f"list[{len(actual_array)}]",
            reference_summary=f"list[{len(reference_fingerprint)}]",
            delta=None,
            tolerance=max(atol, rtol),
            pct_over_budget=None,
        )

    # Type narrowing: after the if actual_is_list block, we know these are not lists
    assert isinstance(actual_array, np.ndarray), f"Expected np.ndarray, got {type(actual_array)}"
    assert isinstance(reference_fingerprint, dict), f"Expected dict, got {type(reference_fingerprint)}"

    return _compare_single_fingerprint(
        actual_array,
        reference_fingerprint,
        field_name=field_name,
        atol=atol,
        rtol=rtol,
        iou_threshold_thumbnail=iou_threshold_thumbnail,
        sample_atol=sample_atol,
    )


def compare_class_map(
    actual: np.ndarray,
    ref_path: Path,
    *,
    iou_threshold: float = 0.95,
) -> FieldResult:
    """Load reference class map (PNG I;16), compute per-class IoU, mean IoU vs threshold."""
    ref = load_class_map(ref_path)
    actual_arr = np.asarray(actual)

    if actual_arr.shape != ref.shape:
        return FieldResult(
            policy=ComparisonPolicy.MASK_IOU,
            passed=False,
            message=(f"class_map: shape mismatch actual={actual_arr.shape} reference={ref.shape}"),
            actual_summary=_summarize_value(actual_arr),
            reference_summary=_summarize_value(ref),
            delta=None,
            tolerance=iou_threshold,
            pct_over_budget=None,
        )

    actual_u16 = actual_arr.astype(np.uint16)
    mean_iou = compute_class_map_iou(actual_u16, ref)

    # Per-class breakdown
    classes = np.union1d(np.unique(actual_u16), np.unique(ref))
    per_class: list[str] = []
    for c in classes:
        p = actual_u16 == c
        r = ref == c
        if not p.any() and not r.any():
            continue
        inter = int(np.logical_and(p, r).sum())
        union = int(np.logical_or(p, r).sum())
        iou = (inter / union) if union else 1.0
        per_class.append(f"class {int(c)}: IoU={iou:.4f}")

    passed = mean_iou >= iou_threshold
    breakdown = "; ".join(per_class) if per_class else "no classes"
    if passed:
        message = f"class_map: mean IoU={mean_iou:.4f} >= {iou_threshold} ({breakdown})"
    else:
        message = f"class_map: mean IoU={mean_iou:.4f} < {iou_threshold} (per-class: {breakdown})"

    return FieldResult(
        policy=ComparisonPolicy.MASK_IOU,
        passed=passed,
        message=message,
        actual_summary=_summarize_value(actual_arr),
        reference_summary=_summarize_value(ref),
        delta=float(1.0 - mean_iou),
        tolerance=iou_threshold,
        pct_over_budget=None,
    )


def compare_binary_mask(
    actual: np.ndarray,
    ref_path: Path,
    *,
    iou_threshold: float = 0.95,
) -> FieldResult:
    """Binarize actual if needed, IoU against reference binary mask."""
    ref = load_binary_mask(ref_path)
    actual_arr = np.asarray(actual)

    actual_bin = actual_arr >= 0.5 if actual_arr.dtype != bool else actual_arr

    if actual_bin.shape != ref.shape:
        return FieldResult(
            policy=ComparisonPolicy.MASK_IOU,
            passed=False,
            message=(f"binary_mask: shape mismatch actual={actual_bin.shape} reference={ref.shape}"),
            actual_summary=_summarize_value(actual_arr),
            reference_summary=_summarize_value(ref),
            delta=None,
            tolerance=iou_threshold,
            pct_over_budget=None,
        )

    iou = compute_mean_iou(actual_bin, ref)
    passed = iou >= iou_threshold
    message = (
        f"binary_mask: IoU={iou:.4f} >= {iou_threshold}" if passed else f"binary_mask: IoU={iou:.4f} < {iou_threshold}"
    )
    return FieldResult(
        policy=ComparisonPolicy.MASK_IOU,
        passed=passed,
        message=message,
        actual_summary=_summarize_value(actual_arr),
        reference_summary=_summarize_value(ref),
        delta=float(1.0 - iou),
        tolerance=iou_threshold,
        pct_over_budget=None,
    )


def compare_instance_masks(
    actual_masks: np.ndarray,
    ref_masks_path: Path,
    actual_bboxes: np.ndarray | None,
    ref_bboxes: np.ndarray | None,
    *,
    iou_threshold: float = 0.95,
    match_iou: float = 0.5,
) -> FieldResult:
    """Load stacked reference, run Hungarian matching, per matched pair check IoU >= threshold.

    Count mismatch is hard failure.
    Unmatched on either side is hard failure.
    """
    ref_masks = load_instance_masks(ref_masks_path)
    actual_arr = np.asarray(actual_masks)

    n_actual = int(actual_arr.shape[0]) if actual_arr.ndim >= 1 else 0
    n_ref = int(ref_masks.shape[0]) if ref_masks.ndim >= 1 else 0

    if n_actual != n_ref:
        return FieldResult(
            policy=ComparisonPolicy.MASK_IOU,
            passed=False,
            message=f"instance_masks: count mismatch: actual {n_actual} vs reference {n_ref}",
            actual_summary=_summarize_value(actual_arr),
            reference_summary=_summarize_value(ref_masks),
            delta=None,
            tolerance=iou_threshold,
            pct_over_budget=None,
        )

    # Binarize actual if float
    actual_bin = actual_arr >= 0.5 if actual_arr.dtype != bool else actual_arr

    # Choose matching strategy
    use_bbox = actual_bin.size == 0 and actual_bboxes is not None and ref_bboxes is not None

    if use_bbox:
        matched, unmatched_pred, unmatched_ref = match_by_bbox_iou(
            np.asarray(actual_bboxes),
            np.asarray(ref_bboxes),
            iou_threshold=match_iou,
        )
    else:
        if actual_bin.shape[1:] != ref_masks.shape[1:]:
            return FieldResult(
                policy=ComparisonPolicy.MASK_IOU,
                passed=False,
                message=(
                    f"instance_masks: spatial shape mismatch actual={actual_bin.shape} reference={ref_masks.shape}"
                ),
                actual_summary=_summarize_value(actual_arr),
                reference_summary=_summarize_value(ref_masks),
                delta=None,
                tolerance=iou_threshold,
                pct_over_budget=None,
            )
        matched, unmatched_pred, unmatched_ref = match_by_mask_iou(
            actual_bin,
            ref_masks,
            iou_threshold=match_iou,
        )

    if unmatched_pred or unmatched_ref:
        return FieldResult(
            policy=ComparisonPolicy.MASK_IOU,
            passed=False,
            message=(
                f"instance_masks: unmatched instances "
                f"actual_unmatched={unmatched_pred} reference_unmatched={unmatched_ref} "
                f"(match_iou={match_iou})"
            ),
            actual_summary=_summarize_value(actual_arr),
            reference_summary=_summarize_value(ref_masks),
            delta=None,
            tolerance=iou_threshold,
            pct_over_budget=None,
        )

    # Compute per-pair IoU
    per_pair: list[str] = []
    failing_pairs: list[str] = []
    min_iou = 1.0
    for a_idx, r_idx in matched:
        pair_iou = compute_mean_iou(actual_bin[a_idx], ref_masks[r_idx])
        per_pair.append(f"(a{a_idx}->r{r_idx}: IoU={pair_iou:.4f})")
        min_iou = min(min_iou, pair_iou)
        if pair_iou < iou_threshold:
            failing_pairs.append(f"a{a_idx}->r{r_idx}: IoU={pair_iou:.4f}")

    passed = len(failing_pairs) == 0
    breakdown = " ".join(per_pair) if per_pair else "no pairs"
    if passed:
        message = (
            f"instance_masks: {len(matched)} matched pairs all IoU>={iou_threshold} "
            f"min_IoU={min_iou:.4f} pairs: {breakdown}"
        )
    else:
        message = (
            f"instance_masks: {len(failing_pairs)}/{len(matched)} pairs below "
            f"threshold {iou_threshold} failing: {failing_pairs} all_pairs: {breakdown}"
        )

    return FieldResult(
        policy=ComparisonPolicy.MASK_IOU,
        passed=passed,
        message=message,
        actual_summary=_summarize_value(actual_arr),
        reference_summary=_summarize_value(ref_masks),
        delta=float(1.0 - min_iou),
        tolerance=iou_threshold,
        pct_over_budget=None,
    )
