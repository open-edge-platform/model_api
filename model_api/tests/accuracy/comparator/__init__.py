"""Public comparator API for accuracy test comparisons."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from model_api.models.result import Label

from .dispatch import dispatch
from .fingerprint import compute_fingerprint
from .policies import (
    ComparisonPolicy,
    ComparisonReport,
    FieldResult,
    compare_binary_mask,
    compare_class_map,
    compare_exact,
    compare_fingerprint,
    compare_instance_masks,
    compare_numeric_close,
)
from .storage import (
    _save_mask,
    build_generated_by,
    load_reference,
    save_reference,
)

__all__ = [
    "ComparisonPolicy",
    "ComparisonReport",
    "FieldResult",
    "assert_result_matches_reference",
    "generate_reference",
]


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__class__": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "values": value.tolist(),
        }
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, Label):
        return {
            "__class__": "Label",
            "id": _to_json_safe(value.id),
            "name": value.name,
            "confidence": _to_json_safe(value.confidence),
        }
    return value


def _from_json_safe(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_json_safe(v) for v in value]

    if isinstance(value, dict):
        match value.get("__class__"):
            case "ndarray":
                arr = np.array(value["values"], dtype=value["dtype"])
                return arr.reshape(value["shape"])
            case "Label":
                return Label(id=value["id"], name=value["name"], confidence=value["confidence"])
    return value


def _coerce_mask_for_save(arr: np.ndarray | list) -> np.ndarray:
    if isinstance(arr, list):
        arr = arr[0] if len(arr) == 1 else np.stack(arr)
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    if arr.ndim == 3 and arr.dtype != np.bool_:
        return arr >= 0.5
    return arr


def generate_reference(
    result,
    reference_dir,
    *,
    test_id: str,
    policy_overrides: dict | None = None,
    overwrite: bool = False,
) -> None:
    """Generate a reference artifact for ``result`` at ``reference_dir``."""
    reference_dir = Path(reference_dir)
    specs = dispatch(result, policy_overrides=policy_overrides)

    result_json: dict[str, Any] = {}
    mask_files: dict[str, str] = {}
    masks_to_save: dict[str, np.ndarray] = {}

    for field_name, spec in specs.items():
        if spec.policy is ComparisonPolicy.STAT_FINGERPRINT:
            result_json[field_name] = compute_fingerprint(spec.value)
        elif spec.policy is ComparisonPolicy.MASK_IOU:
            if spec.value is None:
                result_json[field_name] = None
                continue
            mask_arr = _coerce_mask_for_save(spec.value)
            masks_to_save[field_name] = mask_arr
        else:
            result_json[field_name] = _to_json_safe(spec.value)

    reference_dir.mkdir(parents=True, exist_ok=True)
    if masks_to_save:
        masks_dir = reference_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        for field_name, arr in masks_to_save.items():
            saved_path = _save_mask(masks_dir / field_name, arr)
            mask_files[field_name] = saved_path.name

    if mask_files:
        result_json["mask_files"] = mask_files

    metadata = {"test_id": test_id, "policy_overrides": policy_overrides or {}}
    save_reference(
        reference_dir,
        result_json=result_json,
        masks=None,
        metadata=metadata,
        generated_by=build_generated_by(),
        overwrite=overwrite,
    )


def _compare_mask_field(
    field_name: str,
    spec_value: np.ndarray | list,
    mask_path: Path,
    spec_kwargs: dict,
) -> FieldResult:
    if isinstance(spec_value, list):
        actual = np.asarray(spec_value[0]) if len(spec_value) == 1 else np.stack(spec_value)
    else:
        actual = np.asarray(spec_value)
    if actual.ndim == 4 and actual.shape[0] == 1:
        actual = actual.squeeze(0)
    if mask_path.suffix == ".npz":
        kw = {k: v for k, v in spec_kwargs.items() if k in {"iou_threshold", "match_iou"}}
        return compare_instance_masks(
            actual_masks=actual,
            ref_masks_path=mask_path,
            actual_bboxes=None,
            ref_bboxes=None,
            **kw,
        )
    kw = {k: v for k, v in spec_kwargs.items() if k == "iou_threshold"}
    is_class_map = actual.ndim == 2 and (
        actual.dtype == np.uint16 or (np.issubdtype(actual.dtype, np.integer) and actual.max(initial=0) > 1)
    )
    if is_class_map:
        return compare_class_map(actual, mask_path, **kw)
    return compare_binary_mask(actual, mask_path, **kw)


def assert_result_matches_reference(
    result,
    reference_dir,
    *,
    strict: bool = False,
) -> None:
    """Assert that ``result`` matches the stored reference at ``reference_dir``."""
    reference_dir = Path(reference_dir)
    bundle = load_reference(reference_dir)
    policy_overrides = bundle.metadata.get("policy_overrides") or {}
    specs = dispatch(result, policy_overrides=policy_overrides)

    field_results: dict[str, FieldResult] = {}
    for field_name, spec in specs.items():
        if spec.policy is ComparisonPolicy.EXACT:
            ref = _from_json_safe(bundle.result_json.get(field_name))
            field_results[field_name] = compare_exact(spec.value, ref, field_name=field_name)
        elif spec.policy is ComparisonPolicy.NUMERIC_CLOSE:
            ref_raw = bundle.result_json.get(field_name)
            ref = _from_json_safe(ref_raw) if ref_raw is not None else None
            kw = {k: v for k, v in spec.kwargs.items() if k in {"atol", "rtol"}}
            field_results[field_name] = compare_numeric_close(
                spec.value,
                ref,
                field_name=field_name,
                **kw,
            )
        elif spec.policy is ComparisonPolicy.STAT_FINGERPRINT:
            ref_fp = bundle.result_json.get(field_name)
            kw = {
                k: v for k, v in spec.kwargs.items() if k in {"atol", "rtol", "iou_threshold_thumbnail", "sample_atol"}
            }
            field_results[field_name] = compare_fingerprint(
                spec.value,
                ref_fp,
                field_name=field_name,
                **kw,
            )
        elif spec.policy is ComparisonPolicy.MASK_IOU:
            if spec.value is None and bundle.result_json.get(field_name, "missing") is None:
                field_results[field_name] = FieldResult(
                    policy=ComparisonPolicy.MASK_IOU,
                    passed=True,
                    message=f"{field_name}: both None",
                    actual_summary="None",
                    reference_summary="None",
                    delta=None,
                    tolerance=None,
                    pct_over_budget=None,
                )
                continue
            mask_path = bundle.mask_paths.get(field_name)
            if mask_path is None:
                field_results[field_name] = FieldResult(
                    policy=ComparisonPolicy.MASK_IOU,
                    passed=False,
                    message=f"{field_name}: missing reference mask file",
                    actual_summary="ndarray",
                    reference_summary="missing",
                    delta=None,
                    tolerance=None,
                    pct_over_budget=None,
                )
                continue
            field_results[field_name] = _compare_mask_field(
                field_name,
                spec.value,
                mask_path,
                spec.kwargs,
            )

    passed = all(fr.passed for fr in field_results.values())
    report = ComparisonReport(passed=passed, field_results=field_results)
    if not passed:
        raise AssertionError(report.pretty())
