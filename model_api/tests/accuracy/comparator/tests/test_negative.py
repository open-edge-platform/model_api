"""Negative-test suite for the comparator: failures must be caught."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from model_api.models.result.anomaly import AnomalyResult

from tests.accuracy.comparator import (
    assert_result_matches_reference,
    generate_reference,
)
from tests.accuracy.comparator.policies import (
    compare_binary_mask,
    compare_class_map,
    compare_exact,
    compare_fingerprint,
    compare_instance_masks,
    compare_numeric_close,
)
from tests.accuracy.comparator.storage import (
    save_binary_mask,
    save_class_map,
    save_instance_masks,
)

if TYPE_CHECKING:
    from pathlib import Path


def _anomaly(
    *,
    pred_score: float = 0.1,
    pred_label: str = "Normal",
    anomaly_map: np.ndarray | None = None,
    pred_mask: np.ndarray | None = None,
) -> AnomalyResult:
    if anomaly_map is None:
        anomaly_map = np.zeros((4, 4), dtype=np.float32)
    if pred_mask is None:
        pred_mask = np.zeros((4, 4), dtype=bool)
    return AnomalyResult(
        anomaly_map=anomaly_map,
        pred_mask=pred_mask,
        pred_label=pred_label,
        pred_score=pred_score,
    )


# 1. Numeric field out of tolerance.
def test_numeric_close_out_of_tolerance_reports_delta() -> None:
    result = compare_numeric_close(
        1.0,
        5.0,
        field_name="x",
        atol=1e-3,
        rtol=1e-3,
    )
    assert not result.passed
    assert "delta=" in result.message


# 2. Wrong dtype reported by exact comparator (int vs float values differ).
def test_compare_exact_int_vs_float_with_value_mismatch() -> None:
    actual = np.array([1, 2, 3], dtype=np.int32)
    reference = np.array([1.0, 2.0, 3.5], dtype=np.float64)
    result = compare_exact(actual, reference, field_name="labels")
    assert not result.passed
    assert "exact match failed" in result.message


# 3. Wrong shape (fingerprint).
def test_fingerprint_shape_mismatch_fails() -> None:
    actual = np.zeros((10, 10), dtype=np.float32)
    from tests.accuracy.comparator.fingerprint import compute_fingerprint

    ref_fp = compute_fingerprint(np.zeros((20, 20), dtype=np.float32))
    result = compare_fingerprint(actual, ref_fp, field_name="anomaly_map")
    assert not result.passed
    assert "shape mismatch" in result.message


# 4. Mask IoU below threshold (shifted mask).
def test_binary_mask_shifted_below_threshold(tmp_path: Path) -> None:
    ref_mask = np.zeros((128, 128), dtype=bool)
    ref_mask[10:50, 10:50] = True
    p = tmp_path / "ref.png"
    save_binary_mask(p, ref_mask)

    shifted = np.zeros((128, 128), dtype=bool)
    shifted[80:120, 80:120] = True

    result = compare_binary_mask(shifted, p, iou_threshold=0.9)
    assert not result.passed
    assert "<" in result.message


# 5. Instance count mismatch.
def test_instance_masks_count_mismatch(tmp_path: Path) -> None:
    masks = np.zeros((3, 16, 16), dtype=np.float32)
    masks[0, 0:5, 0:5] = 1.0
    masks[1, 5:10, 5:10] = 1.0
    masks[2, 10:15, 10:15] = 1.0
    p = tmp_path / "ref.npz"
    save_instance_masks(p, masks)

    fewer = masks[:2].copy()
    result = compare_instance_masks(
        fewer,
        p,
        actual_bboxes=None,
        ref_bboxes=None,
    )
    assert not result.passed
    assert "count mismatch" in result.message.lower()


# 6. Class map mismatch (different class IDs).
def test_class_map_class_id_mismatch(tmp_path: Path) -> None:
    ref = np.zeros((16, 16), dtype=np.uint16)
    ref[:8, :] = 1
    ref[8:, :] = 2
    p = tmp_path / "ref.png"
    save_class_map(p, ref)

    actual = np.zeros((16, 16), dtype=np.uint16)
    actual[:8, :] = 1
    actual[8:, :] = 5

    result = compare_class_map(actual, p, iou_threshold=0.95)
    assert not result.passed
    assert "IoU=" in result.message


# 7. Missing reference directory.
def test_assert_missing_reference_raises(tmp_path: Path) -> None:
    result = _anomaly()
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        assert_result_matches_reference(result, missing_dir)


# 8. Overwrite protection.
def test_generate_reference_overwrite_protection(tmp_path: Path) -> None:
    result = _anomaly()
    generate_reference(result, tmp_path, test_id="a")
    with pytest.raises(FileExistsError):
        generate_reference(result, tmp_path, test_id="a")


# 9. Fingerprint mean drift triggers AssertionError via wrapper.
def test_anomaly_map_mean_drift_raises(tmp_path: Path) -> None:
    rng = np.random.RandomState(0)
    ref_map = rng.rand(32, 32).astype(np.float32) * 0.1
    ref_result = _anomaly(anomaly_map=ref_map)
    generate_reference(ref_result, tmp_path, test_id="drift")

    drifted_map = ref_map + 0.5
    bad_result = _anomaly(anomaly_map=drifted_map.astype(np.float32))

    with pytest.raises(AssertionError) as exc:
        assert_result_matches_reference(bad_result, tmp_path)
    assert "anomaly_map" in str(exc.value)


# 10. Fingerprint shape mismatch via wrapper raises AssertionError.
def test_anomaly_map_shape_mismatch_raises(tmp_path: Path) -> None:
    ref_result = _anomaly(
        anomaly_map=np.zeros((16, 16), dtype=np.float32),
        pred_mask=np.zeros((16, 16), dtype=bool),
    )
    generate_reference(ref_result, tmp_path, test_id="shape")

    bad_result = _anomaly(
        anomaly_map=np.zeros((8, 8), dtype=np.float32),
        pred_mask=np.zeros((8, 8), dtype=bool),
    )
    with pytest.raises(AssertionError) as exc:
        assert_result_matches_reference(bad_result, tmp_path)
    assert "shape mismatch" in str(exc.value)


# 11. Pred_score numeric drift through wrapper raises AssertionError with delta info.
def test_pred_score_drift_raises_with_delta(tmp_path: Path) -> None:
    ref_result = _anomaly(pred_score=0.10)
    generate_reference(ref_result, tmp_path, test_id="score")

    bad_result = _anomaly(pred_score=0.95)
    with pytest.raises(AssertionError) as exc:
        assert_result_matches_reference(bad_result, tmp_path)
    msg = str(exc.value)
    assert "pred_score" in msg
    assert "delta=" in msg


# 12. Exact label mismatch through wrapper raises AssertionError.
def test_pred_label_mismatch_raises(tmp_path: Path) -> None:
    ref_result = _anomaly(pred_label="Normal")
    generate_reference(ref_result, tmp_path, test_id="label")

    bad_result = _anomaly(pred_label="Anomalous")
    with pytest.raises(AssertionError) as exc:
        assert_result_matches_reference(bad_result, tmp_path)
    assert "pred_label" in str(exc.value)


# 13. Corrupted reference: result.json removed.
def test_corrupted_reference_missing_result_json(tmp_path: Path) -> None:
    ref_result = _anomaly()
    generate_reference(ref_result, tmp_path, test_id="corrupt")

    (tmp_path / "result.json").unlink()

    with pytest.raises(FileNotFoundError):
        assert_result_matches_reference(ref_result, tmp_path)


# 14. Numeric_close non-numeric values rejected.
def test_numeric_close_non_numeric_values() -> None:
    result = compare_numeric_close(
        np.array(["a", "b"]),
        np.array(["a", "b"]),
        field_name="x",
    )
    assert not result.passed
    assert "non-numeric" in result.message


# 15. Numeric_close shape mismatch direct.
def test_numeric_close_shape_mismatch() -> None:
    result = compare_numeric_close(
        np.zeros((3,)),
        np.zeros((4,)),
        field_name="x",
    )
    assert not result.passed
    assert "shape mismatch" in result.message
