"""Round-trip and failure-message tests for the public comparator API."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from model_api.models.result.anomaly import AnomalyResult
from model_api.models.result.detection import DetectionResult

from tests.accuracy.comparator import (
    assert_result_matches_reference,
    generate_reference,
)


def _make_anomaly_result(seed: int = 0, pred_score: float = 0.85) -> AnomalyResult:
    rng = np.random.RandomState(seed)
    return AnomalyResult(
        anomaly_map=rng.rand(32, 32).astype(np.float32),
        pred_mask=(rng.rand(32, 32) > 0.5),
        pred_label="ok",
        pred_score=pred_score,
    )


def test_round_trip_anomaly_result(tmp_path):
    result = _make_anomaly_result()
    generate_reference(result, tmp_path, test_id="anomaly-1")
    assert_result_matches_reference(result, tmp_path)


def test_round_trip_detection_result(tmp_path):
    result = DetectionResult(
        bboxes=np.zeros((0, 4), dtype=np.float32),
        labels=np.zeros((0,), dtype=np.int32),
    )
    generate_reference(result, tmp_path, test_id="detection-empty")
    assert_result_matches_reference(result, tmp_path)


def test_assertion_failure_reports_field_name(tmp_path):
    ref_result = _make_anomaly_result(seed=0, pred_score=0.85)
    bad_result = AnomalyResult(
        anomaly_map=ref_result.anomaly_map,
        pred_mask=ref_result.pred_mask,
        pred_label=ref_result.pred_label,
        pred_score=0.10,
    )

    generate_reference(ref_result, tmp_path, test_id="anomaly-fail")

    with pytest.raises(AssertionError) as exc_info:
        assert_result_matches_reference(bad_result, tmp_path)

    msg = str(exc_info.value)
    assert "pred_score" in msg
    assert "FAIL" in msg


def test_generate_overwrite_protection(tmp_path):
    result = _make_anomaly_result()
    generate_reference(result, tmp_path, test_id="overwrite-test")
    with pytest.raises(FileExistsError):
        generate_reference(result, tmp_path, test_id="overwrite-test")
    generate_reference(result, tmp_path, test_id="overwrite-test", overwrite=True)


def test_round_trip_4d_mask_with_batch_dim(tmp_path):
    from model_api.models.result.visual_prompting import VisualPromptingResult

    rng = np.random.RandomState(42)
    hard_preds_4d = (rng.rand(1, 4, 32, 48) > 0.5).astype(bool)

    result = VisualPromptingResult(
        hard_predictions=[hard_preds_4d],
        upscaled_masks=[rng.rand(32, 48).astype(np.float32)],
        processed_mask=[(rng.rand(32, 48) > 0.5).astype(bool)],
        low_res_masks=[rng.rand(8, 12).astype(np.float32)],
        iou_predictions=[np.array([0.9, 0.85, 0.8, 0.75], dtype=np.float32)],
        scores=[np.array([0.9], dtype=np.float32)],
        labels=[np.array([1], dtype=np.int32)],
        best_iou=[0.9],
    )

    generate_reference(result, tmp_path, test_id="vp-4d-mask")
    assert_result_matches_reference(result, tmp_path)
