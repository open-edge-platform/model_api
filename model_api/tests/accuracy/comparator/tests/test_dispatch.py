"""Tests for per-result comparator dispatch."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from tests.accuracy.comparator.dispatch import dispatch
from tests.accuracy.comparator.policies import ComparisonPolicy

from model_api.models.result.anomaly import AnomalyResult
from model_api.models.result.base import Result
from model_api.models.result.classification import ClassificationResult, Label
from model_api.models.result.detection import DetectionResult
from model_api.models.result.keypoint import DetectedKeypoints
from model_api.models.result.segmentation import (
    ImageResultWithSoftPrediction,
    InstanceSegmentationResult,
    RotatedSegmentationResult,
)
from model_api.models.result.visual_prompting import (
    PredictedMask,
    VisualPromptingResult,
    ZSLVisualPromptingResult,
)


def _assert_policies(specs, expected):
    assert set(specs) == set(expected)
    for field_name, policy in expected.items():
        assert specs[field_name].policy is policy


def test_classification_result_dispatch():
    result = ClassificationResult(
        top_labels=[Label(id=1, name="cat", confidence=0.9)],
        saliency_map=np.zeros((1, 2, 2)),
        feature_vector=np.zeros((4,)),
        raw_scores=np.zeros((3,)),
    )

    specs = dispatch(result)

    _assert_policies(
        specs,
        {
            "top_labels": ComparisonPolicy.EXACT,
            "raw_scores": ComparisonPolicy.NUMERIC_CLOSE,
            "saliency_map": ComparisonPolicy.STAT_FINGERPRINT,
            "feature_vector": ComparisonPolicy.STAT_FINGERPRINT,
        },
    )
    assert specs["raw_scores"].kwargs == {"atol": 1e-2, "rtol": 1e-2}


def test_anomaly_result_dispatch():
    result = AnomalyResult(
        anomaly_map=np.zeros((4, 4)),
        pred_label="ok",
        pred_mask=np.zeros((4, 4), dtype=bool),
        pred_score=0.1,
    )

    specs = dispatch(result)

    _assert_policies(
        specs,
        {
            "pred_label": ComparisonPolicy.EXACT,
            "pred_score": ComparisonPolicy.NUMERIC_CLOSE,
            "pred_mask": ComparisonPolicy.MASK_IOU,
            "anomaly_map": ComparisonPolicy.STAT_FINGERPRINT,
        },
    )


def test_detection_result_dispatch():
    result = DetectionResult(
        bboxes=np.zeros((0, 4)),
        labels=np.zeros((0,), dtype=int),
    )

    specs = dispatch(result)

    _assert_policies(
        specs,
        {
            "labels": ComparisonPolicy.EXACT,
            "label_names": ComparisonPolicy.EXACT,
            "bboxes": ComparisonPolicy.NUMERIC_CLOSE,
            "scores": ComparisonPolicy.NUMERIC_CLOSE,
            "saliency_map": ComparisonPolicy.STAT_FINGERPRINT,
            "feature_vector": ComparisonPolicy.STAT_FINGERPRINT,
        },
    )
    assert specs["bboxes"].kwargs["atol"] == 2.0
    assert specs["labels"].kwargs["instance_matched"] is True


def test_instance_segmentation_result_dispatch():
    result = InstanceSegmentationResult(
        bboxes=np.zeros((0, 4)),
        labels=np.zeros((0,), dtype=int),
        masks=np.zeros((0, 4, 4), dtype=bool),
    )

    specs = dispatch(result)

    _assert_policies(
        specs,
        {
            "labels": ComparisonPolicy.EXACT,
            "label_names": ComparisonPolicy.EXACT,
            "bboxes": ComparisonPolicy.NUMERIC_CLOSE,
            "scores": ComparisonPolicy.NUMERIC_CLOSE,
            "saliency_map": ComparisonPolicy.STAT_FINGERPRINT,
            "feature_vector": ComparisonPolicy.STAT_FINGERPRINT,
            "masks": ComparisonPolicy.MASK_IOU,
        },
    )
    assert specs["masks"].kwargs["instance_matched"] is True


def test_rotated_segmentation_result_dispatch():
    result = RotatedSegmentationResult(
        bboxes=np.zeros((0, 4)),
        labels=np.zeros((0,), dtype=int),
        masks=np.zeros((0, 4, 4), dtype=bool),
        rotated_rects=[],
    )

    specs = dispatch(result)

    _assert_policies(
        specs,
        {
            "labels": ComparisonPolicy.EXACT,
            "label_names": ComparisonPolicy.EXACT,
            "bboxes": ComparisonPolicy.NUMERIC_CLOSE,
            "scores": ComparisonPolicy.NUMERIC_CLOSE,
            "saliency_map": ComparisonPolicy.STAT_FINGERPRINT,
            "feature_vector": ComparisonPolicy.STAT_FINGERPRINT,
            "masks": ComparisonPolicy.MASK_IOU,
            "rotated_rects": ComparisonPolicy.NUMERIC_CLOSE,
        },
    )
    assert specs["rotated_rects"].kwargs["use_rotated_iou"] is True


def test_image_result_with_soft_prediction_dispatch():
    result = ImageResultWithSoftPrediction(
        resultImage=np.zeros((4, 4), dtype="uint8"),
        soft_prediction=np.zeros((4, 4, 2)),
        saliency_map=np.zeros((4, 4)),
        feature_vector=np.zeros((8,)),
    )

    specs = dispatch(result)

    _assert_policies(
        specs,
        {
            "resultImage": ComparisonPolicy.MASK_IOU,
            "soft_prediction": ComparisonPolicy.STAT_FINGERPRINT,
            "saliency_map": ComparisonPolicy.STAT_FINGERPRINT,
            "feature_vector": ComparisonPolicy.STAT_FINGERPRINT,
        },
    )


def test_detected_keypoints_dispatch():
    result = DetectedKeypoints(
        keypoints=np.zeros((0, 2)),
        scores=np.zeros((0,)),
    )

    specs = dispatch(result)

    _assert_policies(
        specs,
        {
            "keypoints": ComparisonPolicy.NUMERIC_CLOSE,
            "scores": ComparisonPolicy.NUMERIC_CLOSE,
        },
    )


def test_visual_prompting_result_dispatch():
    result = VisualPromptingResult(
        upscaled_masks=[np.zeros((4, 4))],
        processed_mask=[np.zeros((4, 4), dtype=bool)],
        low_res_masks=[np.zeros((2, 2))],
        iou_predictions=[np.zeros((1,))],
        scores=[np.zeros((1,))],
        labels=[np.zeros((1,), dtype=int)],
        hard_predictions=[np.zeros((4, 4), dtype=bool)],
        best_iou=[0.0],
    )

    specs = dispatch(result)

    _assert_policies(
        specs,
        {
            "best_iou": ComparisonPolicy.NUMERIC_CLOSE,
            "iou_predictions": ComparisonPolicy.NUMERIC_CLOSE,
            "scores": ComparisonPolicy.NUMERIC_CLOSE,
            "labels": ComparisonPolicy.EXACT,
            "upscaled_masks": ComparisonPolicy.STAT_FINGERPRINT,
            "processed_masks": ComparisonPolicy.MASK_IOU,
            "low_res_masks": ComparisonPolicy.STAT_FINGERPRINT,
            "hard_predictions": ComparisonPolicy.MASK_IOU,
        },
    )


def test_zsl_visual_prompting_result_dispatch():
    result = ZSLVisualPromptingResult(
        data={
            3: PredictedMask(
                mask=[np.zeros((4, 4), dtype=bool)],
                points=np.zeros((1, 2)),
                scores=np.zeros((1,)),
            ),
        },
    )

    specs = dispatch(result)

    _assert_policies(
        specs,
        {
            "3.mask": ComparisonPolicy.MASK_IOU,
            "3.points": ComparisonPolicy.NUMERIC_CLOSE,
            "3.scores": ComparisonPolicy.NUMERIC_CLOSE,
        },
    )


def test_policy_overrides_merge_into_matching_field():
    result = DetectionResult(
        bboxes=np.zeros((0, 4)),
        labels=np.zeros((0,), dtype=int),
    )

    specs = dispatch(
        result,
        policy_overrides={"bboxes": {"atol": 3.0, "rtol": 0.0}, "missing": {"atol": 9.0}},
    )

    assert specs["bboxes"].kwargs["atol"] == 3.0
    assert specs["bboxes"].kwargs["rtol"] == 0.0


def test_unknown_subclass_raises():
    class FooResult(Result):
        pass

    with pytest.raises(NotImplementedError, match="FooResult"):
        dispatch(FooResult())
