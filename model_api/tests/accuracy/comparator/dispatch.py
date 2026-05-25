"""Dispatch skeleton for routing result comparisons by result type."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tests.accuracy.comparator.policies import ComparisonPolicy

from model_api.models.result.anomaly import AnomalyResult
from model_api.models.result.classification import ClassificationResult
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


@dataclass
class FieldSpec:
    policy: ComparisonPolicy
    value: Any
    kwargs: dict = field(default_factory=dict)


def _field(policy: ComparisonPolicy, value: Any, **kwargs) -> FieldSpec:
    return FieldSpec(policy=policy, value=value, kwargs=kwargs)


def _detection_specs(result: DetectionResult) -> dict[str, FieldSpec]:
    instance_matched = {"instance_matched": True}
    return {
        "labels": _field(ComparisonPolicy.EXACT, result.labels, **instance_matched),
        "label_names": _field(ComparisonPolicy.EXACT, result.label_names, **instance_matched),
        "bboxes": _field(
            ComparisonPolicy.NUMERIC_CLOSE,
            result.bboxes,
            atol=2.0,
            **instance_matched,
        ),
        "scores": _field(ComparisonPolicy.NUMERIC_CLOSE, result.scores, **instance_matched),
        "saliency_map": _field(ComparisonPolicy.STAT_FINGERPRINT, result.saliency_map),
        "feature_vector": _field(ComparisonPolicy.STAT_FINGERPRINT, result.feature_vector),
    }


def _apply_policy_overrides(
    specs: dict[str, FieldSpec],
    policy_overrides: dict | None,
) -> dict[str, FieldSpec]:
    if policy_overrides is None:
        return specs

    for field_name, override_kwargs in policy_overrides.items():
        if field_name in specs:
            specs[field_name].kwargs.update(override_kwargs)
    return specs


def dispatch(result, *, policy_overrides: dict | None = None) -> dict[str, FieldSpec]:
    """Return per-field specifications for a Result instance.

    policy_overrides maps field names to kwargs merged into FieldSpec.kwargs.
    Raises NotImplementedError for unknown subclasses.
    """

    if isinstance(result, ClassificationResult):
        specs = {
            "top_labels": _field(ComparisonPolicy.EXACT, result.top_labels),
            "raw_scores": _field(
                ComparisonPolicy.NUMERIC_CLOSE,
                result.raw_scores,
                atol=1e-2,
                rtol=1e-2,
            ),
            "saliency_map": _field(ComparisonPolicy.STAT_FINGERPRINT, result.saliency_map),
            "feature_vector": _field(ComparisonPolicy.STAT_FINGERPRINT, result.feature_vector),
        }
    elif isinstance(result, AnomalyResult):
        specs = {
            "pred_label": _field(ComparisonPolicy.EXACT, result.pred_label),
            "pred_score": _field(ComparisonPolicy.NUMERIC_CLOSE, result.pred_score),
            "pred_mask": _field(ComparisonPolicy.MASK_IOU, result.pred_mask),
            "anomaly_map": _field(ComparisonPolicy.STAT_FINGERPRINT, result.anomaly_map),
        }
    elif isinstance(result, RotatedSegmentationResult):
        specs = _detection_specs(result)
        specs["masks"] = _field(
            ComparisonPolicy.MASK_IOU,
            result.masks,
            instance_matched=True,
        )
        specs["rotated_rects"] = _field(
            ComparisonPolicy.NUMERIC_CLOSE,
            result.rotated_rects,
            instance_matched=True,
            use_rotated_iou=True,
        )
    elif isinstance(result, InstanceSegmentationResult):
        specs = _detection_specs(result)
        specs["masks"] = _field(
            ComparisonPolicy.MASK_IOU,
            result.masks,
            instance_matched=True,
        )
    elif isinstance(result, DetectionResult):
        specs = _detection_specs(result)
    elif isinstance(result, ImageResultWithSoftPrediction):
        specs = {
            "resultImage": _field(ComparisonPolicy.MASK_IOU, result.resultImage),
            "soft_prediction": _field(ComparisonPolicy.STAT_FINGERPRINT, result.soft_prediction),
            "saliency_map": _field(ComparisonPolicy.STAT_FINGERPRINT, result.saliency_map),
            "feature_vector": _field(ComparisonPolicy.STAT_FINGERPRINT, result.feature_vector),
        }
    elif isinstance(result, DetectedKeypoints):
        specs = {
            "keypoints": _field(ComparisonPolicy.NUMERIC_CLOSE, result.keypoints),
            "scores": _field(ComparisonPolicy.NUMERIC_CLOSE, result.scores),
        }
    elif isinstance(result, VisualPromptingResult):
        specs = {
            "best_iou": _field(ComparisonPolicy.NUMERIC_CLOSE, result.best_iou),
            "iou_predictions": _field(ComparisonPolicy.NUMERIC_CLOSE, result.iou_predictions),
            "scores": _field(ComparisonPolicy.NUMERIC_CLOSE, result.scores),
            "labels": _field(ComparisonPolicy.EXACT, result.labels),
            "upscaled_masks": _field(ComparisonPolicy.STAT_FINGERPRINT, result.upscaled_masks),
            "processed_masks": _field(ComparisonPolicy.MASK_IOU, result.processed_mask),
            "low_res_masks": _field(ComparisonPolicy.STAT_FINGERPRINT, result.low_res_masks),
            "hard_predictions": _field(ComparisonPolicy.MASK_IOU, result.hard_predictions),
        }
    elif isinstance(result, ZSLVisualPromptingResult):
        specs = {}
        for key, predicted_mask in result.data.items():
            if isinstance(predicted_mask, PredictedMask):
                specs[f"{key}.mask"] = _field(ComparisonPolicy.MASK_IOU, predicted_mask.mask)
                specs[f"{key}.points"] = _field(ComparisonPolicy.NUMERIC_CLOSE, predicted_mask.points)
                specs[f"{key}.scores"] = _field(ComparisonPolicy.NUMERIC_CLOSE, predicted_mask.scores)
    else:
        raise NotImplementedError(type(result).__name__)

    return _apply_policy_overrides(specs, policy_overrides)
