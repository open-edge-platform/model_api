#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Dispatcher mapping ``(dataset_type, model_type)`` → :class:`Metric` instance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_converter.metrics.classification import TopOneAccuracy
from model_converter.metrics.coco_detection import CocoDetectionMAP
from model_converter.metrics.multilabel import MultilabelMAP
from model_converter.metrics.semseg_miou import SemSegMIoU

if TYPE_CHECKING:
    from pathlib import Path

    from model_converter.metrics.base import Metric

_IMAGENET_CLASSIFICATION_TYPES = {"Classification"}
_IMAGENET_MULTILABEL_TYPES = {"Classification_Multilabel"}
_COCO_DETECTION_TYPES = {
    "SSD",
    "YOLOv4",
    "YOLOX",
    "YOLO",
    "MaskRCNN",
    "RotatedDetection",
    "DETR",
    "Detection",
}
_COCO_KEYPOINT_TYPES = {"KeypointDetection"}
_SEMSEG_TYPES = {"Segmentation"}

_IMAGENET_LABEL_COUNTS = {
    "imagenet-1k": 1000,
    "imagenet-21k": 21843,
}


def metric_for(
    dataset_type: str | None,
    model_type: str | None,
    *,
    annotation_file: "Path | None" = None,
) -> "Metric | None":
    """Return a fresh metric instance for the given dataset/model combo, or ``None``.

    ``None`` means accuracy is not measured for this configuration (unknown
    dataset type, headless feature extractor, etc.).
    """
    if dataset_type is None or model_type is None:
        return None

    if dataset_type in _IMAGENET_LABEL_COUNTS:
        if model_type in _IMAGENET_CLASSIFICATION_TYPES:
            return TopOneAccuracy()
        if model_type in _IMAGENET_MULTILABEL_TYPES:
            return MultilabelMAP(num_labels=_IMAGENET_LABEL_COUNTS[dataset_type])
        return None

    if dataset_type == "coco-detection":
        if annotation_file is None or model_type not in _COCO_DETECTION_TYPES:
            return None
        return CocoDetectionMAP(annotation_file=annotation_file, iou_type="bbox")

    if dataset_type == "coco-keypoints":
        if annotation_file is None or model_type not in _COCO_KEYPOINT_TYPES:
            return None
        return CocoDetectionMAP(annotation_file=annotation_file, iou_type="keypoints")

    if dataset_type == "ade20k":
        if model_type not in _SEMSEG_TYPES:
            return None
        return SemSegMIoU(num_classes=150, ignore_index=255)

    return None
