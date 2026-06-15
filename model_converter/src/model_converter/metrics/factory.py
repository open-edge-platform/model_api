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
    "YOLO",
    "YOLOv4",
    "YOLOv5",
    "YOLOv8",
    "YOLO11",
    "YOLOX",
    "MaskRCNN",
    "RotatedDetection",
    "DETR",
    "Detection",
}
_SEMSEG_TYPES = {"Segmentation"}

_MULTILABEL_TASKS = {"MULTI_LABEL_CLS"}
_MULTI_CLASS_TASKS = {"MULTI_CLASS_CLS"}
_UNSUPPORTED_TASKS = {"ROTATED_DETECTION"}

_IMAGENET_LABEL_COUNTS = {
    "imagenet-1k": 1000,
    "imagenet-21k": 21843,
}


def metric_for(
    dataset_type: str | None,
    model_type: str | None,
    *,
    annotation_file: "Path | None" = None,
    task: str | None = None,
) -> "Metric | None":
    """Return a fresh metric instance for the given dataset/model combo, or ``None``.

    ``None`` means accuracy is not measured for this configuration (unknown
    dataset type, headless feature extractor, rotated detection, etc.).

    The optional ``task`` argument is the OpenVINO Training Extensions task
    label (``getitune_task``) and is used to disambiguate cases where
    ``model_type`` alone does not carry enough information — for example,
    ``Classification`` model_type with task ``MULTI_LABEL_CLS`` maps to
    multilabel mAP, not top-1 accuracy.
    """
    if dataset_type is None or model_type is None:
        return None

    if task in _UNSUPPORTED_TASKS:
        return None

    if dataset_type in _IMAGENET_LABEL_COUNTS:
        if task in _MULTILABEL_TASKS or model_type in _IMAGENET_MULTILABEL_TYPES:
            return MultilabelMAP(num_labels=_IMAGENET_LABEL_COUNTS[dataset_type])
        if task in _MULTI_CLASS_TASKS or model_type in _IMAGENET_CLASSIFICATION_TYPES:
            return TopOneAccuracy()
        return None

    if dataset_type == "coco-detection":
        if annotation_file is None or model_type not in _COCO_DETECTION_TYPES:
            return None
        return CocoDetectionMAP(annotation_file=annotation_file, iou_type="bbox")

    if dataset_type == "ade20k":
        if model_type not in _SEMSEG_TYPES:
            return None
        return SemSegMIoU(num_classes=150, ignore_index=255)

    return None
