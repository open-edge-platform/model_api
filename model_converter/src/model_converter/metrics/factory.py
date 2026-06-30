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

# Instance segmentation model types whose postprocessor applies ``labels += 1``
# (shifting from 0-indexed to 1-indexed).  When these models use contiguous
# COCO-80 class indices (i.e. ``category_ids_are_coco91`` is ``False``), the
# COCO80→COCO91 remap must compensate for the +1 offset.
_INSTANCE_SEG_TYPES = {"MaskRCNN", "DETRInstSeg"}

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
    model_library: str | None = None,
    labels: str | None = None,
) -> "Metric | None":
    """Return a fresh metric instance for the given dataset/model combo, or ``None``.

    ``None`` means accuracy is not measured for this configuration (unknown
    dataset type, headless feature extractor, rotated detection, etc.).

    The optional ``task`` argument is the OpenVINO Training Extensions task
    label (``getitune_task``) and is used to disambiguate cases where
    ``model_type`` alone does not carry enough information — for example,
    ``Classification`` model_type with task ``MULTI_LABEL_CLS`` maps to
    multilabel mAP, not top-1 accuracy.

    ``model_library`` selects the COCO category-id scheme: getitune models
    using ``COCO_V1`` or ``COCO_92`` label sets report labels that are already
    COCO-91 category IDs, so the metric is told to skip the COCO80->COCO91
    remap for them.

    ``labels`` is the label-set identifier from the model config (e.g.
    ``"COCO_V1"``, ``"COCO_80"``). Models using ``COCO_V1`` or ``"COCO_92"``
    output native COCO-91 category IDs regardless of ``model_type``.

    Instance segmentation models (``MaskRCNN``, ``DETRInstSeg``) add +1 to
    labels in their postprocessor.  When these models use contiguous COCO-80
    class indices (i.e. not COCO_V1/COCO_92), the metric is given a
    ``label_offset=1`` so the COCO80→COCO91 remap operates on the correct
    0-indexed value.
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
        category_ids_are_coco91 = model_library == "getitune" and labels in ("COCO_V1", "COCO_92")
        # Instance segmentation postprocessors apply ``labels += 1``; compensate
        # when using the COCO80→COCO91 remap so it receives 0-indexed values.
        label_offset = 1 if (model_type in _INSTANCE_SEG_TYPES and not category_ids_are_coco91) else 0
        return CocoDetectionMAP(
            annotation_file=annotation_file,
            iou_type="bbox",
            category_ids_are_coco91=category_ids_are_coco91,
            label_offset=label_offset,
        )

    if dataset_type == "ade20k":
        if model_type not in _SEMSEG_TYPES:
            return None
        return SemSegMIoU(num_classes=150, ignore_index=255)

    return None
