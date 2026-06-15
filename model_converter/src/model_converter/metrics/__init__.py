#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Task-aware accuracy metrics for the model converter pipeline.

Each metric implements the :class:`~model_converter.metrics.base.Metric`
interface (``update``, ``compute``, ``reset``, plus a ``name`` label such as
``"top1"``, ``"mAP"``, ``"mIoU"``). The :func:`metric_for` factory dispatches
to the right metric for a given ``dataset_type`` and ``model_type``.
"""

from model_converter.metrics.base import Metric
from model_converter.metrics.classification import TopOneAccuracy
from model_converter.metrics.coco_detection import CocoDetectionMAP
from model_converter.metrics.factory import metric_for
from model_converter.metrics.multilabel import MultilabelMAP
from model_converter.metrics.semseg_miou import SemSegMIoU

__all__ = [
    "CocoDetectionMAP",
    "Metric",
    "MultilabelMAP",
    "SemSegMIoU",
    "TopOneAccuracy",
    "metric_for",
]
