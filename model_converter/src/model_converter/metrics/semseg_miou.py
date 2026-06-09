#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Semantic segmentation mIoU computed from a confusion matrix."""

from __future__ import annotations

import numpy as np

from model_converter.metrics.base import Metric


class SemSegMIoU(Metric):
    """Mean Intersection-over-Union with optional ignore index.

    Accumulates an ``(num_classes, num_classes)`` confusion matrix; per-class
    IoU is averaged across classes that have at least one pixel in either GT
    or prediction (others are skipped to avoid biasing toward easy zeros).
    """

    name = "mIoU"

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, prediction: np.ndarray, ground_truth: np.ndarray) -> None:
        """Accept ``prediction`` and ``ground_truth`` as same-shape class-id arrays."""
        gt = np.asarray(ground_truth).ravel()
        pred = np.asarray(prediction).ravel()
        mask = gt != self.ignore_index
        gt = gt[mask]
        pred = pred[mask]
        index = gt * self.num_classes + pred
        binc = np.bincount(index, minlength=self.num_classes * self.num_classes)
        self._confusion += binc.reshape(self.num_classes, self.num_classes)

    def compute(self) -> float:
        cm = self._confusion
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp
        denom = tp + fp + fn
        valid = denom > 0
        if not np.any(valid):
            return 0.0
        iou = np.zeros_like(denom)
        iou[valid] = tp[valid] / denom[valid]
        return float(iou[valid].mean())

    def reset(self) -> None:
        self._confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
