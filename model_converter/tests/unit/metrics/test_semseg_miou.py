#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for the semantic segmentation mIoU metric."""

import numpy as np
import pytest

from model_converter.metrics.semseg_miou import SemSegMIoU


class TestSemSegMIoU:
    def test_name_is_miou(self):
        assert SemSegMIoU(num_classes=3).name == "mIoU"

    def test_perfect_prediction(self):
        metric = SemSegMIoU(num_classes=3)
        mask = np.array([[0, 1], [2, 0]])
        metric.update(mask, mask)
        assert metric.compute() == pytest.approx(1.0)

    def test_completely_wrong_prediction(self):
        metric = SemSegMIoU(num_classes=2)
        gt = np.array([[0, 0], [0, 0]])
        pred = np.array([[1, 1], [1, 1]])
        metric.update(pred, gt)
        # No intersection for any class present → 0
        assert metric.compute() == pytest.approx(0.0)

    def test_partial_overlap(self):
        # 2 classes, 4 pixels. GT = [0,0,1,1], Pred = [0,1,1,1].
        # Class 0: TP=1, FP=0, FN=1 → IoU = 1/2 = 0.5
        # Class 1: TP=2, FP=1, FN=0 → IoU = 2/3
        # mIoU = (0.5 + 2/3) / 2 ≈ 0.5833
        metric = SemSegMIoU(num_classes=2)
        gt = np.array([0, 0, 1, 1])
        pred = np.array([0, 1, 1, 1])
        metric.update(pred, gt)
        assert metric.compute() == pytest.approx((0.5 + 2 / 3) / 2)

    def test_ignore_index_excluded(self):
        metric = SemSegMIoU(num_classes=2, ignore_index=255)
        gt = np.array([0, 0, 255, 1])
        # If ignored pixel were counted, prediction "1" at index 2 would be FP for class 1.
        pred = np.array([0, 0, 1, 1])
        metric.update(pred, gt)
        # With ignore: class 0 IoU = 2/2 = 1.0, class 1 IoU = 1/1 = 1.0 → mIoU = 1.0
        assert metric.compute() == pytest.approx(1.0)

    def test_reset_clears_confusion_matrix(self):
        metric = SemSegMIoU(num_classes=2)
        metric.update(np.array([1, 1]), np.array([0, 0]))
        metric.reset()
        metric.update(np.array([0, 0]), np.array([0, 0]))
        assert metric.compute() == pytest.approx(1.0)

    def test_compute_without_updates_returns_zero(self):
        assert SemSegMIoU(num_classes=2).compute() == pytest.approx(0.0)
