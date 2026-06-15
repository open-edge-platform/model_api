#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for the top-1 classification metric."""

import numpy as np
import pytest
from model_converter.metrics.classification import TopOneAccuracy


class TestTopOneAccuracy:
    def test_name_is_top1(self):
        assert TopOneAccuracy().name == "top1"

    def test_perfect_predictions(self):
        metric = TopOneAccuracy()
        # logits with argmax matching label
        metric.update(np.array([[0.1, 0.7, 0.2]]), 1)
        metric.update(np.array([[0.9, 0.05, 0.05]]), 0)
        assert metric.compute() == pytest.approx(1.0)

    def test_partial_correct(self):
        metric = TopOneAccuracy()
        metric.update(np.array([[0.1, 0.7, 0.2]]), 1)  # correct
        metric.update(np.array([[0.9, 0.05, 0.05]]), 1)  # wrong
        assert metric.compute() == pytest.approx(0.5)

    def test_reset_clears_counts(self):
        metric = TopOneAccuracy()
        metric.update(np.array([[0.1, 0.7, 0.2]]), 1)
        metric.reset()
        metric.update(np.array([[0.9, 0.05, 0.05]]), 1)
        assert metric.compute() == pytest.approx(0.0)

    def test_compute_without_updates_returns_zero(self):
        assert TopOneAccuracy().compute() == pytest.approx(0.0)
