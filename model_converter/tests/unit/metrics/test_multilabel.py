#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for the multilabel mean Average Precision metric."""

import numpy as np
import pytest

from model_converter.metrics.multilabel import MultilabelMAP


class TestMultilabelMAP:
    def test_name_is_map(self):
        assert MultilabelMAP(num_labels=3).name == "mAP"

    def test_perfect_predictions_score_one(self):
        metric = MultilabelMAP(num_labels=3)
        # logits clearly favour the positive label per sample
        metric.update(np.array([[10.0, -10.0, -10.0]]), np.array([1, 0, 0]))
        metric.update(np.array([[-10.0, 10.0, -10.0]]), np.array([0, 1, 0]))
        metric.update(np.array([[-10.0, -10.0, 10.0]]), np.array([0, 0, 1]))
        assert metric.compute() == pytest.approx(1.0, abs=1e-3)

    def test_random_predictions_below_perfect(self):
        rng = np.random.default_rng(0)
        metric = MultilabelMAP(num_labels=5)
        for _ in range(20):
            logits = rng.normal(size=(1, 5)).astype(np.float32)
            gt = rng.integers(0, 2, size=5)
            metric.update(logits, gt)
        score = metric.compute()
        assert 0.0 <= score < 1.0

    def test_reset_clears_state(self):
        metric = MultilabelMAP(num_labels=2)
        metric.update(np.array([[10.0, -10.0]]), np.array([1, 0]))
        metric.reset()
        metric.update(np.array([[-10.0, 10.0]]), np.array([1, 0]))
        # After reset, predicting wrong label every time → score is low
        assert metric.compute() < 0.6

    def test_accepts_1d_logits_input(self):
        """Per-sample logits may be passed as a 1D vector; the metric promotes to 2D."""
        metric = MultilabelMAP(num_labels=3)
        metric.update(np.array([10.0, -10.0, -10.0]), np.array([1, 0, 0]))
        metric.update(np.array([-10.0, 10.0, -10.0]), np.array([0, 1, 0]))
        metric.update(np.array([-10.0, -10.0, 10.0]), np.array([0, 0, 1]))
        assert metric.compute() == pytest.approx(1.0, abs=1e-3)
