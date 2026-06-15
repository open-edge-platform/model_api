#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Top-1 classification accuracy metric."""

from __future__ import annotations

import numpy as np

from model_converter.metrics.base import Metric


class TopOneAccuracy(Metric):
    """Fraction of samples whose argmax matches the ground-truth label."""

    name = "top1"

    def __init__(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, prediction: np.ndarray, ground_truth: int | None = None) -> None:
        """Accept ``prediction`` as a logits tensor and ``ground_truth`` as a class id."""
        assert ground_truth is not None
        pred_class = int(np.argmax(prediction, axis=1)[0])
        self._correct += int(pred_class == int(ground_truth))
        self._total += 1

    def compute(self) -> float:
        if self._total == 0:
            return 0.0
        return self._correct / self._total

    def reset(self) -> None:
        self._correct = 0
        self._total = 0
