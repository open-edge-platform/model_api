#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Multilabel mean Average Precision (macro) via torchmetrics."""

from __future__ import annotations

import numpy as np
import torch
from torchmetrics.classification import MultilabelAveragePrecision

from model_converter.metrics.base import Metric


class MultilabelMAP(Metric):
    """Threshold-free macro mAP for multi-label classification."""

    name = "mAP"

    def __init__(self, num_labels: int) -> None:
        self.num_labels = num_labels
        self._impl = MultilabelAveragePrecision(num_labels=num_labels, average="macro")

    def update(self, prediction: np.ndarray, ground_truth: np.ndarray | None = None) -> None:
        """Accept ``prediction`` as logits of shape ``(1, num_labels)`` and ``ground_truth`` as a 0/1 vector."""
        logits = torch.from_numpy(np.asarray(prediction, dtype=np.float32))
        target = torch.from_numpy(np.asarray(ground_truth, dtype=np.int64))
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if target.ndim == 1:
            target = target.unsqueeze(0)
        self._impl.update(logits, target)

    def compute(self) -> float:
        return float(self._impl.compute().item())

    def reset(self) -> None:
        self._impl.reset()
