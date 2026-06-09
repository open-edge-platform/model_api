#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Abstract metric interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    """Accumulator-style metric.

    Subclasses are responsible for advertising a short label via :attr:`name`
    (e.g. ``"top1"``, ``"mAP"``, ``"mIoU"``), accepting one prediction at a
    time via :meth:`update`, returning a scalar via :meth:`compute`, and
    clearing internal state via :meth:`reset`.
    """

    name: str

    @abstractmethod
    def update(self, prediction: Any, ground_truth: Any = None) -> None:
        """Accumulate a single prediction with its ground truth."""

    @abstractmethod
    def compute(self) -> float:
        """Return the metric value for everything accumulated so far."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state."""
