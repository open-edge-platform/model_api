#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Common types for dataset readers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class CalibrationSample:
    """One image plus an optional integer label.

    ``label`` is the class id for class-folder datasets and a placeholder
    ``0`` for layouts (COCO, ADE20K) where per-image classification labels
    do not apply.
    """

    image_path: Path
    label: int


class DatasetReader(ABC):
    """Abstract enumerator for calibration/validation samples on disk."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @abstractmethod
    def __iter__(self) -> Iterator[CalibrationSample]:
        """Yield samples in deterministic order."""
