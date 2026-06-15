#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Calibration/validation dataset readers.

Provides a small abstraction over the on-disk layouts used by the
converter for different task families (classification, detection,
segmentation, etc.).

Each reader yields :class:`CalibrationSample` objects with at least an
``image_path``. Readers for classification datasets also populate the
``label`` field with the integer class id; readers for other tasks emit a
placeholder ``0`` because per-image ground truth for those tasks is
consumed via task-specific metric modules (Phase 4), not via the
calibration loader.
"""

from .ade20k import Ade20kReader
from .base import CalibrationSample, DatasetReader
from .class_folder import ClassFolderReader
from .coco import CocoImagesReader
from .factory import reader_for

__all__ = [
    "Ade20kReader",
    "CalibrationSample",
    "ClassFolderReader",
    "CocoImagesReader",
    "DatasetReader",
    "reader_for",
]
