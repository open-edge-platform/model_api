#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Dispatch ``dataset_type`` strings to concrete reader implementations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .ade20k import Ade20kReader
from .class_folder import ClassFolderReader
from .coco import CocoImagesReader

if TYPE_CHECKING:
    from .base import DatasetReader

_CLASS_FOLDER_TYPES = frozenset({"imagenet-1k", "imagenet-21k"})
_COCO_ANNOTATION_FILES = {
    "coco-detection": "instances_val2017.json",
}


def reader_for(dataset_type: str | None, root: Path) -> DatasetReader:
    """Return the :class:`DatasetReader` matching ``dataset_type``.

    ``None`` falls back to :class:`ClassFolderReader` so that callers without
    a configured dataset type (e.g. legacy code paths) keep working.
    """
    root = Path(root)
    if dataset_type is None or dataset_type in _CLASS_FOLDER_TYPES:
        return ClassFolderReader(root)
    if dataset_type in _COCO_ANNOTATION_FILES:
        return CocoImagesReader(root, annotation_filename=_COCO_ANNOTATION_FILES[dataset_type])
    if dataset_type == "ade20k":
        return Ade20kReader(root)
    msg = f"Unknown dataset_type: {dataset_type!r}"
    raise ValueError(msg)
