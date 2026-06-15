#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Class-folder dataset layout (ImageNet-style)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import CalibrationSample, DatasetReader

if TYPE_CHECKING:
    from collections.abc import Iterator

_IMAGE_PATTERNS = ("*.JPEG", "*.jpg", "*.jpeg", "*.png")


class ClassFolderReader(DatasetReader):
    """Enumerate ``<root>/<class_id>/*.JPEG|jpg|png`` samples.

    Class-folder names must be integers. This matches the layout used by
    ImageNet-1k and ImageNet-21k validation sets.
    """

    def __iter__(self) -> Iterator[CalibrationSample]:
        if not self.root.exists():
            return
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_label = int(class_dir.name)  # raises ValueError on bad layout
            img_paths: set[object] = set()
            for pattern in _IMAGE_PATTERNS:
                img_paths.update(class_dir.glob(pattern))
            for img_path in sorted(img_paths):
                yield CalibrationSample(image_path=img_path, label=class_label)
