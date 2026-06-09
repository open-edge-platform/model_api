#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""ADE20K semantic-segmentation dataset reader."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import CalibrationSample, DatasetReader

if TYPE_CHECKING:
    from collections.abc import Iterator


class Ade20kReader(DatasetReader):
    """Enumerate ``<root>/images/*.jpg`` paired with ``<root>/annotations/*.png`` masks.

    Only images with a matching mask (same stem) are emitted. Mask paths
    are not exposed here; downstream mIoU computation derives them from
    the image stem.
    """

    def __init__(self, root) -> None:
        super().__init__(root)
        self.images_dir = self.root / "images"
        self.annotations_dir = self.root / "annotations"

    def __iter__(self) -> Iterator[CalibrationSample]:
        if not self.images_dir.exists():
            msg = f"ADE20K images directory missing: {self.images_dir}"
            raise FileNotFoundError(msg)
        if not self.annotations_dir.exists():
            msg = f"ADE20K annotations directory missing: {self.annotations_dir}"
            raise FileNotFoundError(msg)
        for img_path in sorted(self.images_dir.glob("*.jpg")):
            mask_path = self.annotations_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                yield CalibrationSample(
                    image_path=img_path,
                    label=0,
                    mask_path=mask_path,
                )
