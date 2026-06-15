#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""COCO-format dataset reader (images + annotations JSON)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .base import CalibrationSample, DatasetReader

if TYPE_CHECKING:
    from collections.abc import Iterator


class CocoImagesReader(DatasetReader):
    """Enumerate ``<root>/images/*.jpg`` produced by ``download_coco.py``.

    The reader validates that the configured COCO-format annotation file is
    present so downstream metric code can rely on it without re-checking.
    Per-image ``label`` is a placeholder ``0`` because COCO ground truth is
    consumed via the annotation JSON rather than per-image labels. The
    ``image_id`` field is populated from the annotation JSON so prediction
    records can reference the correct COCO image.
    """

    def __init__(self, root, annotation_filename: str) -> None:
        super().__init__(root)
        self.annotation_filename = annotation_filename
        self.images_dir = self.root / "images"
        self.annotations_path = self.root / "annotations" / annotation_filename

    def __iter__(self) -> Iterator[CalibrationSample]:
        if not self.images_dir.exists():
            msg = f"COCO images directory missing: {self.images_dir}"
            raise FileNotFoundError(msg)
        if not self.annotations_path.exists():
            msg = f"COCO annotations file missing: {self.annotations_path}"
            raise FileNotFoundError(msg)
        filename_to_id = _load_filename_to_image_id(self.annotations_path)
        for img_path in sorted(self.images_dir.glob("*.jpg")):
            yield CalibrationSample(
                image_path=img_path,
                label=0,
                image_id=filename_to_id.get(img_path.name),
            )


def _load_filename_to_image_id(annotation_path) -> dict[str, int]:
    payload = json.loads(annotation_path.read_text())
    return {entry["file_name"]: int(entry["id"]) for entry in payload.get("images", [])}
