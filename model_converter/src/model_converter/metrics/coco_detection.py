#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""COCO-format mean Average Precision (bbox / segm) via pycocotools."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any

from model_converter.metrics.base import Metric

_VALID_IOU_TYPES = {"bbox", "segm"}

# Canonical mapping from COCO 80-class indices (0-79) used by OpenVINO detection
# models to the original COCO 91-class category IDs (non-contiguous) in COCO JSON.
# COCO dropped 11 categories (IDs 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91),
# so class index 11 (stop sign) maps to category ID 13, not 12, etc.
COCO80_TO_COCO91 = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]

# Inverse of :data:`COCO80_TO_COCO91`: maps an original COCO 91-class category ID
# to the contiguous 80-class index (0-79) used by OpenVINO detection models.
# Category IDs that COCO dropped (and the background ID 0) map to ``None``.
COCO91_TO_COCO80: list[int | None] = [None] * 91
for _idx, _cat_id in enumerate(COCO80_TO_COCO91):
    COCO91_TO_COCO80[_cat_id] = _idx
del _idx, _cat_id


class CocoDetectionMAP(Metric):
    """Wraps :class:`pycocotools.cocoeval.COCOeval` for a single ``iouType``."""

    name = "mAP"

    def __init__(
        self,
        annotation_file: Path,
        iou_type: str = "bbox",
        *,
        category_ids_are_coco91: bool = False,
        label_offset: int = 0,
    ) -> None:
        if iou_type not in _VALID_IOU_TYPES:
            error_msg = f"Unsupported iou_type {iou_type!r}; expected one of {sorted(_VALID_IOU_TYPES)}"
            raise ValueError(error_msg)
        self.annotation_file = Path(annotation_file)
        self.iou_type = iou_type
        self.category_ids_are_coco91 = category_ids_are_coco91
        self.label_offset = label_offset
        self._predictions: list[dict[str, Any]] = []

    def update(self, predictions: list[dict[str, Any]] | None = None, ground_truth: Any = None) -> None:
        """Accept a batch of COCO-format detection dicts.

        Each dict must contain ``image_id``, ``category_id``, ``bbox``
        (``[x, y, w, h]``), and ``score``. ``ground_truth`` is ignored — GT is
        loaded from :attr:`annotation_file`.
        """
        del ground_truth
        if predictions is None:
            return
        self._predictions.extend(predictions)

    def compute(self) -> float:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        if not self._predictions:
            return 0.0

        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(str(self.annotation_file))
            coco_dt = coco_gt.loadRes(self._predictions)
            evaluator = COCOeval(coco_gt, coco_dt, iouType=self.iou_type)
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        # stats[0] is the primary mAP @ IoU=0.50:0.95.
        return float(evaluator.stats[0])

    def reset(self) -> None:
        self._predictions = []
