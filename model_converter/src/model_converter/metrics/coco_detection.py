#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""COCO-format mean Average Precision (bbox / segm / keypoints) via pycocotools."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any

from model_converter.metrics.base import Metric

_VALID_IOU_TYPES = {"bbox", "segm", "keypoints"}


class CocoDetectionMAP(Metric):
    """Wraps :class:`pycocotools.cocoeval.COCOeval` for a single ``iouType``."""

    name = "mAP"

    def __init__(self, annotation_file: Path, iou_type: str = "bbox") -> None:
        if iou_type not in _VALID_IOU_TYPES:
            error_msg = f"Unsupported iou_type {iou_type!r}; expected one of {sorted(_VALID_IOU_TYPES)}"
            raise ValueError(error_msg)
        self.annotation_file = Path(annotation_file)
        self.iou_type = iou_type
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
        # stats[0] is the primary mAP @ IoU=0.50:0.95 (or OKS for keypoints).
        return float(evaluator.stats[0])

    def reset(self) -> None:
        self._predictions = []
