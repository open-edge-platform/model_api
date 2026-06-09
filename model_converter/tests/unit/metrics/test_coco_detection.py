#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for the COCO detection mAP metric (wraps pycocotools)."""

import json
from pathlib import Path

import pytest

from model_converter.metrics.coco_detection import CocoDetectionMAP


def _write_minimal_coco_gt(path: Path) -> None:
    """Write a tiny COCO-format ground truth file with one image and one box."""
    gt = {
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "img1.jpg"}],
        "categories": [{"id": 1, "name": "obj", "supercategory": "none"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],  # [x, y, w, h]
                "area": 400,
                "iscrowd": 0,
            },
        ],
    }
    path.write_text(json.dumps(gt))


class TestCocoDetectionMAP:
    def test_name_for_bbox_iou_type(self):
        assert CocoDetectionMAP(annotation_file=Path("/nonexistent"), iou_type="bbox").name == "mAP"

    def test_name_for_keypoints_iou_type(self):
        assert CocoDetectionMAP(annotation_file=Path("/nonexistent"), iou_type="keypoints").name == "mAP"

    def test_perfect_detection_scores_one(self, tmp_path):
        gt_file = tmp_path / "gt.json"
        _write_minimal_coco_gt(gt_file)
        metric = CocoDetectionMAP(annotation_file=gt_file, iou_type="bbox")
        # Predict the exact ground-truth box with score 1.0
        metric.update(
            predictions=[
                {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 1.0},
            ],
        )
        assert metric.compute() == pytest.approx(1.0, abs=1e-3)

    def test_no_predictions_scores_zero(self, tmp_path):
        gt_file = tmp_path / "gt.json"
        _write_minimal_coco_gt(gt_file)
        metric = CocoDetectionMAP(annotation_file=gt_file, iou_type="bbox")
        # Update with no predictions at all
        metric.update(predictions=[])
        assert metric.compute() == pytest.approx(0.0)

    def test_reset_clears_predictions(self, tmp_path):
        gt_file = tmp_path / "gt.json"
        _write_minimal_coco_gt(gt_file)
        metric = CocoDetectionMAP(annotation_file=gt_file, iou_type="bbox")
        metric.update(predictions=[{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 1.0}])
        metric.reset()
        metric.update(predictions=[])
        assert metric.compute() == pytest.approx(0.0)

    def test_invalid_iou_type_raises(self):
        with pytest.raises(ValueError, match="iou_type"):
            CocoDetectionMAP(annotation_file=Path("/nonexistent"), iou_type="invalid")

    def test_update_with_none_predictions_is_a_noop(self, tmp_path):
        gt_file = tmp_path / "gt.json"
        _write_minimal_coco_gt(gt_file)
        metric = CocoDetectionMAP(annotation_file=gt_file, iou_type="bbox")
        metric.update(predictions=None)
        assert metric.compute() == pytest.approx(0.0)
