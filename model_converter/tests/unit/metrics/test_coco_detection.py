#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for the COCO detection mAP metric (wraps pycocotools)."""

import json
from pathlib import Path

import pytest
from model_converter.metrics.coco_detection import COCO80_TO_COCO91, CocoDetectionMAP


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

    def test_category_ids_are_coco91_defaults_false(self):
        metric = CocoDetectionMAP(annotation_file=Path("/nonexistent"), iou_type="bbox")
        assert metric.category_ids_are_coco91 is False

    def test_category_ids_are_coco91_can_be_enabled(self):
        metric = CocoDetectionMAP(
            annotation_file=Path("/nonexistent"),
            iou_type="bbox",
            category_ids_are_coco91=True,
        )
        assert metric.category_ids_are_coco91 is True

    def test_label_offset_defaults_zero(self):
        metric = CocoDetectionMAP(annotation_file=Path("/nonexistent"), iou_type="bbox")
        assert metric.label_offset == 0

    def test_label_offset_can_be_set(self):
        metric = CocoDetectionMAP(
            annotation_file=Path("/nonexistent"),
            iou_type="bbox",
            label_offset=1,
        )
        assert metric.label_offset == 1


class TestCoco80ToCoco91:
    def test_length_is_80(self):
        assert len(COCO80_TO_COCO91) == 80

    def test_no_duplicates(self):
        assert len(set(COCO80_TO_COCO91)) == 80

    def test_first_value_is_person(self):
        assert COCO80_TO_COCO91[0] == 1  # person

    def test_last_value_is_90(self):
        assert COCO80_TO_COCO91[-1] == 90  # toothbrush

    def test_stop_sign_maps_to_13_not_12(self):
        """Class 11 (stop sign) must be COCO ID 13; COCO ID 12 (street sign) is absent."""
        assert COCO80_TO_COCO91[11] == 13
