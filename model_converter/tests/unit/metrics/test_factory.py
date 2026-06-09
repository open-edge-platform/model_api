#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for the metric dispatcher."""

from pathlib import Path

from model_converter.metrics import (
    CocoDetectionMAP,
    MultilabelMAP,
    SemSegMIoU,
    TopOneAccuracy,
    metric_for,
)


class TestMetricFor:
    def test_imagenet_classification_returns_top1(self):
        metric = metric_for(dataset_type="imagenet-1k", model_type="Classification")
        assert isinstance(metric, TopOneAccuracy)

    def test_imagenet21k_classification_returns_top1(self):
        metric = metric_for(dataset_type="imagenet-21k", model_type="Classification")
        assert isinstance(metric, TopOneAccuracy)

    def test_multilabel_classification_returns_map(self):
        metric = metric_for(dataset_type="imagenet-1k", model_type="Classification_Multilabel")
        assert isinstance(metric, MultilabelMAP)

    def test_coco_detection_returns_bbox_map(self, tmp_path):
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        metric = metric_for(
            dataset_type="coco-detection",
            model_type="SSD",
            annotation_file=ann,
        )
        assert isinstance(metric, CocoDetectionMAP)
        assert metric.iou_type == "bbox"

    def test_coco_keypoints_returns_keypoint_map(self, tmp_path):
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        metric = metric_for(
            dataset_type="coco-keypoints",
            model_type="KeypointDetection",
            annotation_file=ann,
        )
        assert isinstance(metric, CocoDetectionMAP)
        assert metric.iou_type == "keypoints"

    def test_ade20k_returns_miou(self):
        metric = metric_for(dataset_type="ade20k", model_type="Segmentation")
        assert isinstance(metric, SemSegMIoU)
        assert metric.num_classes == 150
        assert metric.ignore_index == 255

    def test_unknown_dataset_returns_none(self):
        assert metric_for(dataset_type="not-a-dataset", model_type="Classification") is None

    def test_none_dataset_returns_none(self):
        assert metric_for(dataset_type=None, model_type="Classification") is None

    def test_coco_without_annotation_file_returns_none(self):
        # Without annotation_file, COCO metric cannot be constructed → None
        assert metric_for(dataset_type="coco-detection", model_type="SSD") is None

    def test_dinov2_feature_extractor_returns_none(self):
        # vit_small_patch14_dinov2.lvd142m has no classification head
        assert metric_for(dataset_type="imagenet-1k", model_type="FeatureExtractor") is None

    def test_unknown_model_type_with_imagenet_returns_none(self):
        # An unknown model_type with imagenet should not crash; returns None
        assert metric_for(dataset_type="imagenet-1k", model_type=None) is None

    def test_coco_metric_uses_provided_annotation_file(self, tmp_path):
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        metric = metric_for(
            dataset_type="coco-detection",
            model_type="SSD",
            annotation_file=ann,
        )
        assert isinstance(metric, CocoDetectionMAP)
        # internal field — used to ensure dispatcher routes the path through
        assert Path(metric.annotation_file) == ann

    def test_coco_keypoints_with_wrong_model_type_returns_none(self, tmp_path):
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        assert (
            metric_for(
                dataset_type="coco-keypoints",
                model_type="SSD",
                annotation_file=ann,
            )
            is None
        )

    def test_ade20k_with_non_segmentation_model_returns_none(self):
        assert metric_for(dataset_type="ade20k", model_type="Classification") is None
