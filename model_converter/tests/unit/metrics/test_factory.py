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
        # Non-getitune detection models use contiguous 80-class indices → remap on.
        assert metric.category_ids_are_coco91 is False

    def test_getitune_maskrcnn_with_coco_v1_uses_coco91_labels(self, tmp_path):
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        metric = metric_for(
            dataset_type="coco-detection",
            model_type="MaskRCNN",
            annotation_file=ann,
            model_library="getitune",
            labels="COCO_V1",
        )
        assert isinstance(metric, CocoDetectionMAP)
        # getitune MaskRCNN with COCO_V1 labels → labels are native COCO-91 IDs.
        assert metric.category_ids_are_coco91 is True
        assert metric.label_offset == 0

    def test_getitune_maskrcnn_with_coco81_uses_offset_remap(self, tmp_path):
        """COCO_81 MaskRCNN models output contiguous 1-80 labels (after +=1), not COCO-91."""
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        metric = metric_for(
            dataset_type="coco-detection",
            model_type="MaskRCNN",
            annotation_file=ann,
            model_library="getitune",
            labels="COCO_81",
        )
        assert isinstance(metric, CocoDetectionMAP)
        assert metric.category_ids_are_coco91 is False
        # Instance seg postprocessor applies labels+=1; offset compensates.
        assert metric.label_offset == 1

    def test_getitune_maskrcnn_with_no_labels_uses_offset_remap(self, tmp_path):
        """MaskRCNN without explicit labels falls through to contiguous remap."""
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        metric = metric_for(
            dataset_type="coco-detection",
            model_type="MaskRCNN",
            annotation_file=ann,
            model_library="getitune",
        )
        assert isinstance(metric, CocoDetectionMAP)
        assert metric.category_ids_are_coco91 is False
        assert metric.label_offset == 1

    def test_torchvision_maskrcnn_keeps_coco80_remap(self, tmp_path):
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        metric = metric_for(
            dataset_type="coco-detection",
            model_type="MaskRCNN",
            annotation_file=ann,
            model_library="torchvision",
        )
        assert isinstance(metric, CocoDetectionMAP)
        assert metric.category_ids_are_coco91 is False
        # MaskRCNN postprocessor applies labels+=1 → offset compensates.
        assert metric.label_offset == 1

    def test_getitune_non_maskrcnn_keeps_coco80_remap(self, tmp_path):
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        metric = metric_for(
            dataset_type="coco-detection",
            model_type="YOLOX",
            annotation_file=ann,
            model_library="getitune",
        )
        assert isinstance(metric, CocoDetectionMAP)
        assert metric.category_ids_are_coco91 is False
        # Non-instance-seg models have no labels+=1 → no offset needed.
        assert metric.label_offset == 0

    def test_getitune_ssd_with_coco_v1_labels_uses_coco91(self, tmp_path):
        """RF-DETR detection models use SSD type but output COCO-91 native IDs."""
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        metric = metric_for(
            dataset_type="coco-detection",
            model_type="SSD",
            annotation_file=ann,
            model_library="getitune",
            labels="COCO_V1",
        )
        assert isinstance(metric, CocoDetectionMAP)
        assert metric.category_ids_are_coco91 is True
        assert metric.label_offset == 0

    def test_getitune_ssd_with_coco80_labels_keeps_remap(self, tmp_path):
        """Getitune SSD models with COCO_80 labels use contiguous 0-79 indices."""
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        metric = metric_for(
            dataset_type="coco-detection",
            model_type="SSD",
            annotation_file=ann,
            model_library="getitune",
            labels="COCO_80",
        )
        assert isinstance(metric, CocoDetectionMAP)
        assert metric.category_ids_are_coco91 is False

    def test_detr_instance_segmentation_uses_coco_segm_map(self, tmp_path):
        """RF-DETR segmentation must measure masks with COCO-segm iouType.

        DETRInstanceSegmentation does not shift labels (unlike MaskRCNN), so no
        label_offset is needed to compensate.
        """
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        metric = metric_for(
            dataset_type="coco-detection",
            model_type="DETRInstSeg",
            annotation_file=ann,
            model_library="getitune",
            labels="COCO_V1",
        )
        assert isinstance(metric, CocoDetectionMAP)
        assert metric.iou_type == "segm"
        assert metric.category_ids_are_coco91 is True
        assert metric.label_offset == 0

    def test_getitune_maskrcnn_with_coco_v1_keeps_native_labels_unshifted(self, tmp_path):
        """The RF-DETR COCO_92 correction must not affect existing Mask R-CNN configs."""
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        metric = metric_for(
            dataset_type="coco-detection",
            model_type="MaskRCNN",
            annotation_file=ann,
            model_library="getitune",
            labels="COCO_V1",
        )
        assert isinstance(metric, CocoDetectionMAP)
        assert metric.category_ids_are_coco91 is True
        assert metric.label_offset == 0

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

    def test_ade20k_with_non_segmentation_model_returns_none(self):
        assert metric_for(dataset_type="ade20k", model_type="Classification") is None

    def test_multilabel_via_getitune_task(self):
        # Real configs report model_type="Classification" with getitune_task="MULTI_LABEL_CLS"
        metric = metric_for(
            dataset_type="imagenet-1k",
            model_type="Classification",
            task="MULTI_LABEL_CLS",
        )
        assert isinstance(metric, MultilabelMAP)

    def test_multi_class_cls_task_returns_top1(self):
        metric = metric_for(
            dataset_type="imagenet-1k",
            model_type="Classification",
            task="MULTI_CLASS_CLS",
        )
        assert isinstance(metric, TopOneAccuracy)

    def test_yolo11_bbox_map(self, tmp_path):
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        metric = metric_for(
            dataset_type="coco-detection",
            model_type="YOLO11",
            annotation_file=ann,
        )
        assert isinstance(metric, CocoDetectionMAP)
        assert metric.iou_type == "bbox"

    def test_yolov5_bbox_map(self, tmp_path):
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        metric = metric_for(
            dataset_type="coco-detection",
            model_type="YOLOv5",
            annotation_file=ann,
        )
        assert isinstance(metric, CocoDetectionMAP)

    def test_yolov8_bbox_map(self, tmp_path):
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        metric = metric_for(
            dataset_type="coco-detection",
            model_type="YOLOv8",
            annotation_file=ann,
        )
        assert isinstance(metric, CocoDetectionMAP)

    def test_rotated_detection_task_returns_none(self, tmp_path):
        # Rotated detection has no standard metric implemented here — skip
        ann = tmp_path / "ann.json"
        ann.write_text('{"images":[],"annotations":[],"categories":[]}')
        assert (
            metric_for(
                dataset_type="coco-detection",
                model_type="MaskRCNN",
                annotation_file=ann,
                task="ROTATED_DETECTION",
            )
            is None
        )
