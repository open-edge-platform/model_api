"""Instance Segmentation Scene."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import random
from typing import Union

import cv2
from PIL import Image

from model_api.models.result import InstanceSegmentationResult
from model_api.visualizer.layout import Flatten, HStack, Layout
from model_api.visualizer.primitive import BoundingBox, Label, Overlay, Polygon
from model_api.visualizer.scene import Scene


class InstanceSegmentationScene(Scene):
    """Instance Segmentation Scene."""

    def __init__(self, image: Image, result: InstanceSegmentationResult, layout: Union[Layout, None] = None) -> None:
        # nosec as random is used for color generation
        g = random.Random(0)  # noqa: S311 # nosec B311
        self.color_per_label = {label: f"#{g.randint(0, 0xFFFFFF):06x}" for label in set(result.label_names)}  # nosec B311
        super().__init__(
            base=image,
            label=self._get_labels(result),
            overlay=self._get_overlays(result),
            polygon=self._get_polygons(result),
            layout=layout,
        )

    def _get_labels(self, result: InstanceSegmentationResult) -> list[Label]:
        # add only unique labels
        labels = []
        for label_name in set(result.label_names):
            labels.append(Label(label=label_name, bg_color=self.color_per_label[label_name]))
        return labels

    def _get_polygons(self, result: InstanceSegmentationResult) -> list[Polygon]:
        polygons = []
        for mask, label_name in zip(result.masks, result.label_names):
            polygons.append(Polygon(mask=mask, color=self.color_per_label[label_name]))
        return polygons

    def _get_bounding_boxes(self, result: InstanceSegmentationResult) -> list[BoundingBox]:
        bounding_boxes = []
        for bbox, label_name, score in zip(result.bboxes, result.label_names, result.scores):
            x1, y1, x2, y2 = bbox
            label = f"{label_name} ({score:.2f})"
            bounding_boxes.append(
                BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label, color=self.color_per_label[label_name]),
            )
        return bounding_boxes

    def _get_overlays(self, result: InstanceSegmentationResult) -> list[Overlay]:
        overlays = []
        if result.saliency_map is not None and len(result.saliency_map) > 0:
            labels_label_names_mapping = dict(zip(result.labels, result.label_names))
            for label, label_name in labels_label_names_mapping.items():
                saliency_map = result.saliency_map[label - 1]
                saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
                saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
                overlays.append(Overlay(saliency_map, label=f"{label_name.title()} Saliency Map"))
        return overlays

    @property
    def default_layout(self) -> Layout:
        # by default bounding box is not shown.
        return HStack(Flatten(Label, Polygon), Overlay)
