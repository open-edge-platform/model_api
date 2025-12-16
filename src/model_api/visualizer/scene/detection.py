"""Detection Scene."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import cv2
from PIL import Image

from model_api.models.result import DetectionResult
from model_api.visualizer.layout import Flatten, HStack, Layout
from model_api.visualizer.primitive import BoundingBox, Label, Overlay

from .scene import Scene


class DetectionScene(Scene):
    """Detection Scene."""

    # Color palette for different labels (RGB tuples)
    COLOR_PALETTE = [
        (255, 56, 56),    # Red
        (56, 255, 56),    # Green
        (56, 56, 255),    # Blue
        (255, 255, 56),   # Yellow
        (255, 56, 255),   # Magenta
        (56, 255, 255),   # Cyan
        (255, 128, 0),    # Orange
        (255, 0, 128),    # Pink
        (128, 255, 0),    # Lime
        (0, 255, 128),    # Spring Green
        (128, 0, 255),    # Purple
        (0, 128, 255),    # Sky Blue
        (255, 153, 51),   # Light Orange
        (153, 51, 255),   # Violet
        (51, 255, 153),   # Mint
        (255, 204, 51),   # Gold
        (204, 51, 255),   # Purple Pink
        (51, 204, 255),   # Light Blue
        (255, 102, 102),  # Light Red
        (102, 255, 102),  # Light Green
    ]

    def __init__(self, image: Image, result: DetectionResult, layout: Union[Layout, None] = None) -> None:
        super().__init__(
            base=image,
            bounding_box=self._get_bounding_boxes(result),
            overlay=self._get_overlays(result),
            layout=layout,
        )

    def _get_overlays(self, result: DetectionResult) -> list[Overlay]:
        overlays = []
        # Add only the overlays that are predicted
        label_index_mapping = dict(zip(result.labels, result.label_names))
        for label_index, label_name in label_index_mapping.items():
            # Index 0 as it assumes only one batch
            if result.saliency_map is not None and result.saliency_map.size > 0:
                saliency_map = cv2.applyColorMap(result.saliency_map[0][label_index], cv2.COLORMAP_JET)
                saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
                overlays.append(Overlay(saliency_map, label=label_name.title()))
        return overlays

    def _get_bounding_boxes(self, result: DetectionResult) -> list[BoundingBox]:
        bounding_boxes = []
        for score, label_name, label_idx, bbox in zip(result.scores, result.label_names, result.labels, result.bboxes):
            x1, y1, x2, y2 = bbox
            label = f"{label_name} ({score:.2f})"
            # Use label index to select color from palette (wrap around if more labels than colors)
            color = self.COLOR_PALETTE[label_idx % len(self.COLOR_PALETTE)]
            bounding_boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label, color=color))
        return bounding_boxes

    @property
    def default_layout(self) -> Layout:
        return HStack(Flatten(BoundingBox, Label), Overlay)
