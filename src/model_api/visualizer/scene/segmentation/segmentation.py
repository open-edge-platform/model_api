"""Segmentation Scene."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import cv2
import numpy as np
from PIL import Image

from model_api.models.result import ImageResultWithSoftPrediction
from model_api.visualizer.layout import HStack, Layout
from model_api.visualizer.primitive import Overlay
from model_api.visualizer.scene import Scene


class SegmentationScene(Scene):
    """Segmentation Scene."""

    def __init__(self, image: Image, result: ImageResultWithSoftPrediction, layout: Union[Layout, None] = None) -> None:
        super().__init__(
            base=image,
            overlay=self._get_overlays(result),
            layout=layout,
        )

    def _get_overlays(self, result: ImageResultWithSoftPrediction) -> list[Overlay]:
        overlays = []
        # Use the hard prediction to get the overlays
        hard_prediction = result.resultImage  # shape H,W
        num_classes = hard_prediction.max()
        
        # Create a single colored segmentation map with all classes
        h, w = hard_prediction.shape
        colored_segmentation = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Generate distinct colors for each class using HSV color space
        for i in range(1, num_classes + 1):  # ignore background (class 0)
            class_mask = (hard_prediction == i)
            # Generate a distinct color for each class using HSV
            hue = int(180 * i / num_classes)  # Distribute hues across the spectrum
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0]
            colored_segmentation[class_mask] = color_rgb
        
        overlays.append(Overlay(colored_segmentation, label="Segmentation"))

        # Add saliency map
        if result.saliency_map is not None and result.saliency_map.size > 0:
            saliency_map = cv2.cvtColor(result.saliency_map, cv2.COLOR_BGR2RGB)
            overlays.append(Overlay(saliency_map, label="Saliency Map"))

        return overlays

    @property
    def default_layout(self) -> Layout:
        return HStack(Overlay)
