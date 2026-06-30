"""Visualizer for modelAPI."""

# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from model_api.models.result import (
    AnomalyResult,
    ClassificationResult,
    DetectedKeypoints,
    DetectionResult,
    ImageResultWithSoftPrediction,
    InstanceSegmentationResult,
    Result,
)

from .defaults import SCALE_BASELINE
from .scene import (
    AnomalyScene,
    ClassificationScene,
    DetectionScene,
    InstanceSegmentationScene,
    KeypointScene,
    Scene,
    SegmentationScene,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .layout import Layout


class Visualizer:
    """Utility class to automatically select the correct scene and render/show it.

    Args:
        layout: Optional layout to use for rendering.
        auto_scale: When True, drawing sizes (line widths, font sizes, etc.) are
            automatically scaled relative to 720p so that annotations remain
            visible on high-resolution images.  Defaults to True.
    """

    def __init__(self, layout: Layout | None = None, auto_scale: bool = True) -> None:
        self.layout = layout
        self.auto_scale = auto_scale

    @staticmethod
    def compute_scale_factor(image: Image.Image) -> float:
        """Compute a scale factor based on the image's longer edge relative to 720p (1280px).

        Returns 1.0 for images ≤ 720p; for larger images the factor grows proportionally.

        Args:
            image: PIL Image whose dimensions determine the scale.

        Returns:
            Scale factor (>= 1.0).
        """
        longer_edge = max(image.width, image.height)
        return max(1.0, longer_edge / SCALE_BASELINE)

    @staticmethod
    def _to_rgb(image: np.ndarray | Image.Image) -> Image.Image:
        """Convert an image to RGB PIL Image.

        Handles grayscale images (both PIL mode 'L' and 2D numpy arrays) by converting
        them to RGB. This ensures compatibility with PIL.Image.blend() which requires
        both images to have the same mode.

        Args:
            image: Input image as numpy array or PIL Image.

        Returns:
            PIL Image in RGB mode.
        """
        if isinstance(image, np.ndarray):
            # Handle single-channel 3D arrays (H, W, 1) by squeezing to 2D
            if image.ndim == 3 and image.shape[2] == 1:
                image = image.squeeze(axis=2)
            image = Image.fromarray(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def show(self, image: Image.Image | np.ndarray, result: Result) -> None:
        image = self._to_rgb(image)
        scene = self._scene_from_result(image, result)
        return scene.show()

    def save(self, image: Image.Image | np.ndarray, result: Result, path: Path) -> None:
        image = self._to_rgb(image)
        scene = self._scene_from_result(image, result)
        scene.save(path)

    def render(self, image: Image.Image | np.ndarray, result: Result) -> Image.Image | np.ndarray:
        is_numpy = isinstance(image, np.ndarray)

        image = self._to_rgb(image)

        scene = self._scene_from_result(image, result)
        result_img: Image = scene.render()

        if is_numpy:
            return np.array(result_img)

        return result_img

    def _scene_from_result(self, image: Image, result: Result) -> Scene:
        scale = self.compute_scale_factor(image) if self.auto_scale else 1.0

        scene: Scene
        if isinstance(result, AnomalyResult):
            scene = AnomalyScene(image, result, self.layout, scale=scale)
        elif isinstance(result, ClassificationResult):
            scene = ClassificationScene(image, result, self.layout, scale=scale)
        elif isinstance(result, InstanceSegmentationResult):
            # Note: This has to be before DetectionScene because InstanceSegmentationResult is a subclass
            # of DetectionResult
            scene = InstanceSegmentationScene(image, result, self.layout, scale=scale)
        elif isinstance(result, ImageResultWithSoftPrediction):
            scene = SegmentationScene(image, result, self.layout, scale=scale)
        elif isinstance(result, DetectionResult):
            scene = DetectionScene(image, result, self.layout, scale=scale)
        elif isinstance(result, DetectedKeypoints):
            scene = KeypointScene(image, result, self.layout, scale=scale)
        else:
            msg = f"Unsupported result type: {type(result)}"
            raise ValueError(msg)

        return scene
