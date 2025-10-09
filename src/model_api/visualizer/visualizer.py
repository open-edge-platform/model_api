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
    """Utility class to automatically select the correct scene and render/show it."""

    def __init__(self, layout: Layout | None = None, include_xai: bool = True) -> None:
        self.layout = layout
        self.include_xai = include_xai

    def show(self, image: Image.Image | np.ndarray, result: Result, name: str | None = None, include_xai: bool | None = None) -> None:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(self._convert_bgr_to_rgb(image))
        # Use parameter value if provided, otherwise use instance default
        xai_setting = include_xai if include_xai is not None else self.include_xai
        scene = self._scene_from_result(image, result, include_xai=xai_setting)
        rendered_image = scene.render()
        
        # Add name label in top-left corner if provided
        if name:
            from .primitive import Label
            name_label = Label(label=name, bg_color="white", fg_color="black", size=16)
            rendered_image = name_label.compute(rendered_image.copy())
        
        rendered_image.show()

    def save(self, image: Image.Image | np.ndarray, result: Result, path: Path, name: str | None = None, include_xai: bool | None = None) -> None:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(self._convert_bgr_to_rgb(image))
        # Use parameter value if provided, otherwise use instance default
        xai_setting = include_xai if include_xai is not None else self.include_xai
        scene = self._scene_from_result(image, result, include_xai=xai_setting)
        rendered_image = scene.render()
        
        # Add name label in top-left corner if provided
        if name:
            from .primitive import Label
            name_label = Label(label=name, bg_color="white", fg_color="black", size=16)
            rendered_image = name_label.compute(rendered_image.copy())
        
        rendered_image.save(path)

    def render(self, image: Image.Image | np.ndarray, result: Result, include_xai: bool | None = None) -> Image.Image | np.ndarray:
        is_numpy = isinstance(image, np.ndarray)

        if is_numpy:
            image = Image.fromarray(self._convert_bgr_to_rgb(image))

        # Use parameter value if provided, otherwise use instance default
        xai_setting = include_xai if include_xai is not None else self.include_xai
        scene = self._scene_from_result(image, result, include_xai=xai_setting)
        result_img: Image = scene.render()

        if is_numpy:
            return np.array(result_img)

        return result_img

    def _scene_from_result(self, image: Image, result: Result, include_xai: bool = True) -> Scene:
        scene: Scene
        if isinstance(result, AnomalyResult):
            scene = AnomalyScene(image, result, self.layout, include_xai=include_xai)
        elif isinstance(result, ClassificationResult):
            scene = ClassificationScene(image, result, self.layout, include_xai=include_xai)
        elif isinstance(result, InstanceSegmentationResult):
            # Note: This has to be before DetectionScene because InstanceSegmentationResult is a subclass
            # of DetectionResult
            scene = InstanceSegmentationScene(image, result, self.layout, include_xai=include_xai)
        elif isinstance(result, ImageResultWithSoftPrediction):
            scene = SegmentationScene(image, result, self.layout, include_xai=include_xai)
        elif isinstance(result, DetectionResult):
            scene = DetectionScene(image, result, self.layout, include_xai=include_xai)
        elif isinstance(result, DetectedKeypoints):
            scene = KeypointScene(image, result, self.layout, include_xai=include_xai)
        else:
            msg = f"Unsupported result type: {type(result)}"
            raise ValueError(msg)

        return scene

    def _convert_bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to RGB if it's a 3-channel image.
        
        Args:
            image: Input image array
            
        Returns:
            RGB image array
        """
        # Only convert if it's a 3-channel image (BGR format from OpenCV)
        if len(image.shape) == 3 and image.shape[2] == 3:
            return image[:, :, ::-1]  # Convert BGR to RGB by reversing last dimension
        return image
