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

    def __init__(self, layout: Layout | None = None) -> None:
        self.layout = layout

    def show(self, image: Image.Image | np.ndarray, result: Result) -> None:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        scene = self._scene_from_result(image, result)
        return scene.show()

    def save(self, image: Image.Image | np.ndarray, result: Result, path: Path) -> None:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        scene = self._scene_from_result(image, result)
        scene.save(path)

    def render(self, image: Image.Image | np.ndarray, result: Result) -> Image.Image | np.ndarray:
        is_numpy = isinstance(image, np.ndarray)

        if is_numpy:
            image = Image.fromarray(image)

        scene = self._scene_from_result(image, result)
        result_img: Image = scene.render()

        if is_numpy:
            return np.array(result_img)

        return result_img

    def _scene_from_result(self, image: Image, result: Result) -> Scene:
        scene: Scene
        if isinstance(result, AnomalyResult):
            scene = AnomalyScene(image, result, self.layout)
        elif isinstance(result, ClassificationResult):
            scene = ClassificationScene(image, result, self.layout)
        elif isinstance(result, InstanceSegmentationResult):
            # Note: This has to be before DetectionScene because InstanceSegmentationResult is a subclass
            # of DetectionResult
            scene = InstanceSegmentationScene(image, result, self.layout)
        elif isinstance(result, ImageResultWithSoftPrediction):
            scene = SegmentationScene(image, result, self.layout)
        elif isinstance(result, DetectionResult):
            scene = DetectionScene(image, result, self.layout)
        elif isinstance(result, DetectedKeypoints):
            scene = KeypointScene(image, result, self.layout)
        else:
            msg = f"Unsupported result type: {type(result)}"
            raise ValueError(msg)

        return scene
