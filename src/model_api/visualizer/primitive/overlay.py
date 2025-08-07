"""Overlay primitive."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Union

import numpy as np
import PIL
from PIL import ImageFont

from .primitive import Primitive


class Overlay(Primitive):
    """Overlay primitive.

    Useful for XAI and Anomaly Maps.

    Args:
        image (PIL.Image | np.ndarray): Image to be overlaid.
        label (str | None): Optional label name to overlay.
        opacity (float): Opacity of the overlay.
    """

    def __init__(
        self,
        image: PIL.Image | np.ndarray,
        opacity: float = 0.4,
        label: Union[str, None] = None,
    ) -> None:
        self.image = self._to_pil(image)
        self.label = label
        self.opacity = opacity

    def _to_pil(self, image: PIL.Image | np.ndarray) -> PIL.Image:
        if isinstance(image, np.ndarray):
            return PIL.Image.fromarray(image)
        return image

    def compute(self, image: PIL.Image) -> PIL.Image:
        image_ = self.image.resize(image.size)
        return PIL.Image.blend(image, image_, self.opacity)

    @classmethod
    def overlay_labels(cls, image: PIL.Image, labels: Union[list[str], str, None] = None) -> PIL.Image:
        """Draw labels at the bottom center of the image.

        This is handy when you want to add a label to the image.
        """
        if labels is not None:
            labels = [labels] if isinstance(labels, str) else labels
            font = ImageFont.load_default(size=18)
            buffer_y = 5
            dummy_image = PIL.Image.new("RGB", (1, 1))
            draw = PIL.ImageDraw.Draw(dummy_image)
            textbox = draw.textbbox((0, 0), ", ".join(labels), font=font)
            image_ = PIL.Image.new("RGB", (textbox[2] - textbox[0], textbox[3] + buffer_y - textbox[1]), "white")
            draw = PIL.ImageDraw.Draw(image_)
            draw.text((0, 0), ", ".join(labels), font=font, fill="black")
            image.paste(image_, (image.width // 2 - image_.width // 2, image.height - image_.height - buffer_y))
        return image
