"""Bounding box primitive."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from PIL import Image, ImageDraw

from model_api.visualizer.defaults import DEFAULT_FONT_SIZE, DEFAULT_OUTLINE_WIDTH
from model_api.visualizer.utils import default_font

from .primitive import Primitive


class BoundingBox(Primitive):
    """Bounding box primitive.

    Args:
        x1 (int): x-coordinate of the top-left corner of the bounding box.
        y1 (int): y-coordinate of the top-left corner of the bounding box.
        x2 (int): x-coordinate of the bottom-right corner of the bounding box.
        y2 (int): y-coordinate of the bottom-right corner of the bounding box.
        label (str | None): Label of the bounding box.
        color (str | tuple[int, int, int]): Color of the bounding box.
        outline_width (int): Width of the bounding box outline.
        font_size (int): Font size for the label text.

    Example:
        >>> bounding_box = BoundingBox(x1=10, y1=10, x2=100, y2=100, label="Label Name")
        >>> bounding_box.compute(image)
    """

    def __init__(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        label: str | None = None,
        color: str | tuple[int, int, int] = "blue",
        outline_width: int = DEFAULT_OUTLINE_WIDTH,
        font_size: int = DEFAULT_FONT_SIZE,
    ) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label
        self.color = color
        self.outline_width = outline_width
        self.font_size = font_size
        self.font = default_font(size=self.font_size)
        self.y_buffer = max(3, font_size // 3)  # Text at the bottom of the text box is clipped. This prevents that.

    def compute(self, image: Image) -> Image:
        draw = ImageDraw.Draw(image)
        # draw rectangle
        draw.rectangle((self.x1, self.y1, self.x2, self.y2), outline=self.color, width=self.outline_width)
        # add label
        if self.label:
            # draw the background of the label
            textbox = draw.textbbox((0, 0), self.label, font=self.font)
            label_image = Image.new(
                "RGB",
                (textbox[2] - textbox[0], textbox[3] + self.y_buffer - textbox[1]),
                self.color,
            )
            draw = ImageDraw.Draw(label_image)
            # write the label on the background
            draw.text((0, 0), self.label, font=self.font, fill="white")
            image.paste(label_image, (self.x1, self.y1))
        return image
