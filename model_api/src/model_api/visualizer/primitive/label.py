"""Label primitive."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from typing import Union

from PIL import Image, ImageDraw

from model_api.visualizer.defaults import DEFAULT_FONT_SIZE
from model_api.visualizer.utils import default_font, truetype_font

from .primitive import Primitive


class Label(Primitive):
    """Label primitive.

    Labels require a different processing than other primitives as the class also handles the instance when the layout
    requests all the labels to be drawn on a single image.

    Args:
        label (str): Text of the label.
        score (float | None): Score of the label. This is optional.
        fg_color (str | tuple[int, int, int]): Foreground color of the label.
        bg_color (str | tuple[int, int, int]): Background color of the label.
        font_path (str | None | BytesIO): Path to the font file.
        size (int): Size of the font.

    Examples:
        >>> label = Label(label="Label 1")
        >>> label.compute(image).save("label.jpg")

        >>> label = Label(text="Label 1", fg_color="red", bg_color="blue", font_path="arial.ttf", size=20)
        >>> label.compute(image).save("label.jpg")

        or multiple labels on a single image:
        >>> label1 = Label(text="Label 1")
        >>> label2 = Label(text="Label 2")
        >>> label3 = Label(text="Label 3")
        >>> Label.overlay_labels(image, [label1, label2, label3]).save("labels.jpg")
    """

    def __init__(
        self,
        label: str,
        score: Union[float, None] = None,
        fg_color: Union[str, tuple[int, int, int]] = "black",
        bg_color: Union[str, tuple[int, int, int]] = "yellow",
        font_path: Union[str, BytesIO, None] = None,
        size: int = DEFAULT_FONT_SIZE,
    ) -> None:
        self.label = f"{label} ({score:.2f})" if score is not None else label
        self.fg_color = fg_color
        self.bg_color = bg_color
        self.font = default_font(size=size) if font_path is None else truetype_font(font_path, size)

    def compute(self, image: Image, buffer_y: int = 5) -> Image:
        """Generate label on top of the image.

        Args:
            image (PIL.Image): Image to paste the label on.
            buffer_y (int): Buffer to add to the y-axis of the label.
        """
        label_image = self.generate_label_image(buffer_y)
        image.paste(label_image, (0, 0))
        return image

    def generate_label_image(self, buffer_y: int = 5) -> Image:
        """Generate label image.

        Args:
            buffer_y (int): Buffer to add to the y-axis of the label. This is needed as the text is clipped from the
                bottom otherwise.

        Returns:
            PIL.Image: Image that consists only of the label.
        """
        dummy_image = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_image)
        textbox = draw.textbbox((0, 0), self.label, font=self.font)
        label_image = Image.new("RGB", (textbox[2] - textbox[0], textbox[3] + buffer_y - textbox[1]), self.bg_color)
        draw = ImageDraw.Draw(label_image)
        draw.text((0, 0), self.label, font=self.font, fill=self.fg_color)
        return label_image

    @classmethod
    def overlay_labels(cls, image: Image, labels: list["Label"], buffer_y: int = 5, buffer_x: int = 5) -> Image:
        """Overlay multiple label images on top of the image.
        Paste the labels in a row but wrap the labels if they exceed the image width.

        Args:
            image (PIL.Image): Image to paste the labels on.
            labels (list[Label]): Labels to be pasted on the image.
            buffer_y (int): Buffer to add to the y-axis of the labels.
            buffer_x (int): Space between the labels.

        Returns:
            PIL.Image: Image with the labels pasted on it.
        """
        offset_x = 0
        offset_y = 0
        for label in labels:
            label_image = label.generate_label_image(buffer_y)
            image.paste(label_image, (offset_x, offset_y))
            offset_x += label_image.width + buffer_x
            if offset_x + label_image.width > image.width:
                offset_x = 0
                offset_y += label_image.height
        return image
