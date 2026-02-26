"""Visualizer utilities."""

from functools import lru_cache

from PIL import ImageFont

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

COLOR_PALETTE = [
    "#FF6B6B",  # Red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Blue
    "#FFA07A",  # Light Salmon
    "#98D8C8",  # Mint
    "#F7DC6F",  # Yellow
    "#BB8FCE",  # Purple
    "#85C1E2",  # Sky Blue
    "#F8B739",  # Orange
    "#52BE80",  # Green
    "#EC7063",  # Coral
    "#5DADE2",  # Light Blue
    "#F39C12",  # Dark Orange
    "#8E44AD",  # Dark Purple
    "#16A085",  # Dark Teal
    "#E74C3C",  # Dark Red
    "#3498DB",  # Dodger Blue
    "#2ECC71",  # Emerald
    "#F1C40F",  # Sun Yellow
    "#E67E22",  # Carrot Orange
]


def get_label_color_mapping(labels: list[str]) -> dict[str, str]:
    """Generate a consistent color mapping for a list of labels.

    Args:
        labels: List of label names.

    Returns:
        Dictionary mapping each label to a hex color string.
    """
    unique_labels = sorted(set(labels))
    return {label: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, label in enumerate(unique_labels)}


@lru_cache(maxsize=5)
def default_font(size: int = 10):
    """Get the default font with the specified size using cache to store the object.

    Args:
        size: Font size.

    Returns:
        A PIL ImageFont instance with the default font and specified size.
    """
    return ImageFont.load_default(size=size)


@lru_cache(maxsize=5)
def truetype_font(font_path: str, size: int = 10):
    """Get a TrueType font from the specified path and size using cache to store the object.

    Args:
        font_path: Path to the .ttf font file.
        size: Font size.
    """

    return ImageFont.truetype(font_path, size)
