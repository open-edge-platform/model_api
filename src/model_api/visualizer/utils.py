"""Utility functions for the visualizer."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import hashlib
import random
from typing import Dict, List


def generate_color_palette(labels: List[str], seed: int = 0) -> Dict[str, str]:
    """Generate a consistent color palette for a list of labels.
    
    Uses label names as seeds to ensure the same label always gets the same color,
    regardless of which other labels are present in the current detection.
    
    Args:
        labels: List of label names
        seed: Base seed for color generation (optional, for additional randomization)
        
    Returns:
        Dictionary mapping label names to hex color strings
    """
    color_map = {}
    for label in labels:
        # Create a deterministic seed based on the label name
        label_hash = hashlib.md5(label.encode()).hexdigest()
        label_seed = int(label_hash[:8], 16) + seed
        
        # Generate color using the label-specific seed
        g = random.Random(label_seed)  # noqa: S311 # nosec B311
        color_map[label] = f"#{g.randint(0, 0xFFFFFF):06x}"  # nosec B311
    
    return color_map


def get_distinct_colors(num_colors: int, seed: int = 0) -> List[str]:
    """Generate a list of distinct colors.
    
    Args:
        num_colors: Number of colors to generate
        seed: Random seed for reproducible colors
        
    Returns:
        List of hex color strings
    """
    g = random.Random(seed)  # noqa: S311 # nosec B311
    return [f"#{g.randint(0, 0xFFFFFF):06x}" for _ in range(num_colors)]  # nosec B311