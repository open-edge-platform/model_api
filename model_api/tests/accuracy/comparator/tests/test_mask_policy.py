"""Unit tests for MASK_IOU comparator policy."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from tests.accuracy.comparator.policies import (
    compare_binary_mask,
    compare_class_map,
    compare_instance_masks,
)
from tests.accuracy.comparator.storage import (
    save_binary_mask,
    save_class_map,
    save_instance_masks,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_instances() -> np.ndarray:
    masks = np.zeros((3, 32, 32), dtype=np.float32)
    masks[0, 0:10, 0:10] = 1.0
    masks[1, 10:20, 10:20] = 1.0
    masks[2, 20:30, 20:30] = 1.0
    return masks


def test_identical_mask_iou_pass(tmp_path: Path) -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[5:25, 5:25] = True
    p = tmp_path / "ref.png"
    save_binary_mask(p, mask)

    result = compare_binary_mask(mask, p)
    assert result.passed, result.message
    assert "IoU=1.0" in result.message


def test_shifted_mask_fails(tmp_path: Path) -> None:
    ref_mask = np.zeros((128, 128), dtype=bool)
    ref_mask[20:60, 20:60] = True
    p = tmp_path / "ref.png"
    save_binary_mask(p, ref_mask)

    shifted = np.zeros((128, 128), dtype=bool)
    shifted[70:110, 70:110] = True

    result = compare_binary_mask(shifted, p)
    assert not result.passed
    assert "IoU" in result.message
    assert "<" in result.message


def test_instance_shuffle_pass(tmp_path: Path) -> None:
    masks = _make_instances()
    p = tmp_path / "ref.npz"
    save_instance_masks(p, masks)

    shuffled = masks[[2, 0, 1]].copy()
    result = compare_instance_masks(
        shuffled, p, actual_bboxes=None, ref_bboxes=None,
    )
    assert result.passed, result.message
    assert "3 matched pairs" in result.message


def test_dropped_instance_fails(tmp_path: Path) -> None:
    masks = _make_instances()
    p = tmp_path / "ref.npz"
    save_instance_masks(p, masks)

    dropped = masks[:2].copy()
    result = compare_instance_masks(
        dropped, p, actual_bboxes=None, ref_bboxes=None,
    )
    assert not result.passed
    assert "count mismatch" in result.message.lower()
    assert "2" in result.message
    assert "3" in result.message


def test_class_map_missing_class_fails(tmp_path: Path) -> None:
    ref = np.zeros((32, 32), dtype=np.uint16)
    ref[0:16, :] = 1
    ref[16:, :] = 2
    p = tmp_path / "ref.png"
    save_class_map(p, ref)

    actual = np.zeros((32, 32), dtype=np.uint16)
    actual[0:16, :] = 1
    actual[16:, :] = 3

    result = compare_class_map(actual, p)
    assert not result.passed
    assert "class" in result.message.lower()
    assert "IoU=" in result.message
