"""Tests for comparator storage helpers."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from tests.accuracy.comparator.storage import (
    compute_class_map_iou,
    compute_mean_iou,
    load_binary_mask,
    load_class_map,
    load_instance_masks,
    save_binary_mask,
    save_class_map,
    save_instance_masks,
)


def test_class_map_roundtrip_preserves_high_class_ids(tmp_path):
    arr = np.zeros((64, 48), dtype=np.uint16)
    arr[10:20, 10:20] = 300
    arr[30:40, 5:15] = 65535
    arr[0, 0] = 1

    path = tmp_path / "cm.png"
    save_class_map(path, arr)
    loaded = load_class_map(path)

    assert loaded.dtype == np.uint16
    assert loaded.shape == arr.shape
    assert np.array_equal(loaded, arr)
    assert loaded[15, 15] == 300


def test_binary_mask_roundtrip(tmp_path):
    rs = np.random.RandomState(42)
    mask = rs.rand(80, 80) > 0.5

    path = tmp_path / "bm.png"
    save_binary_mask(path, mask)
    loaded = load_binary_mask(path)

    assert loaded.dtype == bool
    assert loaded.shape == mask.shape
    assert np.array_equal(loaded, mask)


def test_instance_masks_roundtrip_and_dtype(tmp_path):
    rs = np.random.RandomState(0)
    floats = rs.rand(5, 32, 32).astype(np.float32)
    expected = floats >= 0.5

    path = tmp_path / "im.npz"
    save_instance_masks(path, floats)
    loaded = load_instance_masks(path)

    assert loaded.shape == (5, 32, 32)
    assert loaded.dtype == bool
    assert np.array_equal(loaded, expected)


def test_instance_masks_custom_threshold(tmp_path):
    arr = np.array([[[0.1, 0.9], [0.4, 0.6]]], dtype=np.float32)
    path = tmp_path / "im.npz"
    save_instance_masks(path, arr, threshold=0.7)
    loaded = load_instance_masks(path)
    assert np.array_equal(loaded, np.array([[[False, True], [False, False]]]))


def test_save_class_map_rejects_non_2d(tmp_path):
    with pytest.raises(ValueError, match="2-D"):
        save_class_map(tmp_path / "x.png", np.zeros((2, 2, 2), dtype=np.uint16))


def test_save_instance_masks_rejects_non_3d(tmp_path):
    with pytest.raises(ValueError, match="3-D"):
        save_instance_masks(tmp_path / "x.npz", np.zeros((4, 4), dtype=np.float32))


def test_iou_identical_masks_is_one():
    a = np.zeros((10, 10), dtype=bool)
    a[2:8, 2:8] = True
    assert compute_mean_iou(a, a) == 1.0


def test_iou_disjoint_masks_is_zero():
    a = np.zeros((10, 10), dtype=bool)
    a[0:3, 0:3] = True
    b = np.zeros((10, 10), dtype=bool)
    b[5:8, 5:8] = True
    assert compute_mean_iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = np.zeros((10, 10), dtype=bool)
    a[0:5, 0:5] = True
    b = np.zeros((10, 10), dtype=bool)
    b[3:8, 3:8] = True
    iou = compute_mean_iou(a, b)
    assert iou == pytest.approx(4 / (25 + 25 - 4))


def test_iou_both_empty_returns_one():
    a = np.zeros((4, 4), dtype=bool)
    assert compute_mean_iou(a, a) == 1.0


def test_iou_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_mean_iou(np.zeros((3, 3), dtype=bool), np.zeros((4, 4), dtype=bool))


def test_class_map_iou_identical():
    arr = np.zeros((10, 10), dtype=np.uint16)
    arr[0:5, :] = 1
    arr[5:, :] = 2
    assert compute_class_map_iou(arr, arr) == 1.0


def test_class_map_iou_multi_class():
    pred = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]], dtype=np.uint16)
    ref = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]], dtype=np.uint16)
    iou = compute_class_map_iou(pred, ref)
    iou_0 = 3 / 4
    iou_1 = 4 / 5
    iou_2 = 1.0
    assert iou == pytest.approx((iou_0 + iou_1 + iou_2) / 3)


def test_class_map_iou_skips_absent_classes():
    pred = np.zeros((4, 4), dtype=np.uint16)
    ref = np.zeros((4, 4), dtype=np.uint16)
    assert compute_class_map_iou(pred, ref, num_classes=10) == 1.0
