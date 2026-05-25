"""Tests for Hungarian matching helpers."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from tests.accuracy.comparator.matching import match_by_bbox_iou, match_by_mask_iou


def _boxes() -> np.ndarray:
    return np.array(
        [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
        dtype=float,
    )


def test_identity_match():
    pred = _boxes()
    ref = pred.copy()
    pairs, up, ur = match_by_bbox_iou(pred, ref)
    assert sorted(pairs) == [(0, 0), (1, 1), (2, 2)]
    assert up == []
    assert ur == []


def test_shuffled_match():
    ref = _boxes()
    pred = ref[::-1].copy()
    pairs, up, ur = match_by_bbox_iou(pred, ref)
    assert sorted(pairs) == [(0, 2), (1, 1), (2, 0)]
    assert up == []
    assert ur == []


def test_partial_match():
    pred = np.array(
        [[0, 0, 10, 10], [100, 100, 110, 110], [40, 40, 50, 50]],
        dtype=float,
    )
    ref = np.array(
        [[0, 0, 10, 10], [40, 40, 50, 50], [200, 200, 210, 210]],
        dtype=float,
    )
    pairs, up, ur = match_by_bbox_iou(pred, ref, iou_threshold=0.5)
    assert sorted(pairs) == [(0, 0), (2, 1)]
    assert up == [1]
    assert ur == [2]


def test_empty_pred():
    pred = np.zeros((0, 4), dtype=float)
    ref = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=float)
    pairs, up, ur = match_by_bbox_iou(pred, ref)
    assert pairs == []
    assert up == []
    assert ur == [0, 1]


def test_empty_ref():
    pred = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=float)
    ref = np.zeros((0, 4), dtype=float)
    pairs, up, ur = match_by_bbox_iou(pred, ref)
    assert pairs == []
    assert up == [0, 1]
    assert ur == []


def test_all_unmatched_below_threshold():
    pred = np.array([[0, 0, 10, 10]], dtype=float)
    ref = np.array([[100, 100, 110, 110]], dtype=float)
    pairs, up, ur = match_by_bbox_iou(pred, ref, iou_threshold=0.5)
    assert pairs == []
    assert up == [0]
    assert ur == [0]


def test_mask_iou_shuffled_match():
    masks = np.zeros((3, 20, 20), dtype=bool)
    masks[0, 0:5, 0:5] = True
    masks[1, 10:15, 10:15] = True
    masks[2, 15:20, 0:5] = True
    pred = masks[::-1].copy()
    pairs, up, ur = match_by_mask_iou(pred, masks)
    assert sorted(pairs) == [(0, 2), (1, 1), (2, 0)]
    assert up == []
    assert ur == []
