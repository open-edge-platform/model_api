"""Tests for rotated rect IoU helpers."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from tests.accuracy.comparator.matching import rotated_rect_iou


def test_identical_rect_iou_is_one():
    rect = ((50.0, 50.0), (20.0, 10.0), 30.0)
    assert rotated_rect_iou(rect, rect) == 1.0


def test_disjoint_rects_iou_is_zero():
    a = ((0.0, 0.0), (10.0, 10.0), 0.0)
    b = ((100.0, 100.0), (10.0, 10.0), 0.0)
    assert rotated_rect_iou(a, b) == 0.0


def test_180_degree_equivalence():
    a = ((50.0, 50.0), (20.0, 10.0), 30.0)
    b = ((50.0, 50.0), (20.0, 10.0), 210.0)
    assert rotated_rect_iou(a, b) >= 0.99


def test_wh_swap_90_degree():
    a = ((50.0, 50.0), (20.0, 10.0), 0.0)
    b = ((50.0, 50.0), (10.0, 20.0), 90.0)
    assert rotated_rect_iou(a, b) == 1.0


def test_partial_overlap():
    a = ((50.0, 50.0), (20.0, 10.0), 0.0)
    b = ((58.0, 50.0), (20.0, 10.0), 0.0)
    iou = rotated_rect_iou(a, b)
    assert 0.0 < iou < 1.0
