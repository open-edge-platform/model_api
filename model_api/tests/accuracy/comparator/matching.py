"""Field matching helpers for future comparator dispatch."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def _bbox_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized IoU matrix of shape (N_a, N_b) for boxes in [x1, y1, x2, y2] format."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)

    a = a.astype(float)
    b = b.astype(float)

    a_exp = a[:, None, :]  # (N_a, 1, 4)
    b_exp = b[None, :, :]  # (1, N_b, 4)

    inter_x1 = np.maximum(a_exp[..., 0], b_exp[..., 0])
    inter_y1 = np.maximum(a_exp[..., 1], b_exp[..., 1])
    inter_x2 = np.minimum(a_exp[..., 2], b_exp[..., 2])
    inter_y2 = np.minimum(a_exp[..., 3], b_exp[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
    inter = inter_w * inter_h

    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def _hungarian_match(
    iou_matrix: np.ndarray,
    n_pred: int,
    n_ref: int,
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Run Hungarian assignment on an IoU matrix and split out unmatched indices."""
    if n_pred == 0 or n_ref == 0:
        return [], list(range(n_pred)), list(range(n_ref))

    cost = 1.0 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    matched: list[tuple[int, int]] = []
    matched_pred: set[int] = set()
    matched_ref: set[int] = set()
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matched.append((int(r), int(c)))
            matched_pred.add(int(r))
            matched_ref.add(int(c))

    unmatched_pred = [i for i in range(n_pred) if i not in matched_pred]
    unmatched_ref = [i for i in range(n_ref) if i not in matched_ref]
    return matched, unmatched_pred, unmatched_ref


def match_by_bbox_iou(
    pred_bboxes: np.ndarray,
    ref_bboxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching on bbox IoU.

    Returns (matched_pairs, unmatched_pred_indices, unmatched_ref_indices).
    matched_pairs is a list of (pred_idx, ref_idx).
    """
    n_pred = int(pred_bboxes.shape[0]) if pred_bboxes.ndim >= 1 else 0
    n_ref = int(ref_bboxes.shape[0]) if ref_bboxes.ndim >= 1 else 0
    if n_pred == 0 or n_ref == 0:
        return [], list(range(n_pred)), list(range(n_ref))

    iou = _bbox_iou_matrix(pred_bboxes, ref_bboxes)
    return _hungarian_match(iou, n_pred, n_ref, iou_threshold)


def match_by_mask_iou(
    pred_masks: np.ndarray,
    ref_masks: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching on mask IoU for (N, H, W) boolean arrays."""
    n_pred = int(pred_masks.shape[0]) if pred_masks.ndim >= 1 else 0
    n_ref = int(ref_masks.shape[0]) if ref_masks.ndim >= 1 else 0
    if n_pred == 0 or n_ref == 0:
        return [], list(range(n_pred)), list(range(n_ref))

    p = pred_masks.astype(bool).reshape(n_pred, -1)
    r = ref_masks.astype(bool).reshape(n_ref, -1)

    inter = (p.astype(np.uint32) @ r.astype(np.uint32).T).astype(float)
    p_area = p.sum(axis=1).astype(float)
    r_area = r.sum(axis=1).astype(float)
    union = p_area[:, None] + r_area[None, :] - inter
    iou = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)

    return _hungarian_match(iou, n_pred, n_ref, iou_threshold)


def rotated_rect_iou(a: tuple, b: tuple) -> float:
    """IoU between two rotated rects in cv2 RotatedRect format: ((cx,cy),(w,h),angle_deg).

    Uses cv2.rotatedRectangleIntersection for polygon intersection.
    """
    area_a = float(a[1][0]) * float(a[1][1])
    area_b = float(b[1][0]) * float(b[1][1])

    ret_val, intersection_pts = cv2.rotatedRectangleIntersection(a, b)
    if ret_val == cv2.INTERSECT_NONE:
        return 0.0
    if ret_val == cv2.INTERSECT_FULL:
        return min(area_a, area_b) / max(area_a, area_b, 1e-9)

    if intersection_pts is None or len(intersection_pts) == 0:
        return 0.0

    intersection_area = float(cv2.contourArea(intersection_pts))
    union = area_a + area_b - intersection_area
    return intersection_area / max(union, 1e-9)


def match_by_rotated_iou(
    pred_rects: list,
    ref_rects: list,
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching for rotated rects. Same return signature as match_by_bbox_iou."""
    n_pred = len(pred_rects)
    n_ref = len(ref_rects)
    if n_pred == 0 or n_ref == 0:
        return [], list(range(n_pred)), list(range(n_ref))

    iou = np.zeros((n_pred, n_ref), dtype=float)
    for i, pred_rect in enumerate(pred_rects):
        for j, ref_rect in enumerate(ref_rects):
            iou[i, j] = rotated_rect_iou(pred_rect, ref_rect)

    return _hungarian_match(iou, n_pred, n_ref, iou_threshold)
