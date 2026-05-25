"""Tests for the STAT_FINGERPRINT comparator policy."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from tests.accuracy.comparator.fingerprint import compute_fingerprint
from tests.accuracy.comparator.policies import ComparisonPolicy, compare_fingerprint


def _blob_array(seed: int = 0, offset: tuple[int, int] = (20, 20)) -> np.ndarray:
    rng = np.random.RandomState(seed)
    arr = np.zeros((64, 64), dtype="float32")
    y, x = offset
    arr[y : y + 20, x : x + 20] = rng.rand(20, 20).astype("float32")
    return arr


def test_identical_arrays_pass():
    arr = np.random.RandomState(1).rand(64, 64).astype("float32")
    ref_fp = compute_fingerprint(arr)
    result = compare_fingerprint(arr.copy(), ref_fp, field_name="x")
    assert result.passed, result.message
    assert result.policy is ComparisonPolicy.STAT_FINGERPRINT


def test_shifted_saliency_detected():
    base = _blob_array(seed=0, offset=(20, 20))
    shifted = _blob_array(seed=0, offset=(30, 30))  # same blob, +10px shift
    ref_fp = compute_fingerprint(base)
    result = compare_fingerprint(shifted, ref_fp, field_name="saliency")
    assert not result.passed, f"shift should be detected: {result.message}"
    # at least thumbnail or spatial_moments must catch it
    assert ("thumbnail" in result.message and "FAIL" in result.message) or (
        "spatial_moments" in result.message and "FAIL" in result.message
    )


def test_halved_amplitude_fails():
    base = _blob_array(seed=2)
    halved = base * 0.5
    ref_fp = compute_fingerprint(base)
    result = compare_fingerprint(halved, ref_fp, field_name="x")
    assert not result.passed, result.message
    # scalar stats should fail
    assert "[FAIL] mean" in result.message or "[FAIL] std" in result.message or "[FAIL] l2_norm" in result.message


def test_transposed_fails():
    arr = np.arange(64 * 32, dtype="float32").reshape(64, 32)
    ref_fp = compute_fingerprint(arr)
    transposed = arr.T.copy()
    result = compare_fingerprint(transposed, ref_fp, field_name="x")
    assert not result.passed
    assert "shape mismatch" in result.message


def test_none_none_pass():
    result = compare_fingerprint(None, None, field_name="x")
    assert result.passed
    assert result.policy is ComparisonPolicy.STAT_FINGERPRINT


def test_none_present_fail():
    arr = np.ones((4, 4), dtype="float32")
    ref_fp = compute_fingerprint(arr)
    r1 = compare_fingerprint(None, ref_fp, field_name="x")
    r2 = compare_fingerprint(arr, None, field_name="x")
    assert not r1.passed
    assert not r2.passed


def test_list_fingerprint_identical_pass():
    rng = np.random.RandomState(0)
    arrays = [rng.rand(16, 16).astype("float32") for _ in range(3)]
    ref_fp = compute_fingerprint(arrays)
    arrays_copy = [arr.copy() for arr in arrays]
    result = compare_fingerprint(arrays_copy, ref_fp, field_name="saliency_map")
    assert result.passed, result.message
    assert result.policy is ComparisonPolicy.STAT_FINGERPRINT
    assert "all 3 fingerprints OK" in result.message


def test_list_fingerprint_length_mismatch():
    rng = np.random.RandomState(0)
    arrays = [rng.rand(16, 16).astype("float32") for _ in range(3)]
    ref_fp = compute_fingerprint(arrays)
    arrays_short = arrays[:2]
    result = compare_fingerprint(arrays_short, ref_fp, field_name="saliency_map")
    assert not result.passed
    assert "list length mismatch" in result.message


def test_list_fingerprint_one_fails():
    rng = np.random.RandomState(0)
    arrays = [rng.rand(16, 16).astype("float32") for _ in range(3)]
    ref_fp = compute_fingerprint(arrays)
    arrays_mod = [arr.copy() for arr in arrays]
    arrays_mod[1] = arrays_mod[1] * 0.1
    result = compare_fingerprint(arrays_mod, ref_fp, field_name="saliency_map")
    assert not result.passed
    assert "1/3 fingerprints FAILED" in result.message
    assert "indices: [1]" in result.message


def test_list_vs_single_type_mismatch():
    rng = np.random.RandomState(0)
    arr = rng.rand(16, 16).astype("float32")
    arrays = [arr.copy() for _ in range(2)]
    single_fp = compute_fingerprint(arr)
    list_fp = compute_fingerprint(arrays)
    r1 = compare_fingerprint(arrays, single_fp, field_name="x")
    r2 = compare_fingerprint(arr, list_fp, field_name="x")
    assert not r1.passed
    assert "type mismatch" in r1.message
    assert not r2.passed
    assert "type mismatch" in r2.message
