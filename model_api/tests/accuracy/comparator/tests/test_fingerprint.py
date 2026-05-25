"""Tests for compute_fingerprint."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math

import numpy as np
from tests.accuracy.comparator.fingerprint import compute_fingerprint


def test_fingerprint_2d_float():
    arr = np.random.RandomState(0).rand(256, 256).astype("float32")
    fp = compute_fingerprint(arr)
    assert fp is not None
    for key in (
        "shape", "dtype", "min", "max", "mean", "std", "l2_norm",
        "percentiles", "histogram", "spatial_moments", "argmax_index",
        "thumbnail", "sample_values",
    ):
        assert key in fp, f"missing key: {key}"
    assert fp["shape"] == [256, 256]
    assert fp["dtype"] == "float32"
    assert isinstance(fp["mean"], float)
    assert len(fp["histogram"]) == 16
    assert len(fp["sample_values"]) == 16
    assert len(fp["thumbnail"]) == 16
    assert len(fp["thumbnail"][0]) == 16
    enc = json.dumps(fp, separators=(",", ":"))
    assert len(enc) < 2048, f"fingerprint too large: {len(enc)}"


def test_fingerprint_3d():
    arr = np.random.RandomState(1).rand(3, 32, 32).astype("float32")
    fp = compute_fingerprint(arr)
    assert fp is not None
    assert fp["shape"] == [3, 32, 32]
    assert fp["spatial_moments"] is not None
    assert set(fp["spatial_moments"].keys()) == {"cx", "cy"}
    assert fp["thumbnail"] is not None
    assert len(fp["thumbnail"]) == 16
    assert len(fp["thumbnail"][0]) == 16


def test_fingerprint_1d():
    arr = np.arange(100, dtype="float32")
    fp = compute_fingerprint(arr)
    assert fp is not None
    assert fp["shape"] == [100]
    assert fp["spatial_moments"] is None
    assert fp["thumbnail"] is None
    assert len(fp["sample_values"]) == 16


def test_fingerprint_none():
    assert compute_fingerprint(None) is None


def test_fingerprint_empty():
    arr = np.zeros((0, 5), dtype="float32")
    fp = compute_fingerprint(arr)
    assert fp is not None
    assert fp["shape"] == [0, 5]
    assert fp["min"] is None
    assert fp["histogram"] == []
    assert fp["thumbnail"] is None
    assert fp["sample_values"] == []


def test_fingerprint_constant_array():
    arr = np.full((10, 10), math.pi, dtype="float32")
    fp = compute_fingerprint(arr)
    assert fp is not None
    assert fp["histogram"][0] == 100
    assert sum(fp["histogram"][1:]) == 0


def test_fingerprint_deterministic():
    arr = np.random.RandomState(42).rand(20, 20).astype("float32")
    fp1 = compute_fingerprint(arr)
    fp2 = compute_fingerprint(arr.copy())
    assert fp1 == fp2


def test_fingerprint_list_of_arrays():
    rng = np.random.RandomState(0)
    arrays = [rng.rand(16, 16).astype("float32") for _ in range(3)]
    fp = compute_fingerprint(arrays)
    assert isinstance(fp, list)
    assert len(fp) == 3
    for i, sub_fp in enumerate(fp):
        assert sub_fp is not None
        assert sub_fp["shape"] == [16, 16]
        assert sub_fp["dtype"] == "float32"


def test_fingerprint_empty_list():
    fp = compute_fingerprint([])
    assert fp == []
