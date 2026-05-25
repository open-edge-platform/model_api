"""Basic tests for comparator policies."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from tests.accuracy.comparator.policies import compare_exact, compare_numeric_close


def test_exact_pass():
    result = compare_exact(5, 5, field_name="x")
    assert result.passed
    assert result.delta is None


def test_exact_fail():
    result = compare_exact(5, 6, field_name="x")
    assert not result.passed
    assert result.policy.value == "exact"


def test_numeric_close_pass():
    result = compare_numeric_close(np.array([1.0, 2.0]), np.array([1.0, 2.005]), field_name="x")
    assert result.passed


def test_numeric_close_fail_beyond_tolerance():
    result = compare_numeric_close(
        np.array([1.0, 2.0, 5.0]),
        np.array([1.0, 2.0, 3.0]),
        field_name="x",
        atol=0.5,
        rtol=0.0,
    )
    assert not result.passed
    assert "delta=" in result.message.lower()
    assert "atol=" in result.message.lower()
    assert "rtol=" in result.message.lower()


def test_numeric_close_shape_mismatch():
    result = compare_numeric_close(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]), field_name="x")
    assert not result.passed
    assert "shape mismatch" in result.message.lower()


def test_numeric_close_none_vs_none():
    result = compare_numeric_close(None, None, field_name="x")
    assert result.passed


def test_numeric_close_none_vs_value():
    result = compare_numeric_close(None, np.array([1.0]), field_name="x")
    assert not result.passed
