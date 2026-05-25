"""Tests for noise_floor measurement utility."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from tests.accuracy.comparator.noise_floor import (
    FieldStats,
    NoiseFloorReport,
    load_noise_floor_report,
    measure_noise_floor,
    save_noise_floor_report,
)


def test_deterministic_model_has_zero_std():
    def model_fn(_input):
        return {"a": 1.0, "b": 2.5}

    report = measure_noise_floor(model_fn, None, n_runs=5)
    assert report.n_runs == 5
    assert report.per_field["a"].std == 0.0
    assert report.per_field["a"].cv == 0.0
    assert report.per_field["b"].std == 0.0
    assert report.per_field["b"].cv == 0.0
    assert report.per_field["a"].mean == 1.0
    assert report.per_field["a"].min == 1.0
    assert report.per_field["a"].max == 1.0


def test_noisy_model_has_nonzero_std():
    rng = np.random.RandomState(0)

    def model_fn(_input):
        return {"score": 10.0 + rng.normal(0, 0.5)}

    report = measure_noise_floor(model_fn, None, n_runs=20)
    assert report.per_field["score"].std > 0.0
    assert report.per_field["score"].cv > 0.0
    assert report.per_field["score"].min < report.per_field["score"].max


def test_n_runs_respected():
    counter = {"n": 0}

    def model_fn(_input):
        counter["n"] += 1
        return {"x": float(counter["n"])}

    measure_noise_floor(model_fn, None, n_runs=7)
    assert counter["n"] == 7


def test_round_trip_json(tmp_path):
    def model_fn(_input):
        return {"a": 1.0, "b": 2.0}

    original = measure_noise_floor(model_fn, None, n_runs=3)
    path = tmp_path / "report.json"
    save_noise_floor_report(original, path)
    loaded = load_noise_floor_report(path)

    assert loaded.n_runs == original.n_runs
    assert loaded.summary == original.summary
    assert set(loaded.per_field.keys()) == set(original.per_field.keys())
    for name, stats in original.per_field.items():
        assert isinstance(loaded.per_field[name], FieldStats)
        assert loaded.per_field[name] == stats
    assert isinstance(loaded, NoiseFloorReport)


def test_summary_mentions_noisy_fields():
    rng = np.random.RandomState(42)

    def model_fn(_input):
        return {
            "stable": 5.0,
            "wobbly": 5.0 + rng.normal(0, 1.0),
        }

    report = measure_noise_floor(model_fn, None, n_runs=15)
    assert "wobbly" in report.summary
    assert "stable" not in report.summary


def test_zero_mean_yields_zero_cv():
    def model_fn(_input):
        return {"z": 0.0}

    report = measure_noise_floor(model_fn, None, n_runs=4)
    assert report.per_field["z"].mean == 0.0
    assert report.per_field["z"].cv == 0.0


def test_invalid_n_runs_raises():
    with pytest.raises(ValueError, match="n_runs must be >= 1"):
        measure_noise_floor(lambda _x: {"a": 1.0}, None, n_runs=0)


def test_inconsistent_fields_raises():
    state = {"call": 0}

    def model_fn(_input):
        state["call"] += 1
        if state["call"] == 1:
            return {"a": 1.0, "b": 2.0}
        return {"a": 1.0}

    with pytest.raises(ValueError, match="Inconsistent field names"):
        measure_noise_floor(model_fn, None, n_runs=3)
