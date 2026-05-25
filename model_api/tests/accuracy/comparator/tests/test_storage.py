"""Tests for reference bundle storage I/O."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from tests.accuracy.comparator.storage import (
    ReferenceBundle,
    build_generated_by,
    load_reference,
    save_reference,
)

REQUIRED_GENERATED_BY_KEYS = {
    "hostname",
    "os",
    "openvino_version",
    "model_api_version",
    "git_commit",
    "timestamp",
    "python_version",
}


def test_round_trip_preserves_content(tmp_path):
    result_json = {"type": "ClassificationResult", "top_labels": [["cat", 0.9]]}
    metadata = {"test_id": "sample-001", "model": "resnet"}
    generated_by = build_generated_by()
    masks = {
        "binary": (np.eye(16) > 0),
        "classes": np.full((8, 8), 300, dtype=np.uint16),
    }

    ref_dir = tmp_path / "ref"
    save_reference(
        ref_dir,
        result_json=result_json,
        masks=masks,
        metadata=metadata,
        generated_by=generated_by,
    )

    bundle = load_reference(ref_dir)
    assert isinstance(bundle, ReferenceBundle)
    assert bundle.result_json == result_json
    assert bundle.metadata == metadata
    assert bundle.generated_by == generated_by
    assert set(bundle.mask_paths.keys()) == {"binary", "classes"}
    assert bundle.mask_paths["binary"].suffix == ".png"
    assert bundle.mask_paths["classes"].suffix == ".png"


def test_overwrite_protection(tmp_path):
    ref_dir = tmp_path / "ref"
    save_reference(
        ref_dir,
        result_json={"a": 1},
        masks=None,
        metadata={},
        generated_by=build_generated_by(),
    )
    with pytest.raises((FileExistsError, ValueError)):
        save_reference(
            ref_dir,
            result_json={"a": 2},
            masks=None,
            metadata={},
            generated_by=build_generated_by(),
        )


def test_overwrite_allowed_when_flag_set(tmp_path):
    ref_dir = tmp_path / "ref"
    save_reference(
        ref_dir,
        result_json={"a": 1},
        masks=None,
        metadata={},
        generated_by=build_generated_by(),
    )
    save_reference(
        ref_dir,
        result_json={"a": 2},
        masks=None,
        metadata={},
        generated_by=build_generated_by(),
        overwrite=True,
    )
    assert load_reference(ref_dir).result_json == {"a": 2}


def test_generated_by_has_required_keys():
    gb = build_generated_by()
    assert REQUIRED_GENERATED_BY_KEYS.issubset(gb.keys())
    for key in REQUIRED_GENERATED_BY_KEYS:
        assert isinstance(gb[key], str)
        assert gb[key], f"{key} must be a non-empty string"


def test_empty_reference_size(tmp_path):
    ref_dir = tmp_path / "ref"
    save_reference(
        ref_dir,
        result_json={},
        masks=None,
        metadata={},
        generated_by=build_generated_by(),
    )
    total = sum(p.stat().st_size for p in ref_dir.rglob("*") if p.is_file())
    assert total < 2048, f"empty reference exceeded 2 KB: {total} bytes"


def test_instance_masks_saved_as_npz(tmp_path):
    ref_dir = tmp_path / "ref"
    instance_masks = np.zeros((3, 16, 16), dtype=np.bool_)
    instance_masks[0, :8, :8] = True
    instance_masks[1, 8:, 8:] = True
    save_reference(
        ref_dir,
        result_json={},
        masks={"instances": instance_masks},
        metadata={},
        generated_by=build_generated_by(),
    )
    bundle = load_reference(ref_dir)
    assert bundle.mask_paths["instances"].suffix == ".npz"


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_reference(tmp_path / "does-not-exist")
