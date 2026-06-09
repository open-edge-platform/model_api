#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for the ADE20K reader."""

from __future__ import annotations

import pytest

from model_converter.datasets import Ade20kReader


@pytest.fixture
def ade20k_root(tmp_path):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    for stem in ("ADE_val_00000001", "ADE_val_00000002", "ADE_val_00000003"):
        (images / f"{stem}.jpg").write_bytes(b"")
        (annotations / f"{stem}.png").write_bytes(b"")
    return tmp_path


def test_enumerates_all_image_mask_pairs(ade20k_root):
    reader = Ade20kReader(ade20k_root)
    samples = list(reader)
    assert {s.image_path.name for s in samples} == {
        "ADE_val_00000001.jpg",
        "ADE_val_00000002.jpg",
        "ADE_val_00000003.jpg",
    }


def test_labels_are_placeholder_zero(ade20k_root):
    reader = Ade20kReader(ade20k_root)
    assert all(s.label == 0 for s in reader)


def test_skips_images_without_matching_mask(ade20k_root):
    # Add an unmatched image
    (ade20k_root / "images" / "ADE_val_00000004.jpg").write_bytes(b"")
    reader = Ade20kReader(ade20k_root)
    names = [s.image_path.name for s in reader]
    assert "ADE_val_00000004.jpg" not in names
    assert len(names) == 3


def test_raises_when_images_directory_missing(tmp_path):
    (tmp_path / "annotations").mkdir()
    reader = Ade20kReader(tmp_path)
    with pytest.raises(FileNotFoundError, match="images"):
        list(reader)


def test_raises_when_annotations_directory_missing(tmp_path):
    (tmp_path / "images").mkdir()
    reader = Ade20kReader(tmp_path)
    with pytest.raises(FileNotFoundError, match="annotations"):
        list(reader)


def test_results_sorted_by_image_name(ade20k_root):
    reader = Ade20kReader(ade20k_root)
    names = [s.image_path.name for s in reader]
    assert names == sorted(names)


def test_mask_path_populated_for_each_sample(ade20k_root):
    reader = Ade20kReader(ade20k_root)
    for sample in reader:
        assert sample.mask_path is not None
        assert sample.mask_path == ade20k_root / "annotations" / f"{sample.image_path.stem}.png"
        assert sample.mask_path.exists()
