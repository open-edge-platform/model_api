#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for ClassFolderReader."""

from __future__ import annotations

import pytest
from model_converter.datasets import ClassFolderReader


@pytest.fixture
def class_folder_root(tmp_path):
    """Build a minimal class-folder dataset (3 classes, 2 images each)."""
    for class_id in (0, 1, 5):
        class_dir = tmp_path / str(class_id)
        class_dir.mkdir()
        (class_dir / "img_a.JPEG").write_bytes(b"")
        (class_dir / "img_b.png").write_bytes(b"")
    return tmp_path


def test_enumerates_all_images_with_integer_labels(class_folder_root):
    reader = ClassFolderReader(class_folder_root)
    samples = list(reader)

    assert len(samples) == 6
    labels = sorted({s.label for s in samples})
    assert labels == [0, 1, 5]
    assert all(s.image_path.exists() for s in samples)


def test_label_matches_class_folder_name(class_folder_root):
    reader = ClassFolderReader(class_folder_root)
    for sample in reader:
        assert int(sample.image_path.parent.name) == sample.label


def test_ignores_non_directory_entries_at_root(class_folder_root):
    (class_folder_root / "stray_file.txt").write_text("noise")
    reader = ClassFolderReader(class_folder_root)
    samples = list(reader)
    assert len(samples) == 6  # stray file ignored


def test_supports_multiple_image_extensions(class_folder_root):
    (class_folder_root / "0" / "img_c.jpg").write_bytes(b"")
    reader = ClassFolderReader(class_folder_root)
    samples = list(reader)
    assert len(samples) == 7


def test_raises_when_class_folder_name_not_integer(tmp_path):
    bad = tmp_path / "not_an_int"
    bad.mkdir()
    (bad / "img.jpg").write_bytes(b"")
    reader = ClassFolderReader(tmp_path)
    with pytest.raises(ValueError, match="invalid literal"):
        list(reader)


def test_empty_root_returns_no_samples(tmp_path):
    reader = ClassFolderReader(tmp_path)
    assert list(reader) == []


def test_nonexistent_root_returns_no_samples(tmp_path):
    reader = ClassFolderReader(tmp_path / "does_not_exist")
    assert list(reader) == []
