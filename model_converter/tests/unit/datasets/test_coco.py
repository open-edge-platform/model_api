#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for the COCO image reader."""

from __future__ import annotations

import json

import pytest

from model_converter.datasets import CocoImagesReader


@pytest.fixture
def coco_root(tmp_path):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    for name in ("000001.jpg", "000002.jpg", "000003.jpg"):
        (images / name).write_bytes(b"")
    coco_payload = {
        "images": [
            {"id": 1, "file_name": "000001.jpg"},
            {"id": 2, "file_name": "000002.jpg"},
            {"id": 3, "file_name": "000003.jpg"},
        ],
        "annotations": [],
        "categories": [],
    }
    (annotations / "instances_val2017.json").write_text(json.dumps(coco_payload))
    return tmp_path


def test_enumerates_every_image_in_images_subdir(coco_root):
    reader = CocoImagesReader(coco_root, annotation_filename="instances_val2017.json")
    samples = list(reader)
    assert {s.image_path.name for s in samples} == {"000001.jpg", "000002.jpg", "000003.jpg"}


def test_labels_are_placeholder_zero(coco_root):
    """COCO calibration does not carry a single classification label; reader uses 0."""
    reader = CocoImagesReader(coco_root, annotation_filename="instances_val2017.json")
    assert all(s.label == 0 for s in reader)


def test_sorted_for_determinism(coco_root):
    reader = CocoImagesReader(coco_root, annotation_filename="instances_val2017.json")
    names = [s.image_path.name for s in reader]
    assert names == sorted(names)


def test_raises_when_images_directory_missing(tmp_path):
    (tmp_path / "annotations").mkdir()
    reader = CocoImagesReader(tmp_path, annotation_filename="instances_val2017.json")
    with pytest.raises(FileNotFoundError, match="images"):
        list(reader)


def test_raises_when_annotation_file_missing(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations").mkdir()
    reader = CocoImagesReader(tmp_path, annotation_filename="instances_val2017.json")
    with pytest.raises(FileNotFoundError, match="annotations"):
        list(reader)
