#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for the dataset reader factory."""

from __future__ import annotations

import pytest
from model_converter.datasets import (
    Ade20kReader,
    ClassFolderReader,
    CocoImagesReader,
    reader_for,
)


@pytest.mark.parametrize("dataset_type", ["imagenet-1k", "imagenet-21k", None])
def test_class_folder_used_for_classification_types(tmp_path, dataset_type):
    reader = reader_for(dataset_type, tmp_path)
    assert isinstance(reader, ClassFolderReader)


@pytest.mark.parametrize("dataset_type", ["coco-detection"])
def test_coco_reader_used_for_coco_types(tmp_path, dataset_type):
    reader = reader_for(dataset_type, tmp_path)
    assert isinstance(reader, CocoImagesReader)


def test_ade20k_reader_used_for_ade20k_type(tmp_path):
    reader = reader_for("ade20k", tmp_path)
    assert isinstance(reader, Ade20kReader)


def test_coco_detection_picks_instances_annotation_file(tmp_path):
    reader = reader_for("coco-detection", tmp_path)
    assert reader.annotation_filename == "instances_val2017.json"


def test_unknown_dataset_type_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset_type"):
        reader_for("does-not-exist", tmp_path)
