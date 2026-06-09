#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for DatasetRegistry."""

import json

import pytest

from model_converter.dataset_registry import DatasetRegistry


@pytest.fixture
def datasets_config(tmp_path):
    """Create a temporary datasets configuration file."""
    config = {
        "datasets": {
            "imagenet-1k": str(tmp_path / "imagenet"),
            "imagenet-21k": str(tmp_path / "imagenet21k"),
            "coco-detection": str(tmp_path / "coco"),
        },
    }
    config_file = tmp_path / "datasets.json"
    with config_file.open("w") as f:
        json.dump(config, f)
    return config_file


def test_load_valid_config(datasets_config):
    """Test loading a valid datasets configuration."""
    registry = DatasetRegistry(datasets_config)
    assert registry.list_types() == ["coco-detection", "imagenet-1k", "imagenet-21k"]


def test_load_missing_file(tmp_path):
    """Test loading a non-existent configuration file."""
    missing_file = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="Dataset configuration file not found"):
        DatasetRegistry(missing_file)


def test_load_invalid_json(tmp_path):
    """Test loading invalid JSON."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ invalid json }")
    with pytest.raises(ValueError, match="Invalid JSON"):
        DatasetRegistry(bad_json)


def test_load_missing_datasets_key(tmp_path):
    """Test loading configuration without 'datasets' key."""
    config_file = tmp_path / "config.json"
    with config_file.open("w") as f:
        json.dump({"wrong_key": {}}, f)
    with pytest.raises(ValueError, match='Expected JSON with "datasets" key'):
        DatasetRegistry(config_file)


def test_get_path_exists(datasets_config, tmp_path):
    """Test getting path for existing dataset type."""
    registry = DatasetRegistry(datasets_config)
    path = registry.get_path("imagenet-1k")
    assert path == tmp_path / "imagenet"


def test_get_path_missing(datasets_config):
    """Test getting path for non-existent dataset type."""
    registry = DatasetRegistry(datasets_config)
    with pytest.raises(ValueError, match="Dataset type 'missing' not found"):
        registry.get_path("missing")


def test_get_path_validate_exists(datasets_config, tmp_path):
    """Test path validation when dataset directory exists."""
    # Create the directory
    imagenet_dir = tmp_path / "imagenet"
    imagenet_dir.mkdir()

    registry = DatasetRegistry(datasets_config)
    path = registry.get_path("imagenet-1k", validate_exists=True)
    assert path == imagenet_dir


def test_get_path_validate_missing(datasets_config):
    """Test path validation when dataset directory doesn't exist."""
    registry = DatasetRegistry(datasets_config)
    with pytest.raises(FileNotFoundError, match=r"Dataset path.*does not exist"):
        registry.get_path("imagenet-1k", validate_exists=True)


def test_has_type(datasets_config):
    """Test checking if dataset type exists."""
    registry = DatasetRegistry(datasets_config)
    assert registry.has_type("imagenet-1k")
    assert registry.has_type("coco-detection")
    assert not registry.has_type("missing")


def test_list_types(datasets_config):
    """Test listing all dataset types."""
    registry = DatasetRegistry(datasets_config)
    types = registry.list_types()
    assert types == ["coco-detection", "imagenet-1k", "imagenet-21k"]


def test_resolve_from_config_with_dataset_type(datasets_config, tmp_path):
    """Test resolving dataset path from model config."""
    registry = DatasetRegistry(datasets_config)
    config = {"model_short_name": "resnet50", "dataset_type": "imagenet-1k"}
    path = registry.resolve_from_config(config)
    assert path == tmp_path / "imagenet"


def test_resolve_from_config_no_dataset_type(datasets_config):
    """Test resolving when config has no dataset_type."""
    registry = DatasetRegistry(datasets_config)
    config = {"model_short_name": "yolo11"}
    path = registry.resolve_from_config(config)
    assert path is None


def test_resolve_from_config_invalid_type(datasets_config):
    """Test resolving with invalid dataset_type value."""
    registry = DatasetRegistry(datasets_config)
    config = {"model_short_name": "model1", "dataset_type": ""}
    with pytest.raises(ValueError, match="invalid dataset_type"):
        registry.resolve_from_config(config)


def test_resolve_from_config_unknown_type(datasets_config):
    """Test resolving with unknown dataset type."""
    registry = DatasetRegistry(datasets_config)
    config = {"model_short_name": "model1", "dataset_type": "unknown"}
    with pytest.raises(ValueError, match="Dataset type 'unknown' not found"):
        registry.resolve_from_config(config)


def test_load_oserror(tmp_path):
    """Test error when config file cannot be read due to OS error."""
    from unittest.mock import patch

    config_file = tmp_path / "datasets.json"
    config_file.write_text('{"datasets": {}}')

    with (
        patch("pathlib.Path.open", side_effect=OSError("permission denied")),
        pytest.raises(
            ValueError,
            match="Failed to read dataset configuration file",
        ),
    ):
        DatasetRegistry(config_file)


def test_load_datasets_not_dict(tmp_path):
    """Test error when 'datasets' value is not a dictionary."""
    config_file = tmp_path / "datasets.json"
    config_file.write_text('{"datasets": "not-a-dict"}')
    with pytest.raises(ValueError, match="Invalid datasets format"):
        DatasetRegistry(config_file)


def test_load_non_string_path(tmp_path):
    """Test error when a dataset path value is not a string."""
    config_file = tmp_path / "datasets.json"
    config_file.write_text('{"datasets": {"imagenet-1k": 123}}')
    with pytest.raises(ValueError, match="Invalid entry in datasets configuration"):
        DatasetRegistry(config_file)
