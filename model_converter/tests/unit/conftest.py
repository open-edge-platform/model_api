#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Shared fixtures for model_converter unit tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for converted models."""
    return tmp_path / "output"


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Temporary cache directory for downloaded weights."""
    return tmp_path / "cache"


@pytest.fixture
def sample_model_config():
    """Sample torchvision model configuration dictionary."""
    return {
        "model_short_name": "test_model",
        "model_full_name": "Test Model",
        "model_library": "torchvision",
        "model_class_name": "torchvision.models.resnet.resnet18",
        "weights_url": "https://example.com/weights.pth",
        "input_shape": [1, 3, 224, 224],
        "input_names": ["input"],
        "output_names": ["result"],
        "model_type": "Classification",
        "license": "Apache-2.0",
        "license_link": "https://www.apache.org/licenses/LICENSE-2.0",
        "docs": "https://docs.example.com",
        "labels": "IMAGENET1K_V1",
        "mean_values": "123.675 116.28 103.53",
        "scale_values": "58.395 57.12 57.375",
        "reverse_input_channels": True,
        "description": "A test model",
        "dataset_type": "imagenet-1k",
    }


@pytest.fixture
def sample_timm_config():
    """Sample timm model configuration dictionary."""
    return {
        "model_short_name": "test_timm_model",
        "model_full_name": "Test Timm Model",
        "model_library": "timm",
        "huggingface_repo": "timm/resnet50.a1_in1k",
        "huggingface_revision": "abc123",
        "input_shape": [1, 3, 224, 224],
        "input_names": ["input"],
        "output_names": ["result"],
        "model_type": "Classification",
        "license": "Apache-2.0",
        "license_link": "https://www.apache.org/licenses/LICENSE-2.0",
        "docs": "https://docs.example.com",
        "labels": "IMAGENET1K_V1",
        "mean_values": "123.675 116.28 103.53",
        "scale_values": "58.395 57.12 57.375",
        "reverse_input_channels": True,
        "description": "A test timm model",
        "dataset_type": "imagenet-1k",
    }


@pytest.fixture
def facade_converter(tmp_output_dir, tmp_cache_dir):
    """Pre-built facade ModelConverter instance with temporary directories."""
    from model_converter.cli import ModelConverter

    return ModelConverter(
        output_dir=tmp_output_dir,
        cache_dir=tmp_cache_dir,
        verbose=True,
    )


@pytest.fixture
def converter(tmp_output_dir, tmp_cache_dir):
    """Pre-built TorchvisionConverter instance with temporary directories."""
    from model_converter.converters.torchvision import TorchvisionConverter

    return TorchvisionConverter(
        output_dir=tmp_output_dir,
        cache_dir=tmp_cache_dir,
        verbose=True,
    )


@pytest.fixture
def timm_converter(tmp_output_dir, tmp_cache_dir):
    """Pre-built TimmConverter instance with temporary directories."""
    from model_converter.converters.timm import TimmConverter

    return TimmConverter(
        output_dir=tmp_output_dir,
        cache_dir=tmp_cache_dir,
        verbose=True,
    )


@pytest.fixture
def mock_ov_model():
    """Mock OpenVINO model object."""
    model = MagicMock()
    model.inputs = [MagicMock()]
    model.outputs = [MagicMock()]

    input_mock = MagicMock()
    input_mock.get_names.return_value = {"input"}
    model.input.return_value = input_mock
    model.output.return_value = MagicMock()
    model.get_rt_info.return_value = MagicMock(value={"model_type": "Classification"})

    return model


@pytest.fixture
def config_file(tmp_path, sample_model_config):
    """Create a temporary config JSON file."""
    import json

    config = {"models": [sample_model_config]}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path


@pytest.fixture
def mock_torch_model():
    """Mock PyTorch model (nn.Module)."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.load_state_dict = MagicMock()
    return model


@pytest.fixture
def dataset_dir(tmp_path):
    """Create a temporary calibration dataset directory structure."""
    import cv2
    import numpy as np

    dataset_path = tmp_path / "dataset"
    class_dir = dataset_path / "0"
    class_dir.mkdir(parents=True)

    img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.imwrite(str(class_dir / "image_001.jpg"), img)

    class_dir2 = dataset_path / "1"
    class_dir2.mkdir(parents=True)
    cv2.imwrite(str(class_dir2 / "image_002.jpg"), img)

    return dataset_path


@pytest.fixture
def template_dir(tmp_path):
    """Create a temporary template directory with sample templates."""
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "README-timm-fp16.md").write_text("# <<model_name>> (<<variant>>)\nLicense: <<license>>")
    (templates / "README-timm-int8.md").write_text("# <<model_name>> INT8\nLicense: <<license>>")
    (templates / "README-torchvision-fp16.md").write_text("# <<model_name>> (<<variant>>)\nLicense: <<license>>")
    (templates / ".gitattributes").write_text("*.bin filter=lfs diff=lfs merge=lfs -text\n")
    return templates


@pytest.fixture
def datasets_config(tmp_path, dataset_dir):
    """Create a datasets configuration JSON file for testing."""
    import json

    config = {"datasets": {"imagenet-1k": str(dataset_dir), "coco-detection": str(tmp_path / "coco")}}
    config_file = tmp_path / "datasets.json"
    with config_file.open("w") as f:
        json.dump(config, f)
    return config_file


@pytest.fixture
def dataset_registry(datasets_config):
    """Create a DatasetRegistry for testing."""
    from model_converter.dataset_registry import DatasetRegistry

    return DatasetRegistry(datasets_config)


@pytest.fixture
def mock_dataset_registry(dataset_dir):
    """Create a mocked DatasetRegistry for testing."""
    mock = MagicMock()
    mock.resolve_from_config.return_value = dataset_dir
    return mock
