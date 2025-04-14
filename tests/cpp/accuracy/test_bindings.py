#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import json
from pathlib import Path

import cv2

from model_api.models import Model
from py_model_api import ClassificationModel


def read_config(models_config: str, model_type: str):
    with open(models_config, "r") as f:
        data = json.load(f)
        for item in data:
            if item["type"] == model_type:
                yield item


@pytest.fixture(scope="session")
def data(pytestconfig) -> str:
    return pytestconfig.getoption("data")


@pytest.fixture(scope="session")
def models_config(pytestconfig) -> str:
    return pytestconfig.getoption("config")


@pytest.fixture()
def classification_configs(models_config: str):
    return read_config(models_config, "ClassificationModel")


def test_classification_models(data: str, classification_configs):
    for model_data in classification_configs:
        name = model_data["name"]
        if ".xml" not in name:
            continue
        if name.endswith(".xml") or name.endswith(".onnx"):
            name = f"{data}/{name}"

        model = Model.create_model(name, preload=True)
        cpp_model = ClassificationModel.create_model(name, preload=True)

        image_path = Path(data) / next(iter(model_data["test_data"]))["image"]
        image = cv2.imread(str(image_path))

        py_result = model(image)
        cpp_result = cpp_model(image)

        assert str(py_result) == str(cpp_result)
