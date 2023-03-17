import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from openvino.model_api.models import (
    ClassificationModel,
    DetectionModel,
    SegmentationModel,
)


def process_output(output, model_type):
    if model_type == DetectionModel.__name__:
        return f"{output}"
    elif model_type == ClassificationModel.__name__:
        return f"({output[0]}, {output[1]}, {output[2]:.3f})"
    elif model_type == SegmentationModel.__name__:
        hist = cv2.calcHist(output, [0], None, [256], (0, 256), accumulate=False)
        hist = np.nonzero(hist)[0]
        return f"{hist}"
    else:
        raise ValueError("Unknown model type to precess ouput")


def read_config(path: Path):
    with open(path, "r") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def data(pytestconfig):
    return pytestconfig.getoption("data")


@pytest.fixture(scope="session")
def dump(pytestconfig):
    return pytestconfig.getoption("dump")


@pytest.fixture(scope="session")
def result(pytestconfig):
    return pytestconfig.test_results


@pytest.mark.parametrize(
    ("model_data"), read_config(Path(__file__).resolve().parent / "public_scope.json")
)
def test_image_models(data, dump, result, model_data):
    model = eval(model_data["type"]).create_model(model_data["name"], download_dir=data)

    test_result = []

    if dump:
        result.append(model_data)
        inference_results = []

    for test_data in model_data["test_data"]:
        image_path = Path(data) / "coco128/images/train2017/" / test_data["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError("Failed to read the image")
        outputs = model(image)
        if not isinstance(outputs, list):
            outputs = [outputs]

        image_result = []

        for i, output in enumerate(outputs):
            output_str = process_output(output, model_data["type"])
            if len(test_data["reference"]) > i:
                print(f'{test_data["reference"][i]=}, {output_str=}')
                test_result.append(test_data["reference"][i] == output_str)
            else:
                test_result.append(False)

            if dump:
                image_result.append(output_str)

        if dump:
            inference_results.append(
                {"image": test_data["image"], "reference": image_result}
            )
    if dump:
        result[-1]["test_data"] = inference_results

    assert all(test_result)
