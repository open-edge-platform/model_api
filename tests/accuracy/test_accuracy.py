#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import ast
import contextlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import onnx
import pytest

from model_api.adapters.onnx_adapter import ONNXRuntimeAdapter
from model_api.adapters.openvino_adapter import OpenvinoAdapter, create_core
from model_api.adapters.utils import load_parameters_from_onnx

# TODO refactor this test so that it does not use eval
# flake8: noqa: F401
from model_api.models import (
    ActionClassificationModel,
    AnomalyDetection,
    AnomalyResult,
    ClassificationModel,
    ClassificationResult,
    DetectedKeypoints,
    DetectionModel,
    DetectionResult,
    ImageModel,
    ImageResultWithSoftPrediction,
    InstanceSegmentationResult,
    KeypointDetectionModel,
    MaskRCNNModel,
    PredictedMask,
    Prompt,
    SAMDecoder,
    SAMImageEncoder,
    SAMLearnableVisualPrompter,
    SAMVisualPrompter,
    SegmentationModel,
    VisualPromptingResult,
    ZSLVisualPromptingResult,
    add_rotated_rects,
    get_contours,
)
from model_api.tilers import (
    DetectionTiler,
    InstanceSegmentationTiler,
    SemanticSegmentationTiler,
)

# Mapping of model type strings to actual classes for security
MODEL_TYPE_MAPPING = {
    "ActionClassificationModel": ActionClassificationModel,
    "AnomalyDetection": AnomalyDetection,
    "ClassificationModel": ClassificationModel,
    "DetectionModel": DetectionModel,
    "ImageModel": ImageModel,
    "KeypointDetectionModel": KeypointDetectionModel,
    "MaskRCNNModel": MaskRCNNModel,
    "SAMDecoder": SAMDecoder,
    "SAMImageEncoder": SAMImageEncoder,
    "SAMLearnableVisualPrompter": SAMLearnableVisualPrompter,
    "SAMVisualPrompter": SAMVisualPrompter,
    "SegmentationModel": SegmentationModel,
    # Tiler classes
    "DetectionTiler": DetectionTiler,
    "InstanceSegmentationTiler": InstanceSegmentationTiler,
    "SemanticSegmentationTiler": SemanticSegmentationTiler,
}


def read_config(fname):
    with fname.open("r") as f:
        return json.load(f)


def create_models(model_type, model_path, download_dir, force_onnx_adapter=False, device="CPU"):
    if model_path.endswith(".onnx") and force_onnx_adapter:
        wrapper_type = model_type.get_model_class(
            load_parameters_from_onnx(onnx.load(model_path))["model_info"]["model_type"],
        )
        model = wrapper_type(
            ONNXRuntimeAdapter(
                model_path,
                ort_options={"providers": ["CPUExecutionProvider"]},
            ),
        )
        model.load()
        return [model]

    models = [
        model_type.create_model(model_path, device=device, download_dir=download_dir),
    ]
    if model_path.endswith(".xml"):
        model = create_core().read_model(model_path)
        if model.has_rt_info(["model_info", "model_type"]):
            wrapper_type = model_type.get_model_class(
                create_core().read_model(model_path).get_rt_info(["model_info", "model_type"]).astype(str),
            )
            model = wrapper_type(OpenvinoAdapter(create_core(), model_path, device=device))
            model.load()
            models.append(model)
    return models


@pytest.fixture(scope="session")
def data(pytestconfig):
    return pytestconfig.getoption("data")


@pytest.fixture(scope="session")
def device(pytestconfig):
    return pytestconfig.getoption("device")


@pytest.fixture(scope="session")
def dump(pytestconfig):
    return pytestconfig.getoption("dump")


@pytest.fixture(scope="session")
def result(pytestconfig):
    return pytestconfig.test_results


@pytest.fixture(scope="session")
def model_data_file(pytestconfig):
    return pytestconfig.getoption("model_data")


def pytest_generate_tests(metafunc):
    if "model_data" in metafunc.fixturenames:
        model_data_file = metafunc.config.getoption("model_data")
        model_data_path = Path(__file__).resolve().parent / model_data_file
        config_data = read_config(model_data_path)
        metafunc.parametrize("model_data", config_data)


def test_image_models(data, device, dump, result, model_data):  # noqa: C901
    name = model_data["name"]
    if name.endswith((".xml", ".onnx")):
        name = f"{data}/{name}"

    for model in create_models(
        MODEL_TYPE_MAPPING[model_data["type"]],
        name,
        data,
        model_data.get("force_ort", False),
        device=device,
    ):
        if "tiler" in model_data:
            if "extra_model" in model_data:
                extra_adapter = OpenvinoAdapter(
                    create_core(),
                    f"{data}/{model_data['extra_model']}",
                    device=device,
                )

                extra_model = MODEL_TYPE_MAPPING[model_data["extra_type"]](
                    extra_adapter,
                    configuration={},
                    preload=True,
                )
                model = MODEL_TYPE_MAPPING[model_data["tiler"]](
                    model,
                    configuration={},
                    tile_classifier_model=extra_model,
                )
            else:
                model = MODEL_TYPE_MAPPING[model_data["tiler"]](model, configuration={})
        elif "prompter" in model_data:
            encoder_adapter = OpenvinoAdapter(
                create_core(),
                f"{data}/{model_data['encoder']}",
                device=device,
            )

            encoder_model = MODEL_TYPE_MAPPING[model_data["encoder_type"]](
                encoder_adapter,
                configuration={},
                preload=True,
            )
            model = MODEL_TYPE_MAPPING[model_data["prompter"]](encoder_model, model)

        if dump:
            result.append(model_data)
            inference_results = []

        for test_data in model_data["test_data"]:
            image_path = Path(data) / test_data["image"]
            image = cv2.imread(str(image_path))
            if image is None:
                error_message = f"Failed to read the image at {image_path}"
                raise RuntimeError(error_message)
            if "input_res" in model_data:
                image = cv2.resize(image, ast.literal_eval(model_data["input_res"]))
            if isinstance(model, ActionClassificationModel):
                image = np.stack([image for _ in range(8)])
            if "prompter" in model_data:
                if model_data["prompter"] == "SAMLearnableVisualPrompter":
                    model.learn(
                        image,
                        points=[
                            Prompt(
                                np.array([image.shape[0] / 2, image.shape[1] / 2]),
                                0,
                            ),
                        ],
                        polygons=[
                            Prompt(
                                np.array(
                                    [
                                        [image.shape[0] / 4, image.shape[1] / 4],
                                        [image.shape[0] / 4, image.shape[1] / 2],
                                        [image.shape[0] / 2, image.shape[1] / 2],
                                    ],
                                ),
                                1,
                            ),
                        ],
                    )
                    outputs = model(image)
                else:
                    outputs = model(
                        image,
                        points=[
                            Prompt(
                                np.array([image.shape[0] / 2, image.shape[1] / 2]),
                                0,
                            ),
                        ],
                    )
            else:
                outputs = model(image)
            if isinstance(outputs, ClassificationResult) or type(outputs) is DetectionResult:
                assert len(test_data["reference"]) == 1
                output_str = str(outputs)
                assert test_data["reference"][0] == output_str
                image_result = [output_str]
            elif isinstance(outputs, ImageResultWithSoftPrediction):
                assert len(test_data["reference"]) == 1
                if hasattr(model, "get_contours"):
                    contours = model.get_contours(outputs)
                else:
                    contours = model.model.get_contours(outputs)
                contour_str = "; "
                for contour in contours:
                    contour_str += str(contour) + ", "
                output_str = str(outputs) + contour_str
                assert test_data["reference"][0] == output_str
                image_result = [output_str]
            elif type(outputs) is InstanceSegmentationResult:
                assert len(test_data["reference"]) == 1
                output_str = str(add_rotated_rects(outputs)) + "; "
                with contextlib.suppress(RuntimeError):
                    # getContours() assumes each instance generates only one contour.
                    # That doesn't hold for some models
                    output_str += "; ".join(str(contour) for contour in get_contours(outputs)) + "; "
                assert test_data["reference"][0] == output_str
                image_result = [output_str]
            elif isinstance(outputs, AnomalyResult):
                assert len(test_data["reference"]) == 1
                output_str = str(outputs)
                assert test_data["reference"][0] == output_str
                image_result = [output_str]
            elif isinstance(outputs, (ZSLVisualPromptingResult, VisualPromptingResult, DetectedKeypoints)):
                output_str = str(outputs)
                assert test_data["reference"][0] == output_str
                image_result = [output_str]
            else:
                pytest.fail(f"Unexpected output type: {type(outputs)}")
            if dump:
                inference_results.append(
                    {"image": test_data["image"], "reference": image_result},
                )
    save_name = Path(name).name if name.endswith(".xml") else name + ".xml"

    if not model_data.get("force_ort", False):
        if "tiler" in model_data:
            model.get_model().save(data + "/serialized/" + save_name)
        elif "prompter" in model_data:
            pass
        else:
            model.save(data + "/serialized/" + save_name)
            if model_data.get("check_extra_rt_info", False):
                assert (
                    create_core()
                    .read_model(data + "/serialized/" + save_name)
                    .get_rt_info(["model_info", "label_ids"])
                    .astype(str)
                )

    if dump:
        result[-1]["test_data"] = inference_results
