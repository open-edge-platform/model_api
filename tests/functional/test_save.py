#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path

import onnx

from model_api.adapters import ONNXRuntimeAdapter
from model_api.adapters.utils import load_parameters_from_onnx
from model_api.models import Model


def test_detector_save(tmp_path, data):
    detector = Model.create_model(
        Path(data) / "otx_models/detection_model_with_xai_head.xml",
    )
    xml_path = str(tmp_path / "a.xml")
    detector.save(xml_path)
    deserialized = Model.create_model(xml_path)

    assert deserialized.get_model().get_rt_info(["model_info", "embedded_processing"]).astype(bool)
    assert type(detector) is type(deserialized)
    for attr in detector.parameters():
        assert getattr(detector, attr) == getattr(deserialized, attr)


def test_classifier_save(tmp_path, data):
    classifier = Model.create_model(
        Path(data) / "otx_models/tinynet_imagenet.xml",
    )
    xml_path = str(tmp_path / "a.xml")
    classifier.save(xml_path)
    deserialized = Model.create_model(xml_path)

    assert deserialized.get_model().get_rt_info(["model_info", "embedded_processing"]).astype(bool)
    assert type(classifier) is type(deserialized)
    for attr in classifier.parameters():
        assert getattr(classifier, attr) == getattr(deserialized, attr)


def test_segmentor_save(tmp_path, data):
    segmenter = Model.create_model(
        Path(data) / "otx_models/Lite-hrnet-18_mod2.xml",
    )
    xml_path = str(tmp_path / "a.xml")
    segmenter.save(xml_path)
    deserialized = Model.create_model(xml_path)

    assert deserialized.get_model().get_rt_info(["model_info", "embedded_processing"]).astype(bool)
    assert type(segmenter) is type(deserialized)
    for attr in segmenter.parameters():
        assert getattr(segmenter, attr) == getattr(deserialized, attr)


def test_onnx_save(tmp_path, data):
    cls_model = Model.create_model(
        ONNXRuntimeAdapter(Path(data) / "otx_models/cls_mobilenetv3_large_cars.onnx"),
        model_type="Classification",
        preload=True,
        configuration={"reverse_input_channels": True, "topk": 6},
    )

    onnx_path = str(tmp_path / "a.onnx")
    cls_model.save(onnx_path)
    deserialized = Model.create_model(onnx_path)

    assert load_parameters_from_onnx(onnx.load(onnx_path))["model_info"]["embedded_processing"] == "True"
    assert type(cls_model) is type(deserialized)
    for attr in cls_model.parameters():
        assert getattr(cls_model, attr) == getattr(deserialized, attr)
