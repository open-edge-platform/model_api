#
# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import json
import os
from pathlib import Path
from urllib.request import urlopen, urlretrieve


def retrieve_otx_model(data_dir, model_name, format="xml"):
    destination_folder = Path(data_dir) / "otx_models"
    destination_folder.mkdir(parents=True, exist_ok=True)
    if format == "onnx":
        urlretrieve(
            f"https://storage.geti.intel.com/geti_predict/test/otx_models/{model_name}/model.onnx",
            destination_folder / f"{model_name}.onnx",
        )
    else:
        urlretrieve(
            f"https://storage.geti.intel.com/geti_predict/test/otx_models/{model_name}/openvino.xml",
            destination_folder / f"{model_name}.xml",
        )
        urlretrieve(
            f"https://storage.geti.intel.com/geti_predict/test/otx_models/{model_name}/openvino.bin",
            f"{destination_folder}/{model_name}.bin",
        )


def prepare_model(
    data_dir="./data",
    public_scope=Path(__file__).resolve().parent / "public_scope.json",
):
    # TODO refactor this test so that it does not use eval
    # flake8: noqa: F401
    from model_api.models import ClassificationModel, DetectionModel, SegmentationModel

    # Mapping of model type strings to actual classes for security
    MODEL_TYPE_MAPPING = {
        "ClassificationModel": ClassificationModel,
        "DetectionModel": DetectionModel,
        "SegmentationModel": SegmentationModel,
    }

    with Path(public_scope).open("r") as f:
        public_scope = json.load(f)

    for model in public_scope:
        if model["name"].endswith(".xml") or model["name"].endswith(".onnx"):
            continue
        model = MODEL_TYPE_MAPPING[model["type"]].create_model(model["name"], download_dir=data_dir)


def prepare_data(data_dir="./data"):
    from io import BytesIO
    from zipfile import ZipFile

    COCO128_URL = "https://ultralytics.com/assets/coco128.zip"

    with urlopen(COCO128_URL) as zipresp, ZipFile(BytesIO(zipresp.read())) as zfile:  # noqa: S310
        zfile.extractall(data_dir)

    urlretrieve(
        "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00007.jpg",
        Path(data_dir) / "BloodImage_00007.jpg",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data and model preparate script")
    parser.add_argument(
        "-d",
        dest="data_dir",
        default="./data",
        help="Directory to store downloaded models and datasets",
    )
    parser.add_argument(
        "-p",
        dest="public_scope",
        default=Path(__file__).resolve().parent / "public_scope.json",
        help="JSON file with public model description",
    )

    args = parser.parse_args()

    prepare_model(args.data_dir, args.public_scope)
    prepare_data(args.data_dir)
    retrieve_otx_model(args.data_dir, "mlc_mobilenetv3_large_voc")
    retrieve_otx_model(args.data_dir, "detection_model_with_xai_head")
    retrieve_otx_model(args.data_dir, "Lite-hrnet-18_mod2")
    retrieve_otx_model(args.data_dir, "tinynet_imagenet")
    retrieve_otx_model(args.data_dir, "cls_mobilenetv3_large_cars", "onnx")
