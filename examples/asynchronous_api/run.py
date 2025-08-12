#!/usr/bin/env python3
#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import sys

import cv2
from model_api.models import DetectionModel


def main():
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <path_to_image>")

    image = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
    if image is None:
        raise RuntimeError("Failed to read the image")

    # Create Object Detection model using mode name and download from Open Model Zoo
    # Replace numpy preprocessing and embed it directly into a model graph to speed up inference
    # download_dir is used to store downloaded model
    model = DetectionModel.create_model("yolo-v4-tf")

    ITERATIONS = 10
    results = {}  # container for results

    def callback(result, userdata):
        print(f"Done! Number: {userdata}")
        results[userdata] = result

    model.set_callback(callback)
    ## Run parallel inference
    for i in range(ITERATIONS):
        model.infer_async(image, user_data=i)

    model.await_all()
    assert len(results) == ITERATIONS

    for i in range(ITERATIONS):
        print(f"Request {i}: {results[i]}")


if __name__ == "__main__":
    main()
