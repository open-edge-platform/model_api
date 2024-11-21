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

    image = cv2.imread(sys.argv[1])
    if image is None:
        raise RuntimeError("Failed to read the image")

    # Create Object Detection model specifying the OVMS server URL
    model = DetectionModel.create_model(
        "localhost:9000/models/ssd_mobilenet_v1_fpn_coco", model_type="ssd"
    )
    detections = model(image)
    print(f"Detection results: {detections}")


if __name__ == "__main__":
    main()
