#!/usr/bin/env python3
#
# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import sys

import cv2
from model_api.models import ClassificationModel, DetectionModel, SegmentationModel
from PIL import Image


def main():
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <path_to_image>")

    image = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
    if image is None:
        raise RuntimeError("Failed to read the image")

    # Create Image Classification model using mode name and download from Open Model Zoo
    efficientnet_b0 = ClassificationModel.create_model("efficientnet-b0-pytorch")
    classifications = efficientnet_b0(image)
    print(f"Classification results: {classifications}")

    # Create Object Detection model using mode name and download from Open Model Zoo
    # Replace numpy preprocessing and embed it directly into a model graph to speed up inference
    # download_dir is used to store downloaded model
    ssd_mobilenet_fpn = DetectionModel.create_model(
        "ssd_mobilenet_v1_fpn_coco",
        download_dir="tmp",
    )
    detections = ssd_mobilenet_fpn(image)
    print(f"Detection results: {detections}")
    ssd_mobilenet_fpn.save("ssd_mobilenet_v1_fpn_coco_with_preprocessing.xml")

    # Instantiate from a local model (downloaded previously)
    ssd_mobilenet_fpn_local = DetectionModel.create_model(
        "tmp/public/ssd_mobilenet_v1_fpn_coco/FP16/ssd_mobilenet_v1_fpn_coco.xml"
    )
    detections = ssd_mobilenet_fpn_local(image)
    print(f"Detection results for local: {detections}")

    # Create Image Segmentation model
    hrnet = SegmentationModel.create_model("hrnet-v2-c1-segmentation")
    mask = hrnet(image)
    Image.fromarray(mask + 20).save("mask.png")


if __name__ == "__main__":
    main()
