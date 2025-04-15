#!/usr/bin/env python3

from vision_api import ClassificationModel
import cv2

import sys

if len(sys.argv) != 3:
    raise RuntimeError(f"Usage: {sys.argv[0]} <path_to_model> <path_to_image>")

model_path = sys.argv[1]
image_path = sys.argv[2]

model = ClassificationModel.create_model(model_path)
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
model(image)
