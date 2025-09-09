#!/usr/bin/env python3
#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import logging
import sys
from pathlib import Path

import cv2

from model_api.models import Model

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)


def get_image_files(dataset_path: str) -> list[str]:
    """Get list of image files from the dataset directory."""
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
    test_images: list[str] = []
    test_path = Path(dataset_path)

    for ext in image_extensions:
        test_images.extend(str(p) for p in test_path.glob(ext))
        test_images.extend(str(p) for p in test_path.glob(ext.upper()))

    return test_images


def analyze_model_performance(model_path, test_images, device, warmup_runs, test_runs):
    """Complete performance analysis example."""

    # Load model
    model = Model.create_model(model_path, device=device)

    # Load test image
    image = cv2.imread(test_images[0])

    print("Starting warm-up...")
    # Warm-up runs
    for _ in range(warmup_runs):
        model(image)

    # Reset metrics after warm-up
    model.get_performance_metrics().reset()

    print(f"Running {test_runs} test inferences...")
    # Performance measurement runs
    for i, image_path in enumerate(test_images[:test_runs]):
        image = cv2.imread(image_path)
        model(image)
        # Log progress
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{test_runs}")

    # Analyze results
    metrics = model.get_performance_metrics()
    metrics.log_metrics()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmark - a model performance analysis with metrics collection")
    parser.add_argument("model_path", help="Path to the model file (.xml)")
    parser.add_argument("dataset_path", help="Path to the dataset directory containing test images")
    parser.add_argument("--device", type=str, default="CPU", help="OpenVINO device to run the model on (default: CPU)")
    parser.add_argument("--warmup-runs", type=int, default=5, help="Number of warmup runs (default: 5)")
    parser.add_argument("--test-runs", type=int, default=100, help="Number of test runs (default: 100)")

    # Show help if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    model_path = args.model_path
    dataset_path = args.dataset_path

    # Get list of image files from the directory
    test_images = get_image_files(dataset_path)

    print(f"Found {len(test_images)} images in {dataset_path}")

    if not test_images:
        print("Error: No images found in the dataset directory!")
        exit(1)

    analyze_model_performance(model_path, test_images, args.device, args.warmup_runs, args.test_runs)


if __name__ == "__main__":
    main()
