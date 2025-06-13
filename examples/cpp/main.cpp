/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stddef.h>
#include <tasks/classification.h>
#include <tasks/detection.h>
#include <tasks/results.h>

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <openvino/openvino.hpp>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <path_to_model> <path_to_image>");
    }

    cv::Mat image = cv::imread(argv[2]);
    //cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }

    //// Instantiate Object Detection model
    auto model = Classification::load(argv[1]);  // works with SSD models. Download it using Python Model API

    //// Run the inference
    auto result = model.infer(image);

    //// Process detections
    std::cout << result << std::endl;
    std::cout << "expected: " << std::endl << "1 (bicycle): 0.825, 11 (dog): 0.873, 14 (person): 0.824, [0], [0], [0]" << std::endl;
}
