/*
 * Copyright (C) 2020-2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include <models/detection_model.h>
#include <models/input_data.h>
#include <models/results.h>
#include <exception>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <openvino/openvino.hpp>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[]) try {
    if (argc != 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <path_to_model> <path_to_image>");
    }

    cv::Mat image = cv::imread(argv[2]);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }

    // Instantiate Object Detection model
    auto model = DetectionModel::create_model(argv[1]);  // works with SSD models. Download it using Python Model API

    // Run the inference
    auto result = model->infer(image);

    // Process detections
    for (auto& obj : result->boxes) {
        std::cout << obj << std::endl;
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
