/*
// Copyright (C) 2018-2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <stddef.h>

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <models/detection_model.h>
#include <models/input_data.h>
#include <models/results.h>

int main(int argc, char* argv[]) try {
    if (argc != 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <path_to_model> <path_to_image>");
    }

    cv::Mat image = cv::imread(argv[2]);
    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }

    // Instantiate Object Detection model
    auto model = DetectionModel::create_model(argv[1]); // works with SSD models. Download it using Python Model API

    // Run the inference
    auto result = model->infer(image);

    // Process detections
    for (auto& obj : result->objects) {
        std::cout << " " << std::left << std::setw(9) << obj.label << " | " << std::setw(10) << obj.confidence
            << " | " << std::setw(4) << int(obj.x) << " | " << std::setw(4) << int(obj.y) << " | "
            << std::setw(4) << int(obj.x + obj.width) << " | " << std::setw(4) << int(obj.y + obj.height) << "\n";
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
