/*
 * Copyright (C) 2020-2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
#include <adapters/openvino_adapter.h>
#include <tasks/detection.h>
#include <tasks/results.h>
#include <stddef.h>

#include <cstdint>
#include <exception>
#include <iomanip>
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
    auto model = DetectionModel::load(argv[1], {});  // works with SSD models. Download it using Python Model API
    //// Define number of parallel infer requests. Is this number is set to 0, OpenVINO will determine it automatically to
    //// obtain optimal performance.
    //size_t num_requests = 0;
    //static ov::Core core;
    //model->load(core, "CPU", num_requests);

    //std::cout << "Async inference will be carried out by " << model->getNumAsyncExecutors() << " parallel executors\n";
    //// Prepare batch data
    std::vector<cv::Mat> data = {image};
    //for (size_t i = 0; i < 3; i++) {
    //    data.push_back(ImageInputData(image));
    //}

    // Batch inference is done by processing batch with num_requests parallel infer requests
    std::cout << "Starting batch inference\n";
    auto results = model.inferBatch(data);

    std::cout << "Batch mode inference results:\n";
    for (const auto& result : results) {
        for (auto& obj : result.objects) {
            std::cout << " " << std::left << std::setw(9) << obj.confidence << " " << obj.label << "\n";
        }
        std::cout << std::string(10, '-') << "\n";
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
