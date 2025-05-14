/*
 * Copyright (C) 2020-2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
#include <adapters/openvino_adapter.h>
#include <models/detection_model.h>
#include <models/results.h>
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
    auto model = DetectionModel::create_model(argv[1],
                                              {},
                                              "",
                                              false);  // works with SSD models. Download it using Python Model API
    // Define number of parallel infer requests. Is this number is set to 0, OpenVINO will determine it automatically to
    // obtain optimal performance.
    size_t num_requests = 0;
    static ov::Core core;
    model->load(core, "CPU", num_requests);

    std::cout << "Async inference will be carried out by " << model->getNumAsyncExecutors() << " parallel executors\n";
    // Prepare batch data
    std::vector<ImageInputData> data;
    for (size_t i = 0; i < 3; i++) {
        data.push_back(ImageInputData(image));
    }

    // Batch inference is done by processing batch with num_requests parallel infer requests
    std::cout << "Starting batch inference\n";
    auto results = model->inferBatch(data);

    std::cout << "Batch mode inference results:\n";
    for (const auto& result : results) {
        for (auto& obj : result->boxes) {
            std::cout << obj << std::endl;
        }
        std::cout << std::string(10, '-') << "\n";
    }
    std::cout << "Batch mode inference done\n";
    std::cout << "Async mode inference results:\n";

    // Set callback to grab results once the inference is done
    model->setCallback([](std::unique_ptr<Scene> result, const ov::AnyMap& callback_args) {

        // callback_args can contain arbitrary data
        size_t id = callback_args.find("id")->second.as<size_t>();

        std::cout << "Request with id " << id << " is finished\n";
        for (auto& obj : result->boxes) {
            std::cout << " " << obj << std::endl;
        }
        std::cout << std::string(10, '-') << "\n";
    });

    for (size_t i = 0; i < 3; i++) {
        model->inferAsync(image, {{"id", i}});
    }
    model->awaitAll();
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
