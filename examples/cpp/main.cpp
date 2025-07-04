/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stddef.h>
#include <tasks/visual_prompting.h>
#include <tasks/results.h>

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <openvino/openvino.hpp>
#include <stdexcept>
#include <string>
#include <fstream>

void store_tensor(ov::Tensor tensor, const std::string& path){
    std::ofstream file(path, std::ofstream::binary);
    file.write(static_cast<char*>(tensor.data()), tensor.get_byte_size());
    file.close();
    std::cout << "stored tensor of shape: " << tensor.get_shape() << " to " << path << std::endl;
}

ov::Tensor load_tensor(const std::filesystem::path& path, const ov::Shape& shape, const ov::element::Type& element) {
    ov::Tensor tensor(element, shape);
    std::ifstream file(path, std::ofstream::binary);
    file.read(static_cast<char*>(tensor.data()), tensor.get_byte_size());
    file.close();
    return tensor;
}

std::unique_ptr<MaskPredictor> predictor;

int main(int argc, char* argv[]) try {
    if (argc != 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <path_to_encoder_model> <path_to_predictor_model> <path_to_image>");
    }

    std::string tmp_tensor_path = "./image_encodings.ov";

    cv::Mat image = cv::imread(argv[3]);
    //cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }

    // Instantiate Object Detection model
    auto model = VisualPrompting::create_model(argv[1], argv[2]);

    //std::cout << "Building predictor: " << std::endl;
    //auto predictor = model.infer(image);
    //store_tensor(predictor.image_encodings, tmp_tensor_path);

    auto tensor = load_tensor(tmp_tensor_path, ov::Shape{1,256,64,64}, ov::element::f32);

    predictor = std::make_unique<MaskPredictor>(model.predictor_adapter, tensor, image.size());

    std::cout << "done" << std::endl;

    cv::namedWindow("image");

    cv::setMouseCallback("image", [](int event, int x, int y, int, void* ) {
        if (event == 4) {
            std::cout << "got " << event  << " on: " << cv::Size(x, y) << std::endl;

            cv::Point point(x, y);
            predictor->infer({point}, {});
        }

    });

    cv::imshow("image", image);

    cv::waitKey(0);




    // Run the inference
    //auto result = model.infer(image);

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
