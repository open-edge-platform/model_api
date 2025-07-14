/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stddef.h>
#include <tasks/segment_anything.h>
#include <tasks/results.h>
#include <utils/tensor.h>
#include <utils/preprocessing.h>
#include <adapters/openvino_adapter.h>

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/core/graph_util.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <stdexcept>
#include <string>
#include <fstream>

void store_tensor(ov::Tensor tensor, const std::string& path){
    std::ofstream file(path, std::ofstream::binary);
    file.write(static_cast<char*>(tensor.data()), tensor.get_byte_size());
    file.close();
    std::cout << "stored tensor of shape: " << tensor.get_shape() << " and dtype: " << tensor.get_element_type() << " to " << path << std::endl;
}

ov::Tensor load_tensor(const std::filesystem::path& path, const ov::Shape& shape, const ov::element::Type& element) {
    ov::Tensor tensor(element, shape);
    std::ifstream file(path, std::ofstream::binary);
    file.read(static_cast<char*>(tensor.data()), tensor.get_byte_size());
    file.close();
    return tensor;
}

std::shared_ptr<MaskPredictor> predictor;

// SAM normalization constants (in pixel scale)
const std::vector<float> MEAN = {123.675f, 116.28f, 103.53f};
const std::vector<float> STD  = {58.395f, 57.12f, 57.375f};


cv::Matx33f transform;
cv::Mat image;

int main(int argc, char* argv[]) try {
    if (argc != 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <path_to_encoder_model> <path_to_predictor_model> <path_to_image>");
    }

    std::string tmp_tensor_path = "./image_encodings.ov";

    image = cv::imread(argv[3]);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }
    auto model = SegmentAnything::create_model(argv[1], argv[2]);

    auto predictor_adapter = std::make_shared<OpenVINOInferenceAdapter>();
    predictor_adapter->loadModel(argv[2], "CPU", {}, true);


    auto image_encodings = load_tensor("./image_encodings.ov", ov::Shape{1,256,64,64}, ov::element::f32);
    predictor = std::make_shared<MaskPredictor>(predictor_adapter, image_encodings, image.size(), utils::RESIZE_KEEP_ASPECT);

    cv::namedWindow("image");

    cv::Size input_size(1024, 1024);
    float scaleX = input_size.width / (float)image.cols;
    float scaleY = input_size.height / (float)image.rows;
    float s = std::min(scaleX, scaleY);
    float sx = s;
    float sy = s;

    transform = {
        sx, 0, 0,
        0, sy, 0,
        0, 0, 1,
    };


    cv::setMouseCallback("image", [](int event, int x, int y, int, void* ) {
        if (event == 4) {
            cv::Point point(x,y);
            auto masks = predictor->infer({point}, {});

            cv::Mat resizedMask;
            cv::resize(masks[0], resizedMask, image.size());
            cv::Mat output;
            auto overlay = cv::Mat::ones(image.size(), CV_8UC3);
            cv::Mat resizedMaskInt;
            resizedMask.convertTo(resizedMaskInt, CV_8U, 255, 0);
            cv::bitwise_and(overlay, overlay, output, resizedMaskInt);
            cv::Mat blended;
            cv::addWeighted(image, 1.0, output, 180.0f, 0.0, blended);


            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(resizedMaskInt, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

            cv::drawContours(blended, contours, 0, cv::Scalar{255, 0, 0}, 2);
            cv::cvtColor(blended, blended, cv::COLOR_BGR2RGB);
            cv::imshow("image", blended);
        }

    });

    cv::Size new_size(1024, 1024);
    if (image.cols > image.rows) {
        new_size.height = (float)image.rows / image.cols * new_size.width;
    } else {
        new_size.width = (float)image.cols / image.rows * new_size.height;
    }


    std::cout << "new size: " << new_size << std::endl;

    cv::Mat resized;
    cv::resize(image, resized, new_size);

    cv::imshow("image", image);

    cv::waitKey(0);

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
