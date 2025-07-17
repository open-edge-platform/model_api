/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include <adapters/openvino_adapter.h>
#include <stddef.h>
#include <tasks/results.h>
#include <tasks/segment_anything.h>
#include <utils/preprocessing.h>
#include <utils/tensor.h>

#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/core/graph_util.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <optional>
#include <stdexcept>
#include <string>

void store_tensor(ov::Tensor tensor, const std::string& path) {
    std::ofstream file(path, std::ofstream::binary);
    file.write(static_cast<char*>(tensor.data()), tensor.get_byte_size());
    file.close();
    std::cout << "stored tensor of shape: " << tensor.get_shape() << " and dtype: " << tensor.get_element_type()
              << " to " << path << std::endl;
}

ov::Tensor load_tensor(const std::filesystem::path& path, const ov::Shape& shape, const ov::element::Type& element) {
    ov::Tensor tensor(element, shape);
    std::ifstream file(path, std::ofstream::binary);
    file.read(static_cast<char*>(tensor.data()), tensor.get_byte_size());
    file.close();
    return tensor;
}

std::shared_ptr<MaskPredictor> predictor;

cv::Mat image;
cv::Mat initial;

cv::Point start;

std::vector<cv::Point> positive_points;
std::vector<cv::Point> negative_points;

std::optional<cv::Rect> box;
int main(int argc, char* argv[]) try {
    if (argc != 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                 " <path_to_encoder_model> <path_to_predictor_model> <path_to_image>");
    }

    std::string tmp_tensor_path = "./image_encodings.ov";

    initial = cv::imread(argv[3]);
    cv::cvtColor(initial, image, cv::COLOR_BGR2RGB);

    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }
    auto model = SegmentAnything::create_model(argv[1], argv[2]);
    predictor = std::make_unique<MaskPredictor>(model.infer(image));
    cv::namedWindow("image");

    cv::setMouseCallback("image", [](int event, int x, int y, int, void*) {
        bool run_inference = false;

        if (event == 1) {
            start = cv::Point(x, y);
        }
        if (event == 6) {
            // reset!
            box.reset();
            positive_points.clear();
            negative_points.clear();
            predictor->reset_mask_input();
            cv::imshow("image", initial);
        }
        if (event == 5) {
            negative_points.emplace_back(x, y);
            run_inference = true;
        }

        if (event == 4) {
            float distance = (x - start.x) * (x - start.x) + (y - start.y) * (y - start.y);
            if (distance > 10) {
                box = cv::Rect(cv::Point(x, y), start);
            } else {
                positive_points.emplace_back(x, y);
            }
            run_inference = true;
        }
        if (run_inference) {
            std::vector<SegmentAnythingMask> masks;
            if (box.has_value()) {
                masks = predictor->infer(box.value());
            } else {
                masks = predictor->infer(positive_points, negative_points);
            }

            cv::Mat resizedMask;
            cv::resize(masks[0].mask, resizedMask, image.size());
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

            for (auto& point : positive_points) {
                cv::circle(blended, point, 3, cv::Scalar{0, 255, 0});
            }
            for (auto& point : negative_points) {
                cv::circle(blended, point, 3, cv::Scalar{255, 0, 0});
            }
            if (box.has_value()) {
                cv::rectangle(blended, box.value(), cv::Scalar{255, 0, 0}, 2);
            }

            cv::imshow("image", blended);
        }
    });
    cv::imshow("image", initial);

    cv::waitKey(0);

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
