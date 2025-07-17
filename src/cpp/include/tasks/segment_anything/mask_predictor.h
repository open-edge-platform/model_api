#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>

#include "utils/config.h"
#include "utils/preprocessing.h"
#include "adapters/inference_adapter.h"
#include "utils/vision_pipeline.h"

class MaskPredictor {
private:
    void build_transform();

public:
    std::shared_ptr<InferenceAdapter> adapter;
    ov::Tensor image_encodings;
    cv::Size input_image_size;
    cv::Size input_image_tensor_size;
    utils::RESIZE_MODE resize_mode;

    MaskPredictor() {}
    MaskPredictor(std::shared_ptr<InferenceAdapter> adapter, ov::Tensor image_encodings, cv::Size input_image_size, cv::Size input_image_tensor_size, utils::RESIZE_MODE resize_mode):
        adapter(adapter), image_encodings(image_encodings), input_image_size(input_image_size), input_image_tensor_size(input_image_tensor_size), resize_mode(resize_mode) {
        build_transform();
    }

    std::vector<cv::Mat> infer(std::vector<cv::Point> positive, std::vector<cv::Point> negative = {});

    std::vector<cv::Mat> infer(cv::Rect box);

    std::vector<cv::Mat> postprocess(InferenceResult result);
    std::map<std::string, ov::Tensor> preprocess(std::vector<float> points, std::vector<float> labels);

    cv::Matx33f resize_transform;

    cv::Point transform(cv::Point);
};
