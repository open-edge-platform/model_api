#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>

#include "tasks/results.h"
#include "utils/config.h"
#include "utils/preprocessing.h"
#include "adapters/inference_adapter.h"
#include "utils/vision_pipeline.h"

class MaskPredictor {
public:
    std::shared_ptr<InferenceAdapter> adapter;
    ov::Tensor image_encodings;
    cv::Size input_image_size;
    cv::Size input_image_tensor_size;
    utils::RESIZE_MODE resize_mode;

    MaskPredictor() {}
    MaskPredictor(std::shared_ptr<InferenceAdapter> adapter, ov::Tensor image_encodings, cv::Size input_image_size, cv::Size input_image_tensor_size, utils::RESIZE_MODE resize_mode):
        adapter(adapter), image_encodings(image_encodings), input_image_size(input_image_size), input_image_tensor_size(input_image_tensor_size), resize_mode(resize_mode) {
        resize_transform = build_transform();

        reset_mask_input();
    }

    std::vector<SegmentAnythingMask> infer(std::vector<cv::Point> positive, std::vector<cv::Point> negative = {});

    std::vector<SegmentAnythingMask> infer(cv::Rect box);

    std::vector<SegmentAnythingMask> postprocess(InferenceResult result);
    std::map<std::string, ov::Tensor> preprocess(std::vector<float> points, std::vector<float> labels);

    cv::Matx33f resize_transform;
    cv::Point transform(cv::Point);
    void reset_mask_input();

private:
    cv::Matx33f build_transform();
    ov::Tensor mask_input_tensor;
    bool use_previous_mask_input = 0;

};
