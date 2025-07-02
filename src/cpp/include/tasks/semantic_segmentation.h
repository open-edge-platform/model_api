/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "adapters/inference_adapter.h"
#include "tasks/results.h"
#include "utils/config.h"
#include "utils/preprocessing.h"
#include "utils/vision_pipeline.h"

class SemanticSegmentation {
public:
    VisionPipeline<SemanticSegmentationResult> pipeline;
    std::shared_ptr<InferenceAdapter> adapter;
    SemanticSegmentation(std::shared_ptr<InferenceAdapter> adapter, const ov::AnyMap& user_config) : adapter(adapter) {
        pipeline = VisionPipeline<SemanticSegmentationResult>(
            adapter,
            [&](cv::Mat image) {
                return preprocess(image);
            },
            [&](InferenceResult result) {
                return postprocess(result);
            });

        auto model_config = adapter->getModelConfig();
        labels = utils::get_from_any_maps("labels", user_config, model_config, labels);
        soft_threshold = utils::get_from_any_maps("soft_threshold", user_config, model_config, soft_threshold);
        blur_strength = utils::get_from_any_maps("blur_strength", user_config, model_config, blur_strength);
    }

    static void serialize(std::shared_ptr<ov::Model>& ov_model);
    static SemanticSegmentation create_model(const std::string& model_path,
                                             const ov::AnyMap& user_config = {},
                                             bool preload = true,
                                             const std::string& device = "AUTO");

    std::map<std::string, ov::Tensor> preprocess(cv::Mat);
    SemanticSegmentationResult postprocess(InferenceResult& infResult);
    std::vector<Contour> getContours(const SemanticSegmentationResult& result);

    SemanticSegmentationResult infer(cv::Mat image);
    std::vector<SemanticSegmentationResult> inferBatch(std::vector<cv::Mat> image);

private:
    cv::Mat create_hard_prediction_from_soft_prediction(cv::Mat, float threshold, int blur_strength);

    // from config
    int blur_strength = -1;
    float soft_threshold = -std::numeric_limits<float>::infinity();
    bool return_soft_prediction = true;

    std::vector<std::string> labels;

    std::string getLabelName(size_t labelID) {
        return labelID < labels.size() ? labels[labelID] : std::string("Label #") + std::to_string(labelID);
    }
};
