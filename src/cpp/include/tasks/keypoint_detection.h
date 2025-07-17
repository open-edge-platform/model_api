/*
 * Copyright (C) 2024-2025 Intel Corporation
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

class KeypointDetection {
public:
    VisionPipeline<KeypointDetectionResult> pipeline;
    std::shared_ptr<InferenceAdapter> adapter;
    KeypointDetection(std::shared_ptr<InferenceAdapter> adapter, const ov::AnyMap& user_config) : adapter(adapter) {
        pipeline = VisionPipeline<KeypointDetectionResult>(
            adapter,
            [&](cv::Mat image) {
                return preprocess(image);
            },
            [&](InferenceResult result) {
                return postprocess(result);
            });

        auto model_config = adapter->getModelConfig();
        labels = utils::get_from_any_maps("labels", user_config, model_config, labels);
        apply_softmax = utils::get_from_any_maps("apply_softmax", user_config, model_config, apply_softmax);

        input_shape.width = utils::get_from_any_maps("orig_width", user_config, model_config, input_shape.width);
        input_shape.height = utils::get_from_any_maps("orig_height", user_config, model_config, input_shape.width);
        resize_mode = utils::get_from_any_maps("resize_type", user_config, model_config, resize_mode);
    }

    static void serialize(std::shared_ptr<ov::Model>& ov_model);
    static KeypointDetection create_model(const std::string& model_path,
                                          const ov::AnyMap& user_config = {},
                                          bool preload = true,
                                          const std::string& device = "AUTO");

    std::map<std::string, ov::Tensor> preprocess(cv::Mat);
    KeypointDetectionResult postprocess(InferenceResult& infResult);

    KeypointDetectionResult infer(cv::Mat image);
    std::vector<KeypointDetectionResult> inferBatch(std::vector<cv::Mat> image);

private:
    cv::Size input_shape;
    bool apply_softmax = true;
    utils::RESIZE_MODE resize_mode = utils::RESIZE_MODE::RESIZE_FILL;
    std::vector<std::string> labels;
};
