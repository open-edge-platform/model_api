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
#include "utils/vision_pipeline.h"

class Anomaly {
public:
    std::shared_ptr<InferenceAdapter> adapter;
    VisionPipeline<AnomalyResult> pipeline;

    Anomaly(std::shared_ptr<InferenceAdapter> adapter, const ov::AnyMap& user_config) : adapter(adapter) {
        pipeline = VisionPipeline<AnomalyResult>(
            adapter,
            [&](cv::Mat image) {
                return preprocess(image);
            },
            [&](InferenceResult result) {
                return postprocess(result);
            });

        auto model_config = adapter->getModelConfig();
        image_threshold = utils::get_from_any_maps("image_threshold", user_config, model_config, image_threshold);
        pixel_threshold = utils::get_from_any_maps("pixel_threshold", user_config, model_config, pixel_threshold);
        normalization_scale =
            utils::get_from_any_maps("normalization_scale", user_config, model_config, normalization_scale);
        task = utils::get_from_any_maps("pixel_threshold", user_config, model_config, task);
        labels = utils::get_from_any_maps("labels", user_config, model_config, labels);
        input_shape.width = utils::get_from_any_maps("orig_width", user_config, model_config, input_shape.width);
        input_shape.height = utils::get_from_any_maps("orig_height", user_config, model_config, input_shape.height);
    }

    static void serialize(std::shared_ptr<ov::Model>& ov_model);
    static Anomaly create_model(const std::string& model_path, const ov::AnyMap& user_config = {});

    AnomalyResult infer(cv::Mat image);
    std::vector<AnomalyResult> inferBatch(std::vector<cv::Mat> image);

    std::map<std::string, ov::Tensor> preprocess(cv::Mat);
    AnomalyResult postprocess(InferenceResult& infResult);

private:
    cv::Mat normalize(cv::Mat& tensor, float threshold);
    double normalize(double& tensor, float threshold);
    std::vector<cv::Rect> getBoxes(cv::Mat& mask);

private:
    cv::Size input_shape;
    std::vector<std::string> labels;

    float image_threshold = 0.5f;
    float pixel_threshold = 0.5f;
    float normalization_scale = 1.0f;
    std::string task = "segmentation";
};
