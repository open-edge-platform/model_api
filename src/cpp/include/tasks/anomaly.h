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

    Anomaly(std::shared_ptr<InferenceAdapter> adapter, cv::Size input_shape)
        : adapter(adapter),
          input_shape(input_shape) {
        pipeline = VisionPipeline<AnomalyResult>(
            adapter,
            [&](cv::Mat image) {
                return preprocess(image);
            },
            [&](InferenceResult result) {
                return postprocess(result);
            });

        auto config = adapter->getModelConfig();
        image_threshold = utils::get_from_any_maps("image_threshold", config, {}, image_threshold);
        pixel_threshold = utils::get_from_any_maps("pixel_threshold", config, {}, pixel_threshold);
        normalization_scale = utils::get_from_any_maps("pixel_threshold", config, {}, normalization_scale);
        task = utils::get_from_any_maps("pixel_threshold", config, {}, task);
        labels = utils::get_from_any_maps("labels", config, {}, labels);

        // labels = utils::get_from_any_maps("labels", config, {}, labels);
        // confidence_threshold = utils::get_from_any_maps("confidence_threshold", config, {}, confidence_threshold);
    }

    static cv::Size serialize(std::shared_ptr<ov::Model>& ov_model);
    static Anomaly load(const std::string& model_path);

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
