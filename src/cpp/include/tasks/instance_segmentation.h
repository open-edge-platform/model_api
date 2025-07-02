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

class InstanceSegmentation {
public:
    std::shared_ptr<InferenceAdapter> adapter;
    VisionPipeline<InstanceSegmentationResult> pipeline;

    InstanceSegmentation(std::shared_ptr<InferenceAdapter> adapter, const ov::AnyMap& user_config) : adapter(adapter) {
        pipeline = VisionPipeline<InstanceSegmentationResult>(
            adapter,
            [&](cv::Mat image) {
                return preprocess(image);
            },
            [&](InferenceResult result) {
                return postprocess(result);
            });

        auto model_config = adapter->getModelConfig();
        labels = utils::get_from_any_maps("labels", user_config, model_config, labels);
        confidence_threshold =
            utils::get_from_any_maps("confidence_threshold", user_config, model_config, confidence_threshold);
        input_shape.width = utils::get_from_any_maps("orig_width", user_config, model_config, input_shape.width);
        input_shape.height = utils::get_from_any_maps("orig_height", user_config, model_config, input_shape.width);
    }

    static void serialize(std::shared_ptr<ov::Model>& ov_model);
    static InstanceSegmentation create_model(const std::string& model_path,
                                             const ov::AnyMap& user_config = {},
                                             bool preload = true,
                                             const std::string& device = "AUTO");

    InstanceSegmentationResult infer(cv::Mat image);
    std::vector<InstanceSegmentationResult> inferBatch(std::vector<cv::Mat> image);

    std::map<std::string, ov::Tensor> preprocess(cv::Mat);
    InstanceSegmentationResult postprocess(InferenceResult& infResult);

    static std::vector<SegmentedObjectWithRects> getRotatedRectangles(const InstanceSegmentationResult& result);
    static std::vector<Contour> getContours(const std::vector<SegmentedObject>& objects);

    bool postprocess_semantic_masks = true;

private:
    std::vector<std::string> labels;
    std::string getLabelName(size_t labelID) {
        return labelID < labels.size() ? labels[labelID] : std::string("Label #") + std::to_string(labelID);
    }

    cv::Size input_shape;
    float confidence_threshold = 0.5f;
};
