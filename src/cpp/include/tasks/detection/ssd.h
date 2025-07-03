/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <openvino/openvino.hpp>

#include "adapters/inference_adapter.h"
#include "tasks/results.h"
#include "utils/config.h"
#include "utils/preprocessing.h"

enum class SSDOutputMode { single, multi };

class NumAndStep {
public:
    size_t detectionsNum, objectSize;

    static inline NumAndStep fromSingleOutput(const ov::Shape& shape);
    static inline NumAndStep fromMultipleOutputs(const ov::Shape& boxesShape);
};

constexpr float box_area_threshold = 1.0f;

class SSD {
public:
    std::shared_ptr<InferenceAdapter> adapter;

    SSD(std::shared_ptr<InferenceAdapter> adapter) : adapter(adapter), input_shape(input_shape) {
        auto config = adapter->getModelConfig();
        labels = utils::get_from_any_maps("labels", config, {}, labels);
        confidence_threshold = utils::get_from_any_maps("confidence_threshold", config, {}, confidence_threshold);
        input_shape.width = utils::get_from_any_maps("orig_width", config, {}, input_shape.width);
        input_shape.height = utils::get_from_any_maps("orig_height", config, {}, input_shape.height);
        resize_mode = utils::get_from_any_maps("resize_type", config, {}, resize_mode);
    }
    std::map<std::string, ov::Tensor> preprocess(cv::Mat);
    DetectionResult postprocess(InferenceResult& infResult);

    static void serialize(std::shared_ptr<ov::Model>& ov_model);

    SSDOutputMode output_mode;

private:
    static void prepareSingleOutput(std::shared_ptr<ov::Model> ov_model);
    static void prepareMultipleOutputs(std::shared_ptr<ov::Model> ov_model);

    DetectionResult postprocessSingleOutput(InferenceResult& infResult);
    DetectionResult postprocessMultipleOutputs(InferenceResult& infResult);

    float confidence_threshold = 0.5f;

    std::vector<std::string> labels;
    std::vector<std::string> filterOutXai(const std::vector<std::string>&);

    std::vector<std::string> output_names = {};
    utils::RESIZE_MODE resize_mode = utils::RESIZE_MODE::RESIZE_FILL;
    ov::Layout layout;
    cv::InterpolationFlags interpolation_mode;
    cv::Size input_shape;
};
