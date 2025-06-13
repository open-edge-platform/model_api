/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "adapters/inference_adapter.h"
#include "tasks/results.h"
#include "utils/vision_pipeline.h"
#include "utils/config.h"

class Classification {
public:
    std::shared_ptr<InferenceAdapter> adapter;
    VisionPipeline<ClassificationResult> pipeline;

    Classification(std::shared_ptr<InferenceAdapter> adapter, cv::Size input_shape)
        : adapter(adapter),
          input_shape(input_shape) {
        pipeline = VisionPipeline<ClassificationResult>(
            adapter,
            [&](cv::Mat image) {
                return preprocess(image);
            },
            [&](InferenceResult result) {
                return postprocess(result);
            });

        auto config = adapter->getModelConfig();
        labels = utils::get_from_any_maps("labels", config, {}, labels);
        //auto iter = config.find("labels");
        //if (iter != config.end()) {
        //    labels = iter->second.as<std::vector<std::string>>();
        //} else {
        //    std::cout << "could not find labels from model config" << std::endl;
        //}

        multilabel = utils::get_from_any_maps("multilabel", config, {}, multilabel);
        hierarchical = utils::get_from_any_maps("hierarchical", config, {}, hierarchical);
        output_raw_scores = utils::get_from_any_maps("output_raw_scores", config, {}, output_raw_scores);

        //{
        //    auto iter = config.find("confidence_threshold");
        //    if (iter != config.end()) {
        //        confidence_threshold = iter->second.as<float>();
        //    }
        //}
    }
    static cv::Size serialize(std::shared_ptr<ov::Model>& ov_model);
    static Classification load(const std::string& model_path);

    ClassificationResult infer(cv::Mat image);
    std::vector<InstanceSegmentationResult> inferBatch(std::vector<cv::Mat> image);

    std::map<std::string, ov::Tensor> preprocess(cv::Mat);
    ClassificationResult postprocess(InferenceResult& infResult);

    bool postprocess_semantic_masks = true;

private:
    ClassificationResult get_multilabel_predictions(InferenceResult& infResult, bool add_raw_scores);
    ClassificationResult get_multiclass_predictions(InferenceResult& infResult, bool add_raw_scores);
    ClassificationResult get_hierarchical_predictions(InferenceResult& infResult, bool add_raw_scores);

private:
    cv::Size input_shape;
    std::vector<std::string> labels;
    //float confidence_threshold = 0.5f;

    bool multilabel = false;
    bool hierarchical = false;
    bool output_raw_scores = false;

};

