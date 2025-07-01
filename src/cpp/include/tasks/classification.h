/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "adapters/inference_adapter.h"
#include "tasks/classification/resolvers.h"
#include "tasks/results.h"
#include "utils/config.h"
#include "utils/vision_pipeline.h"

class Classification {
public:
    std::shared_ptr<InferenceAdapter> adapter;
    VisionPipeline<ClassificationResult> pipeline;

    Classification(std::shared_ptr<InferenceAdapter> adapter, const ov::AnyMap& user_config) : adapter(adapter) {
        pipeline = VisionPipeline<ClassificationResult>(
            adapter,
            [&](cv::Mat image) {
                return preprocess(image);
            },
            [&](InferenceResult result) {
                return postprocess(result);
            });

        auto model_config = adapter->getModelConfig();
        labels = utils::get_from_any_maps("labels", user_config, model_config, labels);

        topk = utils::get_from_any_maps("topk", user_config, model_config, topk);
        multilabel = utils::get_from_any_maps("multilabel", user_config, model_config, multilabel);
        output_raw_scores = utils::get_from_any_maps("output_raw_scores", user_config, model_config, output_raw_scores);
        confidence_threshold =
            utils::get_from_any_maps("confidence_threshold", user_config, model_config, confidence_threshold);
        hierarchical = utils::get_from_any_maps("hierarchical", user_config, model_config, hierarchical);
        hierarchical_config =
            utils::get_from_any_maps("hierarchical_config", user_config, model_config, hierarchical_config);
        hierarchical_postproc =
            utils::get_from_any_maps("hierarchical_postproc", user_config, model_config, hierarchical_postproc);
        if (hierarchical) {
            if (hierarchical_config.empty()) {
                throw std::runtime_error("Error: empty hierarchical classification config");
            }
            hierarchical_info = HierarchicalConfig(hierarchical_config);
            if (hierarchical_postproc == "probabilistic") {
                resolver = std::make_unique<ProbabilisticLabelsResolver>(hierarchical_info);
            } else if (hierarchical_postproc == "greedy") {
                resolver = std::make_unique<GreedyLabelsResolver>(hierarchical_info);
            } else {
                throw std::runtime_error("Wrong hierarchical labels postprocessing type");
            }
        }
    }

    static void serialize(std::shared_ptr<ov::Model>& ov_model);
    static Classification create_model(const std::string& model_path,
                                       const ov::AnyMap& user_config = {},
                                       bool preload = true,
                                       const std::string& device = "AUTO");

    ClassificationResult infer(cv::Mat image);
    std::vector<ClassificationResult> inferBatch(std::vector<cv::Mat> image);

    std::map<std::string, ov::Tensor> preprocess(cv::Mat);
    ClassificationResult postprocess(InferenceResult& infResult);

    bool postprocess_semantic_masks = true;

private:
    ClassificationResult get_multilabel_predictions(InferenceResult& infResult, bool add_raw_scores);
    ClassificationResult get_multiclass_predictions(InferenceResult& infResult, bool add_raw_scores);
    ClassificationResult get_hierarchical_predictions(InferenceResult& infResult, bool add_raw_scores);

    ov::Tensor reorder_saliency_maps(const ov::Tensor& source_maps);

    // multiclass serialization step
    static void addOrFindSoftmaxAndTopkOutputs(std::shared_ptr<ov::Model>& model, size_t topk, bool add_raw_scores);

private:
    cv::Size input_shape;
    std::vector<std::string> labels;
    float confidence_threshold = 0.5f;

    bool multilabel = false;
    bool hierarchical = false;
    bool output_raw_scores = false;

    // hierarchical
    size_t topk = 1;
    std::string hierarchical_config;
    std::string hierarchical_postproc = "greedy";
    HierarchicalConfig hierarchical_info;
    std::unique_ptr<GreedyLabelsResolver> resolver;
};
