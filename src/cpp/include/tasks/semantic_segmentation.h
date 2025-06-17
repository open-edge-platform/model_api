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
#include "utils/tiling.h"
#include "utils/vision_pipeline.h"

class SemanticSegmentation {
public:
    std::unique_ptr<Pipeline<SemanticSegmentationResult>> pipeline;
    std::shared_ptr<InferenceAdapter> adapter;
    SemanticSegmentation(std::shared_ptr<InferenceAdapter> adapter, const ov::AnyMap& configuration)
        : adapter(adapter) {
        auto config = adapter->getModelConfig();
        tiling = utils::get_from_any_maps("tiling", configuration, config, tiling);
        if (tiling) {
            pipeline = std::make_unique<TilingPipeline<SemanticSegmentationResult>>(
                adapter,
                utils::get_tiling_info_from_config(config),
                [&](cv::Mat image) {
                    return preprocess(image);
                },
                [&](InferenceResult result) {
                    return postprocess(result);
                },
                [&](SemanticSegmentationResult& result, const cv::Rect& coord) {
                    return postprocess_tile(result, coord);
                },
                [&](const std::vector<SemanticSegmentationResult>& tiles_results,
                    const cv::Size& image_size,
                    const std::vector<cv::Rect>& tile_coords,
                    const utils::TilingInfo& tiling_info) {
                    return merge_tiling_results(tiles_results, image_size, tile_coords, tiling_info);
                });
        } else {
            pipeline = std::make_unique<VisionPipeline<SemanticSegmentationResult>>(
                adapter,
                [&](cv::Mat image) {
                    return preprocess(image);
                },
                [&](InferenceResult result) {
                    return postprocess(result);
                });
        }

        labels = utils::get_from_any_maps("labels", config, {}, labels);
        soft_threshold = utils::get_from_any_maps("soft_threshold", config, {}, soft_threshold);
        blur_strength = utils::get_from_any_maps("blur_strength", config, {}, blur_strength);
    }

    static void serialize(std::shared_ptr<ov::Model>& ov_model);
    static SemanticSegmentation load(const std::string& model_path, const ov::AnyMap& configuration = {});

    std::map<std::string, ov::Tensor> preprocess(cv::Mat);
    SemanticSegmentationResult postprocess(InferenceResult& infResult);
    std::vector<Contour> getContours(const SemanticSegmentationResult& result);

    SemanticSegmentationResult infer(cv::Mat image);
    std::vector<SemanticSegmentationResult> inferBatch(std::vector<cv::Mat> image);
    SemanticSegmentationResult postprocess_tile(SemanticSegmentationResult, const cv::Rect&);
    SemanticSegmentationResult merge_tiling_results(const std::vector<SemanticSegmentationResult>& tiles_results,
                                                    const cv::Size& image_size,
                                                    const std::vector<cv::Rect>& tile_coords,
                                                    const utils::TilingInfo& tiling_info);

private:
    cv::Mat create_hard_prediction_from_soft_prediction(cv::Mat, float threshold, int blur_strength);

    // from config
    int blur_strength = -1;
    float soft_threshold = -std::numeric_limits<float>::infinity();
    bool return_soft_prediction = true;
    bool tiling = false;

    std::vector<std::string> labels;

    std::string getLabelName(size_t labelID) {
        return labelID < labels.size() ? labels[labelID] : std::string("Label #") + std::to_string(labelID);
    }
};
