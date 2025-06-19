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
    std::unique_ptr<Pipeline<InstanceSegmentationResult>> pipeline;

    InstanceSegmentation(std::shared_ptr<InferenceAdapter> adapter, const ov::AnyMap& configuration) : adapter(adapter) {
        auto config = adapter->getModelConfig();
        tiling = utils::get_from_any_maps("tiling", configuration, config, tiling);
        if (tiling) {
            pipeline = std::make_unique<TilingPipeline<InstanceSegmentationResult>>(
                adapter,
                utils::get_tiling_info_from_config(config),
                [&](cv::Mat image) {
                    return preprocess(image);
                },
                [&](InferenceResult result) {
                    return postprocess(result);
                },
                [&](InstanceSegmentationResult result, const cv::Rect& coord) {
                    return postprocess_tile(result, coord);
                },
                [&](const std::vector<InstanceSegmentationResult>& tiles_results,
                    const cv::Size& image_size,
                    const std::vector<cv::Rect>& tile_coords,
                    const utils::TilingInfo& tiling_info) {
                    return merge_tiling_results(tiles_results, image_size, tile_coords, tiling_info);
                });
        } else {
            pipeline = std::make_unique<VisionPipeline<InstanceSegmentationResult>>(
                adapter,
                [&](cv::Mat image) {
                    return preprocess(image);
                },
                [&](InferenceResult result) {
                    return postprocess(result);
                });
        }
        labels = utils::get_from_any_maps("labels", config, {}, labels);
        confidence_threshold = utils::get_from_any_maps("confidence_threshold", config, {}, confidence_threshold);
        input_shape.width = utils::get_from_any_maps("orig_width", config, {}, input_shape.width);
        input_shape.height = utils::get_from_any_maps("orig_height", config, {}, input_shape.width);
        resize_mode = utils::get_from_any_maps("resize_type", config, {}, resize_mode);
    }

    static void serialize(std::shared_ptr<ov::Model>& ov_model);
    static InstanceSegmentation load(const std::string& model_path, const ov::AnyMap& configuration);

    InstanceSegmentationResult infer(cv::Mat image);
    std::vector<InstanceSegmentationResult> inferBatch(std::vector<cv::Mat> image);

    std::map<std::string, ov::Tensor> preprocess(cv::Mat);
    InstanceSegmentationResult postprocess(InferenceResult& infResult);
    InstanceSegmentationResult postprocess_tile(InstanceSegmentationResult, const cv::Rect&);
    InstanceSegmentationResult merge_tiling_results(const std::vector<InstanceSegmentationResult>& tiles_results,
                                                    const cv::Size& image_size,
                                                    const std::vector<cv::Rect>& tile_coords,
                                                    const utils::TilingInfo& tiling_info);
    std::vector<cv::Mat_<std::uint8_t>> merge_saliency_maps(const std::vector<InstanceSegmentationResult>&,
                                                            const cv::Size&,
                                                            const std::vector<cv::Rect>&,
                                                            const utils::TilingInfo&);

    
    static std::vector<SegmentedObjectWithRects> getRotatedRectangles(const InstanceSegmentationResult& result);
    static std::vector<Contour> getContours(const std::vector<SegmentedObject>& objects);

    bool postprocess_semantic_masks = true;

private:

    bool tiling;

    utils::RESIZE_MODE resize_mode;
    std::vector<std::string> labels;
    std::string getLabelName(size_t labelID) {
        return labelID < labels.size() ? labels[labelID] : std::string("Label #") + std::to_string(labelID);
    }

    cv::Size input_shape;
    float confidence_threshold = 0.5f;
};
