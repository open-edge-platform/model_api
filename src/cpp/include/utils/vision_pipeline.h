/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "adapters/inference_adapter.h"
#include "tasks/results.h"
#include "utils/tiling.h"

template <typename ResultType>
class Pipeline {
public:
    Pipeline() {}
    virtual ResultType infer(cv::Mat image) = 0;
    virtual std::vector<ResultType> inferBatch(std::vector<cv::Mat> images) = 0;
};

template <typename ResultType>
class VisionPipeline : public Pipeline<ResultType> {
private:
    std::shared_ptr<InferenceAdapter> adapter;
    std::function<InferenceInput(cv::Mat)> preprocess;
    std::function<ResultType(InferenceResult)> postprocess;

public:
    VisionPipeline() {}
    VisionPipeline(std::shared_ptr<InferenceAdapter> adapter,
                   std::function<InferenceInput(cv::Mat)> preprocess,
                   std::function<ResultType(InferenceResult)> postprocess)
        : adapter(adapter),
          preprocess(preprocess),
          postprocess(postprocess) {}

    inline ResultType infer(cv::Mat image) {
        auto input = preprocess(image);
        InferenceResult result;
        result.inputImageSize = image.size();
        result.data = adapter->infer(input);
        return postprocess(result);
    }

    inline std::vector<ResultType> inferBatch(std::vector<cv::Mat> images) {
        auto results = std::vector<ResultType>(images.size());

        adapter->setCallback([&](ov::InferRequest request, CallbackData additional_data) {
            InferenceResult result;
            size_t index = additional_data->at("index").as<size_t>();
            result.inputImageSize = additional_data->at("inputImageSize").as<cv::Size>();
            for (const auto& item : adapter->getOutputNames()) {
                result.data.emplace(item, request.get_tensor(item));
            }
            results[index] = postprocess(result);
        });

        for (size_t i = 0; i < images.size(); i++) {
            auto input = preprocess(images[i]);
            auto additional_data = std::make_shared<ov::AnyMap>();
            additional_data->insert({"index", i});
            additional_data->insert({"inputImageSize", images[i].size()});
            adapter->inferAsync(input, additional_data);
        }

        adapter->awaitAll();

        return results;
    }
};

template <typename ResultType>
class TilingPipeline : public Pipeline<ResultType> {
private:
    std::shared_ptr<InferenceAdapter> adapter;
    utils::TilingInfo tiling_info;
    std::function<InferenceInput(cv::Mat)> preprocess;
    std::function<ResultType(InferenceResult)> postprocess;
    std::function<ResultType(ResultType&, const cv::Rect&)> postprocess_tile;
    std::function<ResultType(const std::vector<ResultType>&,
                             const cv::Size&,
                             const std::vector<cv::Rect>&,
                             const utils::TilingInfo&)>
        merge_tiling_results;

public:
    TilingPipeline() {}
    TilingPipeline(std::shared_ptr<InferenceAdapter> adapter,
                   utils::TilingInfo tiling_info,
                   std::function<InferenceInput(cv::Mat)> preprocess,
                   std::function<ResultType(InferenceResult)> postprocess,
                   std::function<ResultType(ResultType&, const cv::Rect&)> postprocess_tile,
                   std::function<ResultType(const std::vector<ResultType>&,
                                            const cv::Size&,
                                            const std::vector<cv::Rect>&,
                                            const utils::TilingInfo&)> merge_tiling_results)
        : adapter(adapter),
          tiling_info(tiling_info),
          preprocess(preprocess),
          postprocess(postprocess),
          postprocess_tile(postprocess_tile),
          merge_tiling_results(merge_tiling_results) {}

    inline ResultType infer(cv::Mat image) {
        std::vector<ResultType> tile_results;
        auto tile_coords = tile(image.size());

        for (const auto& coord : tile_coords) {
            auto tile_img = cv::Mat(image, coord);
            auto input = preprocess(tile_img.clone());
            InferenceResult result;
            result.inputImageSize = coord.size();
            result.data = adapter->infer(input);
            auto tile_result = postprocess(result);
            tile_results.push_back(postprocess_tile(tile_result, coord));
        }

        return merge_tiling_results(tile_results, image.size(), tile_coords, tiling_info);
    }

    inline std::vector<ResultType> inferBatch(std::vector<cv::Mat> images) {
        std::vector<std::vector<ResultType>> tile_results_for_all_images(images.size());
        std::vector<ResultType> output(images.size());
        std::vector<std::vector<cv::Rect>> tile_coordinates(images.size());

        adapter->setCallback([&](ov::InferRequest request, CallbackData additional_data) {
            InferenceResult result;
            size_t index = additional_data->at("index").as<size_t>();
            result.inputImageSize = additional_data->at("inputImageSize").as<cv::Size>();
            auto coord = additional_data->at("tileCoord").as<cv::Rect>();
            for (const auto& item : adapter->getOutputNames()) {
                result.data.emplace(item, request.get_tensor(item));
            }
            auto tile_result = postprocess(result);
            tile_results_for_all_images[index].push_back(postprocess_tile(tile_result, coord));
            tile_coordinates[index].push_back(coord);
        });

        for (size_t i = 0; i < images.size(); i++) {
            auto tile_coords = tile(images[i].size());

            for (const auto& coord : tile_coords) {
                auto tile_img = cv::Mat(images[i], coord);
                auto input = preprocess(tile_img.clone());
                auto additional_data = std::make_shared<ov::AnyMap>();
                additional_data->insert({"index", i});
                additional_data->insert({"inputImageSize", coord.size()});
                additional_data->insert({"tileCoord", coord});
                adapter->inferAsync(input, additional_data);
            }
        }
        adapter->awaitAll();
        for (size_t i = 0; i < images.size(); i++) {
            output[i] = merge_tiling_results(tile_results_for_all_images[i],
                                             images[i].size(),
                                             tile_coordinates[i],
                                             tiling_info);
        }

        return output;
    }

private:
    inline std::vector<cv::Rect> tile(const cv::Size& image_size) {
        std::vector<cv::Rect> coords;

        size_t tile_step = static_cast<size_t>(tiling_info.tile_size * (1.f - tiling_info.tiles_overlap));
        size_t num_h_tiles = image_size.height / tile_step;
        size_t num_w_tiles = image_size.width / tile_step;

        if (num_h_tiles * tile_step < static_cast<size_t>(image_size.height)) {
            num_h_tiles += 1;
        }

        if (num_w_tiles * tile_step < static_cast<size_t>(image_size.width)) {
            num_w_tiles += 1;
        }

        if (tiling_info.tile_with_full_image) {
            coords.reserve(num_h_tiles * num_w_tiles + 1);
            coords.push_back(cv::Rect(0, 0, image_size.width, image_size.height));
        } else {
            coords.reserve(num_h_tiles * num_w_tiles);
        }

        for (size_t i = 0; i < num_w_tiles; ++i) {
            for (size_t j = 0; j < num_h_tiles; ++j) {
                int loc_h = static_cast<int>(j * tile_step);
                int loc_w = static_cast<int>(i * tile_step);

                coords.push_back(
                    cv::Rect(loc_w,
                             loc_h,
                             std::min(static_cast<int>(tiling_info.tile_size), image_size.width - loc_w),
                             std::min(static_cast<int>(tiling_info.tile_size), image_size.height - loc_h)));
            }
        }
        return coords;
    }
};
