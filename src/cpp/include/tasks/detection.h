#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "adapters/inference_adapter.h"
#include "tasks/detection/ssd.h"
#include "tasks/results.h"
#include "utils/vision_pipeline.h"
#include "utils/tiling.h"

class DetectionModel {
public:
    std::unique_ptr<Pipeline<DetectionResult>> pipeline;

    DetectionModel(std::unique_ptr<SSD> algorithm, const ov::AnyMap& configuration): algorithm(std::move(algorithm)) {
        auto config = this->algorithm->adapter->getModelConfig();
        if (configuration.count("tiling") && configuration.at("tiling").as<bool>()) {
            if (!utils::config_contains_tiling_info(config)) {
                throw std::runtime_error("Model config does not contain tiling properties.");
            }
            pipeline = std::make_unique<TilingPipeline<DetectionResult>>(this->algorithm->adapter, 
                utils::get_tiling_info_from_config(config),
                [&](cv::Mat image) { return preprocess(image);},
                [&](InferenceResult result) { return postprocess(result);},
                [&](DetectionResult& result, const cv::Rect& coord) { return postprocess_tile(result, coord);},
                [&](const std::vector<DetectionResult>& tiles_results, const cv::Size& image_size, const std::vector<cv::Rect>& tile_coords, const utils::TilingInfo& tiling_info) { return merge_tiling_results(tiles_results, image_size, tile_coords, tiling_info);}
            );
        } else {
            pipeline = std::make_unique<VisionPipeline<DetectionResult>>(this->algorithm->adapter, 
                [&](cv::Mat image) { return preprocess(image);},
                [&](InferenceResult result) { return postprocess(result);}
            );
        }
    }

    InferenceInput preprocess(cv::Mat);
    DetectionResult postprocess(InferenceResult);
    DetectionResult postprocess_tile(DetectionResult& result, const cv::Rect& coord);
    DetectionResult merge_tiling_results(const std::vector<DetectionResult>& tiles_results, const cv::Size& image_size, const std::vector<cv::Rect>& tile_coords, const utils::TilingInfo& tiling_info);
    ov::Tensor merge_saliency_maps(const std::vector<DetectionResult>& tiles_results, const cv::Size& image_size, const std::vector<cv::Rect>& tile_coords, const utils::TilingInfo& tiling_info);

    static DetectionModel load(const std::string& model_path, const ov::AnyMap& configuration = {});

    DetectionResult infer(cv::Mat image);
    void inferAsync(cv::Mat image, ov::AnyMap user_data);
    void setCallback(std::function<void(DetectionResult, ov::AnyMap)>);

    std::vector<DetectionResult> inferBatch(std::vector<cv::Mat> image);

private:
    std::unique_ptr<SSD> algorithm;
};
