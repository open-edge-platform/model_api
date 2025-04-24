/*
 * Copyright (C) 2020-2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include <models/results.h>
#include <tilers/detection.h>

#include <algorithm>
#include <functional>
#include <opencv2/core.hpp>
#include <utils/nms.hpp>
#include <vector>

namespace {

cv::Mat non_linear_normalization(cv::Mat& class_map) {
    double min_soft_score, max_soft_score;
    cv::Mat tmp;

    class_map.convertTo(tmp, CV_32F);
    cv::minMaxLoc(tmp, &min_soft_score);
    cv::pow(tmp - min_soft_score, 1.5, tmp);

    cv::minMaxLoc(tmp, &min_soft_score, &max_soft_score);
    tmp = 255.0 / (max_soft_score + 1e-12) * tmp;

    tmp.convertTo(class_map, class_map.type());
    return class_map;
}

}  // namespace

DetectionTiler::DetectionTiler(const std::shared_ptr<BaseModel>& _model,
                               const ov::AnyMap& configuration,
                               ExecutionMode exec_mode)
    : TilerBase(_model, configuration, exec_mode) {
    ov::AnyMap extra_config;
    try {
        auto ov_model = model->getModel();
        extra_config = ov_model->get_rt_info<ov::AnyMap>("model_info");
    } catch (const std::runtime_error&) {
        extra_config = model->getInferenceAdapter()->getModelConfig();
    }

    max_pred_number = get_from_any_maps("max_pred_number", configuration, extra_config, max_pred_number);
}

std::unique_ptr<Scene> DetectionTiler::postprocess_tile(std::unique_ptr<Scene> tile_result,
                                                             const cv::Rect& coord) {
    for (auto& det : tile_result->boxes) {
        det.shape.x += coord.x;
        det.shape.y += coord.y;
    }
    return tile_result;
}

std::unique_ptr<Scene> DetectionTiler::merge_results(const std::vector<std::unique_ptr<Scene>>& tiles_results,
                                                          const cv::Size& image_size,
                                                          const std::vector<cv::Rect>& tile_coords) {
    auto scene = std::make_unique<Scene>();

    std::vector<AnchorLabeled> all_detections;
    std::vector<std::reference_wrapper<Box>> all_detections_refs;
    std::vector<float> all_scores;

    for (const auto& result : tiles_results) {
        for (auto& det : result->boxes) {
            size_t id;
            sscanf(det.labels[0].id.c_str(), "%zu", &id);
            all_detections.emplace_back(det.shape.x, det.shape.y, det.shape.x + det.shape.width, det.shape.y + det.shape.height, id);
            all_scores.push_back(det.labels[0].score);
            all_detections_refs.push_back(det);
        }
    }

    auto keep_idx = multiclass_nms(all_detections, all_scores, iou_threshold, false, max_pred_number);

    scene->boxes.reserve(keep_idx.size());
    for (auto idx : keep_idx) {
        scene->boxes.push_back(all_detections_refs[idx]);
    }

    if (!tiles_results.empty()) {
        auto& feature_vectors = tiles_results.begin()->get()->feature_vectors;
        if (!feature_vectors.empty()) {
            auto tensor = ov::Tensor(feature_vectors[0].get_element_type(), feature_vectors[0].get_shape());

            float* feature_ptr = tensor.data<float>();
            size_t feature_size = tensor.get_size();

            std::fill(feature_ptr, feature_ptr + feature_size, 0.f);

            for (const auto& result : tiles_results) {
                const float* current_feature_ptr = result->feature_vectors[0].data<float>();

                for (size_t i = 0; i < feature_size; ++i) {
                    feature_ptr[i] += current_feature_ptr[i];
                }
            }

            for (size_t i = 0; i < feature_size; ++i) {
                feature_ptr[i] /= tiles_results.size();
            }

            scene->feature_vectors.push_back(tensor);
        }

        scene->saliency_maps = merge_saliency_maps(tiles_results, image_size, tile_coords);
    }

    return scene;
}

std::vector<cv::Mat> DetectionTiler::merge_saliency_maps(const std::vector<std::unique_ptr<Scene>>& tiles_results,
                                               const cv::Size& image_size,
                                               const std::vector<cv::Rect>& tile_coords) {

    auto map_size = tiles_results[0]->saliency_maps[0].size();

    auto dtype = tiles_results[0]->saliency_maps[0].type();
    auto num_classes = tiles_results[0]->saliency_maps.size();
    size_t map_h = map_size.height;
    size_t map_w = map_size.width;

    float ratio_h = static_cast<float>(map_h) / std::min(tile_size, static_cast<size_t>(image_size.height));
    float ratio_w = static_cast<float>(map_w) / std::min(tile_size, static_cast<size_t>(image_size.width));

    cv::Size ratio(ratio_w, ratio_h);


    size_t image_map_h = static_cast<size_t>(image_size.height * ratio_h);
    size_t image_map_w = static_cast<size_t>(image_size.width * ratio_w);

    cv::Size merged_map_size(image_map_w, image_map_h);

    std::vector<cv::Mat> saliency_maps(num_classes);

    size_t start_index = (tile_with_full_img ? 1 : 0);
    for (size_t class_index = 0; class_index < saliency_maps.size(); class_index++) {
        saliency_maps[class_index] = cv::Mat(merged_map_size, dtype, 0.f);

        for (size_t i = start_index; i < tiles_results.size(); i++) {
            cv::Rect map_location(
                static_cast<int>(tile_coords[i].x * ratio_w),
                static_cast<int>(tile_coords[i].y * ratio_h),
                static_cast<int>(static_cast<int>(tile_coords[i].width + tile_coords[i].x) * ratio_w -
                                 static_cast<int>(tile_coords[i].x * ratio_w)),
                static_cast<int>(static_cast<int>(tile_coords[i].height + tile_coords[i].y) * ratio_h -
                                 static_cast<int>(tile_coords[i].y * ratio_h)));
            saliency_maps[class_index](map_location) = tiles_results[i]->saliency_maps[class_index];
        }

        if (tile_with_full_img) {
            auto image_map_cls = tiles_results[0]->saliency_maps[class_index];
            cv::resize(image_map_cls, image_map_cls, cv::Size(image_map_w, image_map_h));
            cv::addWeighted(saliency_maps[class_index], 1.0, image_map_cls, 0.5, 0., saliency_maps[class_index]);
            non_linear_normalization(saliency_maps[class_index]);
        }
    }
    return saliency_maps;
}

std::unique_ptr<Scene> DetectionTiler::run(const ImageInputData& inputData) {
    return this->run_impl(inputData);
}
