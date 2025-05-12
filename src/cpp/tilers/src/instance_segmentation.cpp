/*
 * Copyright (C) 2020-2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include <models/instance_segmentation.h>
#include <models/results.h>
#include <tilers/instance_segmentation.h>

#include <algorithm>
#include <functional>
#include <opencv2/core.hpp>
#include <utils/nms.hpp>
#include <vector>

#include "utils/common.hpp"

namespace {
class MaskRCNNModelParamsSetter {
public:
    std::shared_ptr<BaseModel> model;
    bool state;
    MaskRCNNModel* model_ptr;
    MaskRCNNModelParamsSetter(std::shared_ptr<BaseModel> model_) : model(model_) {
        model_ptr = static_cast<MaskRCNNModel*>(model.get());
        state = model_ptr->postprocess_semantic_masks;
        model_ptr->postprocess_semantic_masks = false;
    }
    ~MaskRCNNModelParamsSetter() {
        model_ptr->postprocess_semantic_masks = state;
    }
};
}  // namespace

InstanceSegmentationTiler::InstanceSegmentationTiler(std::shared_ptr<BaseModel> _model,
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

    postprocess_semantic_masks =
        get_from_any_maps("postprocess_semantic_masks", configuration, extra_config, postprocess_semantic_masks);
    max_pred_number = get_from_any_maps("max_pred_number", configuration, extra_config, max_pred_number);
}

std::unique_ptr<Scene> InstanceSegmentationTiler::run(const ImageInputData& inputData) {
    auto setter = MaskRCNNModelParamsSetter(model);
    return this->run_impl(inputData);
}

std::unique_ptr<Scene> InstanceSegmentationTiler::postprocess_tile(std::unique_ptr<Scene> tile_result,
                                                                        const cv::Rect& coord) {
    for (auto& det : tile_result->new_masks) {
        det.roi.x += coord.x;
        det.roi.y += coord.y;
    }

    return tile_result;
}

std::unique_ptr<Scene> InstanceSegmentationTiler::merge_results(
    const std::vector<std::unique_ptr<Scene>>& tiles_results,
    const cv::Size& image_size,
    const std::vector<cv::Rect>& tile_coords) {
    auto scene = std::make_unique<Scene>();

    std::vector<AnchorLabeled> all_detections;
    std::vector<std::reference_wrapper<Mask>> all_detections_ptrs;
    std::vector<float> all_scores;

    for (const auto& result : tiles_results) {
        for (auto& det : result->new_masks) {
            all_detections.emplace_back(det.roi.x, det.roi.y, det.roi.x + det.roi.width, det.roi.y + det.roi.height, det.label.label.id);
            all_scores.push_back(det.label.score);
            all_detections_ptrs.push_back(det);
        }
    }

    auto keep_idx = multiclass_nms(all_detections, all_scores, iou_threshold, false, max_pred_number);

    scene->new_masks.reserve(keep_idx.size());
    for (auto idx : keep_idx) {
        if (postprocess_semantic_masks) {
            all_detections_ptrs[idx].get().mask = segm_postprocess(all_detections_ptrs[idx],
                                                                   all_detections_ptrs[idx].get().mask,
                                                                   image_size.height,
                                                                   image_size.width);
        }
        scene->new_masks.push_back(all_detections_ptrs[idx]);
    }

    if (tiles_results.size()) {
        auto& feature_vectors = tiles_results.begin()->get()->feature_vectors;
        if (!feature_vectors.empty()) {
            scene->feature_vectors.push_back(ov::Tensor(feature_vectors[0].get_element_type(), feature_vectors[0].get_shape()));
        }
    }

    if (!scene->feature_vectors.empty()) {
        auto feature_vector = scene->feature_vectors[0];
        float* feature_ptr = feature_vector.data<float>();
        size_t feature_size = feature_vector.get_size();

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
    }

    scene->saliency_maps = merge_saliency_maps(tiles_results, image_size, tile_coords);
    return scene;
}

std::vector<cv::Mat> InstanceSegmentationTiler::merge_saliency_maps(
    const std::vector<std::unique_ptr<Scene>>& tiles_results,
    const cv::Size& image_size,
    const std::vector<cv::Rect>& tile_coords) {
    std::vector<std::vector<cv::Mat>> all_saliecy_maps;
    all_saliecy_maps.reserve(tiles_results.size());


    for (const auto& result : tiles_results) {
        all_saliecy_maps.push_back(result->saliency_maps);
    }

    std::vector<cv::Mat> image_saliency_map;
    if (all_saliecy_maps.size()) {
        image_saliency_map = all_saliecy_maps[0];
    }

    if (image_saliency_map.empty()) {
        return image_saliency_map;
    }


    size_t num_classes = image_saliency_map.size();
    std::vector<cv::Mat> merged_map(num_classes);
    for (auto& map : merged_map) {
        map = cv::Mat(image_size, 0);
    }

    size_t start_idx = tile_with_full_img ? 1 : 0;
    for (size_t i = start_idx; i < all_saliecy_maps.size(); ++i) {
        for (size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
            auto current_cls_map_mat = all_saliecy_maps[i][class_idx];
            if (current_cls_map_mat.empty()) {
                continue;
            }
            const auto& tile = tile_coords[i];
            cv::Mat tile_map;
            cv::resize(current_cls_map_mat, tile_map, tile.size());
            auto tile_map_merged = cv::Mat(merged_map[class_idx], tile);
            cv::Mat(cv::max(tile_map, tile_map_merged)).copyTo(tile_map_merged);
        }
    }

    for (size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
        auto image_map_cls = tile_with_full_img ? image_saliency_map[class_idx] : cv::Mat();
        if (image_map_cls.empty()) {
            if (cv::sum(merged_map[class_idx]) == cv::Scalar(0.)) {
                merged_map[class_idx] = cv::Mat();
            }
        } else {
            cv::resize(image_map_cls, image_map_cls, image_size);
            cv::Mat(cv::max(merged_map[class_idx], image_map_cls)).copyTo(merged_map[class_idx]);
        }
    }

    return merged_map;
}
