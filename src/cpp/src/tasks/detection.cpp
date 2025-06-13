/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tasks/detection.h"

#include "adapters/openvino_adapter.h"
#include "tasks/detection/ssd.h"
#include "utils/config.h"
#include "utils/nms.h"
#include "utils/tensor.h"

DetectionModel DetectionModel::load(const std::string& model_path, const ov::AnyMap& configuration) {
    auto core = ov::Core();
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    if (model->has_rt_info("model_info", "model_type")) {
        std::cout << "has model type in info: " << model->get_rt_info<std::string>("model_info", "model_type")
                  << std::endl;
    } else {
        throw std::runtime_error("Incorrect or unsupported model_type");
    }

    cv::Size origin_input_shape;
    if (utils::model_has_embedded_processing(model)) {
        std::cout << "model already was serialized" << std::endl;
        origin_input_shape = utils::get_input_shape_from_model_info(model);
    } else {
        origin_input_shape = SSD::serialize(model);
    }
    auto adapter = std::make_shared<OpenVINOInferenceAdapter>();
    adapter->loadModel(model, core, "AUTO");
    return DetectionModel(std::make_unique<SSD>(adapter, origin_input_shape), configuration);
}

InferenceInput DetectionModel::preprocess(cv::Mat image) {
    return algorithm->preprocess(image);
}

DetectionResult DetectionModel::postprocess(InferenceResult result) {
    return algorithm->postprocess(result);
}

DetectionResult DetectionModel::postprocess_tile(DetectionResult& result, const cv::Rect& coord) {
    for (auto& det : result.objects) {
        det.x += coord.x;
        det.y += coord.y;
    }

    return result;
}

DetectionResult DetectionModel::merge_tiling_results(const std::vector<DetectionResult>& tiles_results,
                                                     const cv::Size& image_size,
                                                     const std::vector<cv::Rect>& tile_coords,
                                                     const utils::TilingInfo& tiling_info) {
    size_t max_pred_number = 200;

    DetectionResult result;

    std::vector<AnchorLabeled> all_detections;
    std::vector<std::reference_wrapper<const DetectedObject>> all_detections_refs;
    std::vector<float> all_scores;

    for (const auto& result : tiles_results) {
        for (const auto& det : result.objects) {
            all_detections.emplace_back(det.x, det.y, det.x + det.width, det.y + det.height, det.labelID);
            all_scores.push_back(det.confidence);
            all_detections_refs.push_back(std::ref(det));
        }
    }

    auto keep_idx = multiclass_nms(all_detections, all_scores, tiling_info.iou_threshold, false, max_pred_number);

    result.objects.reserve(keep_idx.size());
    for (auto idx : keep_idx) {
        result.objects.push_back(all_detections_refs[idx]);
    }

    if (tiles_results.size()) {
        auto det_res = tiles_results.begin();
        if (det_res->feature_vector) {
            result.feature_vector =
                ov::Tensor(det_res->feature_vector.get_element_type(), det_res->feature_vector.get_shape());
        }
        if (det_res->saliency_map) {
            result.saliency_map = merge_saliency_maps(tiles_results, image_size, tile_coords, tiling_info);
        }
    }

    if (result.feature_vector) {
        float* feature_ptr = result.feature_vector.data<float>();
        size_t feature_size = result.feature_vector.get_size();

        std::fill(feature_ptr, feature_ptr + feature_size, 0.f);

        for (const auto& result : tiles_results) {
            const float* current_feature_ptr = result.feature_vector.data<float>();

            for (size_t i = 0; i < feature_size; ++i) {
                feature_ptr[i] += current_feature_ptr[i];
            }
        }

        for (size_t i = 0; i < feature_size; ++i) {
            feature_ptr[i] /= tiles_results.size();
        }
    }

    return result;
}
ov::Tensor DetectionModel::merge_saliency_maps(const std::vector<DetectionResult>& tiles_results,
                                               const cv::Size& image_size,
                                               const std::vector<cv::Rect>& tile_coords,
                                               const utils::TilingInfo& tiling_info) {
    std::vector<ov::Tensor> all_saliency_maps;
    all_saliency_maps.reserve(tiles_results.size());
    for (const auto& result : tiles_results) {
        all_saliency_maps.push_back(result.saliency_map);
    }

    ov::Tensor image_saliency_map;
    if (all_saliency_maps.size()) {
        image_saliency_map = all_saliency_maps[0];
    }

    if ((image_saliency_map.get_size() == 1) || (all_saliency_maps.size() == 1)) {
        return image_saliency_map;
    }

    size_t shape_shift = (image_saliency_map.get_shape().size() > 3) ? 1 : 0;
    size_t num_classes = image_saliency_map.get_shape()[shape_shift];
    size_t map_h = image_saliency_map.get_shape()[shape_shift + 1];
    size_t map_w = image_saliency_map.get_shape()[shape_shift + 2];

    float ratio_h = static_cast<float>(map_h) / std::min(tiling_info.tile_size, static_cast<size_t>(image_size.height));
    float ratio_w = static_cast<float>(map_w) / std::min(tiling_info.tile_size, static_cast<size_t>(image_size.width));

    size_t image_map_h = static_cast<size_t>(image_size.height * ratio_h);
    size_t image_map_w = static_cast<size_t>(image_size.width * ratio_w);

    std::vector<cv::Mat_<float>> merged_map_mat(num_classes);
    for (auto& class_map : merged_map_mat) {
        class_map = cv::Mat_<float>(cv::Size{int(image_map_w), int(image_map_h)}, 0.f);
    }

    size_t start_idx = tiling_info.tile_with_full_image ? 1 : 0;
    for (size_t i = start_idx; i < all_saliency_maps.size(); ++i) {
        for (size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
            auto current_cls_map_mat =
                utils::wrap_saliency_map_tensor_to_mat(all_saliency_maps[i], shape_shift, class_idx);
            cv::Mat current_cls_map_mat_float;
            current_cls_map_mat.convertTo(current_cls_map_mat_float, CV_32F);

            cv::Rect map_location(
                static_cast<int>(tile_coords[i].x * ratio_w),
                static_cast<int>(tile_coords[i].y * ratio_h),
                static_cast<int>(static_cast<int>(tile_coords[i].width + tile_coords[i].x) * ratio_w -
                                 static_cast<int>(tile_coords[i].x * ratio_w)),
                static_cast<int>(static_cast<int>(tile_coords[i].height + tile_coords[i].y) * ratio_h -
                                 static_cast<int>(tile_coords[i].y * ratio_h)));

            if (current_cls_map_mat.rows > map_location.height && map_location.height > 0 &&
                current_cls_map_mat.cols > map_location.width && map_location.width > 0) {
                cv::resize(current_cls_map_mat_float,
                           current_cls_map_mat_float,
                           cv::Size(map_location.width, map_location.height));
            }

            auto class_map_roi = cv::Mat(merged_map_mat[class_idx], map_location);
            for (int row_i = 0; row_i < map_location.height; ++row_i) {
                for (int col_i = 0; col_i < map_location.width; ++col_i) {
                    float merged_mixel = class_map_roi.at<float>(row_i, col_i);
                    if (merged_mixel > 0) {
                        class_map_roi.at<float>(row_i, col_i) =
                            0.5f * (merged_mixel + current_cls_map_mat_float.at<float>(row_i, col_i));
                    } else {
                        class_map_roi.at<float>(row_i, col_i) = current_cls_map_mat_float.at<float>(row_i, col_i);
                    }
                }
            }
        }
    }

    ov::Tensor merged_map;
    if (shape_shift) {
        merged_map = ov::Tensor(ov::element::Type("u8"), {1, num_classes, image_map_h, image_map_w});
    } else {
        merged_map = ov::Tensor(ov::element::Type("u8"), {num_classes, image_map_h, image_map_w});
    }

    for (size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
        if (tiling_info.tile_with_full_image) {
            auto image_map_cls = utils::wrap_saliency_map_tensor_to_mat(image_saliency_map, shape_shift, class_idx);
            cv::resize(image_map_cls, image_map_cls, cv::Size(image_map_w, image_map_h));
            cv::addWeighted(merged_map_mat[class_idx], 1.0, image_map_cls, 0.5, 0., merged_map_mat[class_idx]);
        }
        merged_map_mat[class_idx] = utils::non_linear_normalization(merged_map_mat[class_idx]);
        auto merged_cls_map_mat = utils::wrap_saliency_map_tensor_to_mat(merged_map, shape_shift, class_idx);
        merged_map_mat[class_idx].convertTo(merged_cls_map_mat, merged_cls_map_mat.type());
    }

    return merged_map;
}

DetectionResult DetectionModel::infer(cv::Mat image) {
    return pipeline->infer(image);
}

std::vector<DetectionResult> DetectionModel::inferBatch(std::vector<cv::Mat> images) {
    return pipeline->inferBatch(images);
}
