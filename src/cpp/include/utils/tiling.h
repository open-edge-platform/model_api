/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "utils/config.h"

namespace utils {

struct TilingInfo {
    size_t tile_size = 400;
    float tiles_overlap = 0.5f;
    float iou_threshold = 0.45f;
    bool tile_with_full_image = true;
};

inline bool config_contains_tiling_info(const ov::AnyMap& config) {
    auto iter = config.find("tile_size");
    return iter != config.end();
}

inline TilingInfo get_tiling_info_from_config(const ov::AnyMap& config) {
    TilingInfo info;
    info.tile_size = utils::get_from_any_maps("tile_size", config, {}, info.tile_size);
    info.tiles_overlap = utils::get_from_any_maps("tiles_overlap", config, {}, info.tiles_overlap);
    info.iou_threshold = utils::get_from_any_maps("iou_threshold", config, {}, info.iou_threshold);
    info.tile_with_full_image = utils::get_from_any_maps("tile_with_full_image", config, {}, info.tile_with_full_image);
    return info;
}

static inline cv::Mat wrap_saliency_map_tensor_to_mat(ov::Tensor& t, size_t shape_shift, size_t class_idx) {
    int ocv_dtype;
    switch (t.get_element_type()) {
    case ov::element::u8:
        ocv_dtype = CV_8U;
        break;
    case ov::element::f32:
        ocv_dtype = CV_32F;
        break;
    default:
        throw std::runtime_error("Unsupported saliency map data type in ov::Tensor to cv::Mat wrapper: " +
                                 t.get_element_type().get_type_name());
    }
    void* t_ptr = static_cast<char*>(t.data()) + class_idx * t.get_strides()[shape_shift];
    auto mat_size =
        cv::Size(static_cast<int>(t.get_shape()[shape_shift + 2]), static_cast<int>(t.get_shape()[shape_shift + 1]));

    return cv::Mat(mat_size, ocv_dtype, t_ptr, t.get_strides()[shape_shift + 1]);
}

inline cv::Mat non_linear_normalization(cv::Mat& class_map) {
    double min_soft_score, max_soft_score;
    cv::minMaxLoc(class_map, &min_soft_score);
    cv::pow(class_map - min_soft_score, 1.5, class_map);

    cv::minMaxLoc(class_map, &min_soft_score, &max_soft_score);
    class_map = 255.0 / (max_soft_score + 1e-12) * class_map;

    return class_map;
}

}  // namespace utils
