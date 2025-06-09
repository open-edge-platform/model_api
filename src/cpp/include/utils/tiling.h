#pragma once

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

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
    {
        auto iter = config.find("tile_size");
        if (iter != config.end()) {
            info.tile_size = iter->second.as<size_t>();
        }
    }
    {
        auto iter = config.find("tiles_overlap");
        if (iter != config.end()) {
            info.tiles_overlap = iter->second.as<float>();
        }
    }
    {
        auto iter = config.find("iou_threshold");
        if (iter != config.end()) {
            info.iou_threshold = iter->second.as<float>();
        }
    }
    {
        auto iter = config.find("tile_with_full_img");
        if (iter != config.end()) {
            info.tile_with_full_image = iter->second.as<bool>();
        }
    }
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



}