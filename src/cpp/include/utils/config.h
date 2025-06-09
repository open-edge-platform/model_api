#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
namespace utils {
template <typename Type>
Type get_from_any_maps(const std::string& key,
                       const ov::AnyMap& top_priority,
                       const ov::AnyMap& mid_priority,
                       Type low_priority) {
    auto topk_iter = top_priority.find(key);
    if (topk_iter != top_priority.end()) {
        return topk_iter->second.as<Type>();
    }
    topk_iter = mid_priority.find(key);
    if (topk_iter != mid_priority.end()) {
        return topk_iter->second.as<Type>();
    }
    return low_priority;
}

inline bool model_has_embedded_processing(std::shared_ptr<ov::Model> model) {
    if (model->has_rt_info("model_info")) {
        auto model_info = model->get_rt_info<ov::AnyMap>("model_info");
        auto iter = model_info.find("embedded_processing");
        if (iter != model_info.end()) {
            return iter->second.as<std::string>() == "YES";
        }
    }

    return false;
}

inline cv::Size get_input_shape_from_model_info(std::shared_ptr<ov::Model> model) {
    cv::Size result;
    if (model->has_rt_info("model_info")) {
        auto model_info = model->get_rt_info<ov::AnyMap>("model_info");
        {
            auto iter = model_info.find("orig_height");
            if (iter != model_info.end()) {
                result.height = iter->second.as<int>();
            }
        }
        {
            auto iter = model_info.find("orig_width");
            if (iter != model_info.end()) {
                result.width = iter->second.as<int>();
            }
        }
    }

    return result;
}
struct IntervalCondition {
    using DimType = size_t;
    using IndexType = size_t;
    using ConditionChecker = std::function<bool(IndexType, const ov::PartialShape&)>;

    template <class Cond>
    constexpr IntervalCondition(IndexType i1, IndexType i2, Cond c)
        : impl([=](IndexType i0, const ov::PartialShape& shape) {
              return c(shape[i0].get_max_length(), shape[i1].get_max_length()) &&
                     c(shape[i0].get_max_length(), shape[i2].get_max_length());
          }) {}
    bool operator()(IndexType i0, const ov::PartialShape& shape) const {
        return impl(i0, shape);
    }

private:
    ConditionChecker impl;
};

template <template <class> class Cond, class... Args>
IntervalCondition makeCond(Args&&... args) {
    return IntervalCondition(std::forward<Args>(args)..., Cond<IntervalCondition::DimType>{});
}
using LayoutCondition = std::tuple<size_t /*dim index*/, IntervalCondition, std::string>;

static inline std::tuple<bool, ov::Layout> makeGuesLayoutFrom4DShape(const ov::PartialShape& shape) {
    // at the moment we make assumption about NCHW & NHCW only
    // if hypothetical C value is less than hypothetical H and W - then
    // out assumption is correct and we pick a corresponding layout
    static const std::array<LayoutCondition, 2> hypothesisMatrix{
        {{1, makeCond<std::less_equal>(2, 3), "NCHW"}, {3, makeCond<std::less_equal>(1, 2), "NHWC"}}};
    for (const auto& h : hypothesisMatrix) {
        auto channel_index = std::get<0>(h);
        const auto& cond = std::get<1>(h);
        if (cond(channel_index, shape)) {
            return std::make_tuple(true, ov::Layout{std::get<2>(h)});
        }
    }
    return {false, ov::Layout{}};
}

static inline ov::Layout getLayoutFromShape(const ov::PartialShape& shape) {
    if (shape.size() == 2) {
        return "NC";
    }
    if (shape.size() == 3) {
        if (shape[0] == 1) {
            return "NHW";
        }
        if (shape[2] == 1) {
            return "HWN";
        }
        throw std::runtime_error("Can't guess layout for " + shape.to_string());
    }
    if (shape.size() == 4) {
        if (ov::Interval{1, 4}.contains(shape[1].get_interval())) {
            return "NCHW";
        }
        if (ov::Interval{1, 4}.contains(shape[3].get_interval())) {
            return "NHWC";
        }
        if (shape[1] == shape[2]) {
            return "NHWC";
        }
        if (shape[2] == shape[3]) {
            return "NCHW";
        }
        bool guesResult = false;
        ov::Layout guessedLayout;
        std::tie(guesResult, guessedLayout) = makeGuesLayoutFrom4DShape(shape);
        if (guesResult) {
            return guessedLayout;
        }
    }
    throw std::runtime_error("Usupported " + std::to_string(shape.size()) + "D shape");
}

}  // namespace utils
