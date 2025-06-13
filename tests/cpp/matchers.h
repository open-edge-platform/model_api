#pragma once

#include "tasks/instance_segmentation.h"
#include "tasks/semantic_segmentation.h"

inline std::string format_test_output_to_string(const InstanceSegmentation& model,
                                                const InstanceSegmentationResult& result) {
    const std::vector<SegmentedObjectWithRects>& withRects = model.getRotatedRectangles(result);
    std::stringstream ss;
    for (const SegmentedObjectWithRects& obj : withRects) {
        ss << obj << "; ";
    }
    size_t filled = 0;
    for (const cv::Mat_<std::uint8_t>& cls_map : result.saliency_map) {
        if (cls_map.data) {
            ++filled;
        }
    }
    ss << filled << "; ";
    try {
        ss << result.feature_vector.get_shape();
    } catch (ov::Exception&) {
        ss << "[0]";
    }
    ss << "; ";
    try {
        // getContours() assumes each instance generates only one contour.
        // That doesn't hold for some models
        for (const Contour& contour : model.getContours(result.segmentedObjects)) {
            ss << contour << "; ";
        }
    } catch (const std::runtime_error&) {
    }
    return ss.str();
}

inline std::string format_test_output_to_string(SemanticSegmentation& model, const SemanticSegmentationResult& result) {
    const std::vector<Contour>& contours = model.getContours(result);
    std::stringstream ss;
    ss << result << "; ";
    std::cout << contours.size() << std::endl;
    for (const Contour& contour : contours) {
        ss << contour << ", ";
    }

    return ss.str();
}
