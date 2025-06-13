/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

class InferenceResult {
public:
    std::map<std::string, ov::Tensor> data;
    cv::Size inputImageSize;
};

struct DetectedObject : public cv::Rect2f {
    size_t labelID;
    std::string label;
    float confidence;

    friend std::ostream& operator<<(std::ostream& os, const DetectedObject& detection) {
        return os << int(detection.x) << ", " << int(detection.y) << ", " << int(detection.x + detection.width) << ", "
                  << int(detection.y + detection.height) << ", " << detection.labelID << " (" << detection.label
                  << "): " << std::fixed << std::setprecision(3) << detection.confidence;
    }
};

struct DetectionResult {
    DetectionResult() {}
    std::vector<DetectedObject> objects;
    ov::Tensor saliency_map, feature_vector;  // Contan "saliency_map" and "feature_vector" model outputs if such exist

    friend std::ostream& operator<<(std::ostream& os, const DetectionResult& prediction) {
        for (const DetectedObject& obj : prediction.objects) {
            os << obj << "; ";
        }
        try {
            os << prediction.saliency_map.get_shape() << "; ";
        } catch (ov::Exception&) {
            os << "[0]; ";
        }
        try {
            os << prediction.feature_vector.get_shape();
        } catch (ov::Exception&) {
            os << "[0]";
        }
        return os;
    }

    explicit operator std::string() {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }
};

struct Contour {
    std::string label;
    float probability;
    std::vector<cv::Point> shape;

    friend std::ostream& operator<<(std::ostream& os, const Contour& contour) {
        return os << contour.label << ": " << std::fixed << std::setprecision(3) << contour.probability << ", "
                  << contour.shape.size();
    }
};

struct SemanticSegmentationResult {
    SemanticSegmentationResult() {}
    cv::Mat resultImage;
    cv::Mat soft_prediction;
    cv::Mat saliency_map;
    ov::Tensor feature_vector;

    friend std::ostream& operator<<(std::ostream& os, const SemanticSegmentationResult& prediction) {
        cv::Mat predicted_mask[] = {prediction.resultImage};
        int nimages = 1;
        int* channels = nullptr;
        cv::Mat mask;
        cv::Mat outHist;
        int dims = 1;
        int histSize[] = {256};
        float range[] = {0, 256};
        const float* ranges[] = {range};
        cv::calcHist(predicted_mask, nimages, channels, mask, outHist, dims, histSize, ranges);

        os << std::fixed << std::setprecision(3);
        for (int i = 0; i < range[1]; ++i) {
            const float count = outHist.at<float>(i);
            if (count > 0) {
                os << i << ": " << count / prediction.resultImage.total() << ", ";
            }
        }
        os << '[';
        for (int i = 0; i < prediction.soft_prediction.dims; ++i) {
            os << prediction.soft_prediction.size[i] << ',';
        }
        os << prediction.soft_prediction.channels() << "], [";
        if (prediction.saliency_map.data) {
            for (int i = 0; i < prediction.saliency_map.dims; ++i) {
                os << prediction.saliency_map.size[i] << ',';
            }
            os << prediction.saliency_map.channels() << "], ";
        } else {
            os << "0], ";
        }
        try {
            os << prediction.feature_vector.get_shape();
        } catch (ov::Exception&) {
            os << "[0]";
        }
        return os;
    }
    explicit operator std::string() {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }
};

struct SegmentedObject : DetectedObject {
    cv::Mat mask;

    friend std::ostream& operator<<(std::ostream& os, const SegmentedObject& prediction) {
        return os << static_cast<const DetectedObject&>(prediction) << ", " << cv::countNonZero(prediction.mask > 0.5);
    }
};

struct SegmentedObjectWithRects : SegmentedObject {
    cv::RotatedRect rotated_rect;

    SegmentedObjectWithRects(const SegmentedObject& segmented_object) : SegmentedObject(segmented_object) {}

    friend std::ostream& operator<<(std::ostream& os, const SegmentedObjectWithRects& prediction) {
        os << static_cast<const SegmentedObject&>(prediction) << std::fixed << std::setprecision(3);
        auto rect = prediction.rotated_rect;
        os << ", RotatedRect: " << rect.center.x << ' ' << rect.center.y << ' ' << rect.size.width << ' '
           << rect.size.height << ' ' << rect.angle;
        return os;
    }
};

struct InstanceSegmentationResult {
    std::vector<SegmentedObject> segmentedObjects;
    std::vector<cv::Mat_<std::uint8_t>> saliency_map;
    ov::Tensor feature_vector;
};
