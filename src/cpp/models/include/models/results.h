/*
 * Copyright (C) 2020-2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <map>
#include <memory>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "internal_model_data.h"

struct MetaData;

struct ResultBase {
    ResultBase(int64_t frameId = -1, const std::shared_ptr<MetaData>& metaData = nullptr)
        : frameId(frameId),
          metaData(metaData) {}
    virtual ~ResultBase() {}

    int64_t frameId;

    std::shared_ptr<MetaData> metaData;
    bool IsEmpty() {
        return frameId < 0;
    }

    template <class T>
    T& asRef() {
        return dynamic_cast<T&>(*this);
    }

    template <class T>
    const T& asRef() const {
        return dynamic_cast<const T&>(*this);
    }
};

struct InferenceResult : public ResultBase {
    std::shared_ptr<InternalModelData> internalModelData;
    std::map<std::string, ov::Tensor> outputsData;

    /// Returns the first output tensor
    /// This function is a useful addition to direct access to outputs list as many models have only one output
    /// @returns first output tensor
    ov::Tensor getFirstOutputTensor() {
        if (outputsData.empty()) {
            throw std::out_of_range("Outputs map is empty.");
        }
        return outputsData.begin()->second;
    }

    /// Returns true if object contains no valid data
    /// @returns true if object contains no valid data
    bool IsEmpty() {
        return outputsData.empty();
    }
};

struct DetectedKeypoints {
    std::vector<cv::Point2f> keypoints;
    std::vector<float> scores;

    friend std::ostream& operator<<(std::ostream& os, const DetectedKeypoints& prediction) {
        float kp_x_sum = 0.f;
        for (const cv::Point2f& keypoint : prediction.keypoints) {
            kp_x_sum += keypoint.x;
        }
        float scores_sum = std::accumulate(prediction.scores.begin(), prediction.scores.end(), 0.f);

        os << "keypoints: (" << prediction.keypoints.size() << ", 2), keypoints_x_sum: ";
        os << std::fixed << std::setprecision(3) << kp_x_sum << ", scores: (" << prediction.scores.size() << ",) "
           << std::fixed << std::setprecision(3) << scores_sum;
        return os;
    }

    explicit operator std::string() {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }
};

class Label {
public:
    Label() {}
    Label(int id, std::string name):  id(id), name(name) {}

    int id;
    std::string name;

    friend std::ostream& operator<< (std::ostream& os, const Label& label) {
        return os << label.id << " (" << label.name << ")";
    }
};

class LabelScore {
public:
    LabelScore() {}
    LabelScore(int id, std::string name, float score): label(Label(id, name)), score(score) {}
    LabelScore(Label label, float score):  label(label), score(score) {}

    Label label;
    float score;

    friend std::ostream& operator<< (std::ostream& os, const LabelScore& label) {
        return os << label.label << ": " << std::fixed << std::setprecision(3) << label.score;
    }
};

class Mask {
public:
    Mask(LabelScore label, cv::Rect roi, cv::Mat mask): label(label), roi(roi), mask(mask) {}

    LabelScore label;
    cv::Rect roi;
    cv::Mat mask;

    friend std::ostream& operator<< (std::ostream& os, const Mask& mask) {

        double min_mask, max_mask;
        cv::minMaxLoc(mask.mask, &min_mask, &max_mask);
        os << mask.label << mask.roi << " min:" << min_mask << " max:" << max_mask << ";";
        return os;
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

static inline std::vector<Contour> getContours(const std::vector<Mask>& segmentedObjects) {
    std::vector<Contour> combined_contours;
    std::vector<std::vector<cv::Point>> contours;
    for (const Mask& obj : segmentedObjects) {
        cv::findContours(obj.mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        // Assuming one contour output for findContours. Based on OTX this is a safe
        // assumption
        if (contours.size() != 1) {
            throw std::runtime_error("findContours() must have returned only one contour");
        }
        combined_contours.push_back({obj.label.label.name, obj.label.score, contours[0]});
    }
    return combined_contours;
}

class Box {
public:
    Box(cv::Rect shape, std::vector<LabelScore> labels): shape(shape), labels(labels) {}
    cv::Rect shape;
    std::vector<LabelScore> labels;

    friend std::ostream& operator<< (std::ostream& os, const Box& box) {

        os << int(box.shape.x) << ", " << int(box.shape.y) << ", " << int(box.shape.x + box.shape.width) << ", "
                  << int(box.shape.y + box.shape.height) << ", ";
        for (size_t i = 0; i < box.labels.size(); i++) {
            os << box.labels[i];
            if (i == box.labels.size() - 1)  {
                os << "; ";
            } else {
                os << ", ";
            }
        }


        return os;
    }

    explicit operator std::string() {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }
};

class RotatedRect {
public:
    LabelScore label;
    cv::RotatedRect shape;

    friend std::ostream& operator<< (std::ostream& os, const RotatedRect& box) {

        os << "RotatedRect: ";
        os << std::fixed << std::setprecision(3);
        os << box.shape.center.x << ", " << box.shape.center.y << ", " << box.shape.size.width << ", "
                  << box.shape.size.height << ", " << box.shape.angle;
        os << box.label << "; ";
        return os;
    }

    explicit operator std::string() {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }
};

static inline std::vector<RotatedRect> get_rotated_rects(std::vector<Mask> masks) {
    std::vector<RotatedRect> result;
    result.reserve(masks.size());
    for (const Mask& m : masks) {
        cv::Mat mask;
        m.mask.convertTo(mask, CV_8UC1);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Point> contour = {};
        for (size_t i = 0; i < contours.size(); i++) {
            contour.insert(contour.end(), contours[i].begin(), contours[i].end());
        }
        if (contour.size() > 0) {
            std::vector<cv::Point> hull;
            cv::convexHull(contour, hull);

            result.push_back(RotatedRect{m.label, cv::minAreaRect(hull)});
        }
    }
    return result;
}

class Scene {
public:
    Scene(int64_t frameId = -1, const std::shared_ptr<MetaData>& metaData = nullptr)
        : frameId(frameId),
          metaData(metaData) {}

    int64_t frameId;
    std::shared_ptr<MetaData> metaData;

    std::vector<Box> boxes;
    std::vector<DetectedKeypoints> poses;

    std::vector<cv::Mat> saliency_maps;
    std::vector<ov::Tensor> feature_vectors;

    std::vector<Mask> masks;

    std::map<std::string, ov::Tensor> additional_tensors;

    friend std::ostream& operator<<(std::ostream& os, const Scene& scene) {
        for (auto& box: scene.boxes) {
            os << box;
        }

        for (auto& pose: scene.poses) {
            os << pose;
        }

        if (scene.saliency_maps.empty()){
            os << "[0]; ";
        } else {
            os << "[1," << scene.saliency_maps.size() << "," << scene.saliency_maps[0].rows << "," << scene.saliency_maps[0].cols << "]; ";
        }

        for (auto& m: scene.masks) {
            os << m;
        }

        if (scene.feature_vectors.empty()){
            os << "[0]";
        } else {
            for (auto& feature_vector: scene.feature_vectors){
                os << feature_vector.get_shape();
            }
        }

        for (auto& v: scene.additional_tensors) {
            os << ", " << v.second.get_shape();
        }

        return os;
    }

    explicit operator std::string() {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }
};
