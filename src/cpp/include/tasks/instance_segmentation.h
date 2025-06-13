#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "adapters/inference_adapter.h"
#include "tasks/results.h"
#include "utils/vision_pipeline.h"

class InstanceSegmentation {
public:
    std::shared_ptr<InferenceAdapter> adapter;
    VisionPipeline<InstanceSegmentationResult> pipeline;

    InstanceSegmentation(std::shared_ptr<InferenceAdapter> adapter, cv::Size input_shape)
        : adapter(adapter),
          input_shape(input_shape) {
        pipeline = VisionPipeline<InstanceSegmentationResult>(
            adapter,
            [&](cv::Mat image) {
                return preprocess(image);
            },
            [&](InferenceResult result) {
                return postprocess(result);
            });

        auto config = adapter->getModelConfig();
        auto iter = config.find("labels");
        if (iter != config.end()) {
            labels = iter->second.as<std::vector<std::string>>();
        } else {
            std::cout << "could not find labels from model config" << std::endl;
        }

        {
            auto iter = config.find("confidence_threshold");
            if (iter != config.end()) {
                confidence_threshold = iter->second.as<float>();
            }
        }
    }
    static cv::Size serialize(std::shared_ptr<ov::Model>& ov_model);
    static InstanceSegmentation load(const std::string& model_path);

    InstanceSegmentationResult infer(cv::Mat image);
    std::vector<InstanceSegmentationResult> inferBatch(std::vector<cv::Mat> image);

    std::map<std::string, ov::Tensor> preprocess(cv::Mat);
    InstanceSegmentationResult postprocess(InferenceResult& infResult);

    static std::vector<SegmentedObjectWithRects> getRotatedRectangles(const InstanceSegmentationResult& result);
    static std::vector<Contour> getContours(const std::vector<SegmentedObject>& objects);

    bool postprocess_semantic_masks = true;

private:
    std::vector<std::string> labels;
    std::string getLabelName(size_t labelID) {
        return labelID < labels.size() ? labels[labelID] : std::string("Label #") + std::to_string(labelID);
    }

    cv::Size input_shape;
    float confidence_threshold = 0.5f;
};
