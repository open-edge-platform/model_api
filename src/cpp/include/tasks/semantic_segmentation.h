#pragma once
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

#include "adapters/inference_adapter.h"
#include "tasks/results.h"
#include "utils/preprocessing.h"
#include "utils/vision_pipeline.h"

class SemanticSegmentation {
public:
    VisionPipeline<SemanticSegmentationResult> pipeline;
    std::shared_ptr<InferenceAdapter> adapter;
    SemanticSegmentation(std::shared_ptr<InferenceAdapter> adapter): adapter(adapter) {
        pipeline = VisionPipeline<SemanticSegmentationResult>(adapter, 
            [&](cv::Mat image) { return preprocess(image);},
            [&](InferenceResult result) { return postprocess(result);}
        );

        auto config = adapter->getModelConfig();
        auto iter = config.find("labels");
        if (iter != config.end()) {
            labels = iter->second.as<std::vector<std::string>>();
        } else {
            std::cout << "could not find labels from model config" << std::endl;
        }

        {
            auto iter = config.find("soft_threshold");
            if (iter != config.end()) {
                soft_threshold = iter->second.as<float>();
            }
        }

        {
            auto iter = config.find("blur_strength");
            if (iter != config.end()) {
                blur_strength = iter->second.as<int>();
            }
        }

    }

    static cv::Size serialize(std::shared_ptr<ov::Model>& ov_model);
    static SemanticSegmentation load(const std::string& model_path);

    std::map<std::string, ov::Tensor> preprocess(cv::Mat);
    SemanticSegmentationResult postprocess(InferenceResult& infResult);
    std::vector<Contour> getContours(const SemanticSegmentationResult& result);

    SemanticSegmentationResult infer(cv::Mat image);
    void inferAsync(cv::Mat image, ov::AnyMap user_data);
    void setCallback(std::function<void(SemanticSegmentationResult, ov::AnyMap)>);
    std::vector<SemanticSegmentationResult> inferBatch(std::vector<cv::Mat> image);
private:
    cv::Mat create_hard_prediction_from_soft_prediction(cv::Mat, float threshold, int blur_strength);

    //from config
    int blur_strength = -1;
    float soft_threshold = -std::numeric_limits<float>::infinity();
    bool return_soft_prediction = true;

    std::vector<std::string> labels;

    std::string getLabelName(size_t labelID) {
        return labelID < labels.size() ? labels[labelID] : std::string("Label #") + std::to_string(labelID);
    }
};