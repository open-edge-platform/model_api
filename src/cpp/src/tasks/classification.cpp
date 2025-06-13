/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tasks/classification.h"

#include "adapters/openvino_adapter.h"
#include "utils/tensor.h"

namespace {
    constexpr char indices_name[]{"indices"};
    constexpr char raw_scores_name[]{"raw_scores"};
    constexpr char scores_name[]{"scores"};
}

cv::Size Classification::serialize(std::shared_ptr<ov::Model>& ov_model) {

    return {};
}

Classification Classification::load(const std::string& model_path) {
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
        origin_input_shape = Classification::serialize(model);
    }
    auto adapter = std::make_shared<OpenVINOInferenceAdapter>();
    adapter->loadModel(model, core, "AUTO");
    return Classification(adapter, origin_input_shape);
}

ClassificationResult Classification::infer(cv::Mat image) {
    return pipeline.infer(image);
}

std::vector<InstanceSegmentationResult> Classification::inferBatch(std::vector<cv::Mat> image) {

}

std::map<std::string, ov::Tensor> Classification::preprocess(cv::Mat image) {
    std::map<std::string, ov::Tensor> input = {};
    input.emplace(adapter->getInputNames()[0], utils::wrapMat2Tensor(image));
    return input;
}

ClassificationResult Classification::postprocess(InferenceResult& infResult) {
    ClassificationResult result;
    if (multilabel) {
        result = get_multilabel_predictions(infResult, output_raw_scores);
    } else if (hierarchical) {
        result = get_hierarchical_predictions(infResult, output_raw_scores);
    } else {
        result = get_multiclass_predictions(infResult, output_raw_scores);
    }

    return result;
}


ClassificationResult Classification::get_multilabel_predictions(InferenceResult& infResult, bool add_raw_scores) {

}

ClassificationResult Classification::get_multiclass_predictions(InferenceResult& infResult, bool add_raw_scores) {
    const ov::Tensor& indicesTensor = infResult.data.find(indices_name)->second;
    const int* indicesPtr = indicesTensor.data<int>();
    const ov::Tensor& scoresTensor = infResult.data.find(scores_name)->second;
    const float* scoresPtr = scoresTensor.data<float>();

    ClassificationResult result;

    if (add_raw_scores) {
        const ov::Tensor& logitsTensor = infResult.data.find(raw_scores_name)->second;
        result.raw_scores = ov::Tensor(logitsTensor.get_element_type(), logitsTensor.get_shape());
        logitsTensor.copy_to(result.raw_scores);
        result.raw_scores.set_shape(ov::Shape({result.raw_scores.get_size()}));
    }

    result.topLabels.reserve(scoresTensor.get_size());
    for (size_t i = 0; i < scoresTensor.get_size(); ++i) {
        int ind = indicesPtr[i];
        if (ind < 0 || ind >= static_cast<int>(labels.size())) {
            throw std::runtime_error("Invalid index for the class label is found during postprocessing");
        }
        result.topLabels.emplace_back(ind, labels[ind], scoresPtr[i]);
    }

    return result;

}

ClassificationResult Classification::get_hierarchical_predictions(InferenceResult& infResult, bool add_raw_scores) {

}