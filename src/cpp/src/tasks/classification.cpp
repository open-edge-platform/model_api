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
    
    float sigmoid(float x) noexcept {
        return 1.0f / (1.0f + std::exp(-x));
    }

    size_t fargmax(const float* x_start, const float* x_end) noexcept {
        size_t argmax = 0;

        for (const float* iter = x_start; iter < x_end; ++iter) {
            if (x_start[argmax] < *iter) {
                argmax = iter - x_start;
            }
        }

        return argmax;
    }

    void softmax(float* x_start, float* x_end, float eps = 1e-9) {
        if (x_start == x_end) {
            return;
        }

        float x_max = *std::max_element(x_start, x_end);
        float x_sum = 0.f;
        for (auto it = x_start; it < x_end; ++it) {
            *it = exp(*it - x_max);
            x_sum += *it;
        }

        for (auto it = x_start; it < x_end; ++it) {
            *it /= x_sum + eps;
        }
    }
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
    auto logitsTensorName = adapter->getOutputNames().front();
    const ov::Tensor& logitsTensor = infResult.data.find(logitsTensorName)->second;
    const float* logitsPtr = logitsTensor.data<float>();

    ClassificationResult result;
    auto raw_scores = ov::Tensor();
    float* raw_scoresPtr = nullptr;
    if (add_raw_scores) {
        raw_scores = ov::Tensor(logitsTensor.get_element_type(), logitsTensor.get_shape());
        raw_scoresPtr = raw_scores.data<float>();
        result.raw_scores = raw_scores;
    }

    result.topLabels.reserve(labels.size());
    for (size_t i = 0; i < labels.size(); ++i) {
        float score = sigmoid(logitsPtr[i]);
        if (score > confidence_threshold) {
            result.topLabels.emplace_back(i, labels[i], score);
        }
        if (add_raw_scores) {
            raw_scoresPtr[i] = score;
        }
    }

    return result;

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
    ClassificationResult result;

    auto logitsTensorName = adapter->getOutputNames().front();
    const ov::Tensor& logitsTensor = infResult.data.find(logitsTensorName)->second;
    float* logitsPtr = logitsTensor.data<float>();

    auto raw_scores = ov::Tensor();
    float* raw_scoresPtr = nullptr;
    if (add_raw_scores) {
        raw_scores = ov::Tensor(logitsTensor.get_element_type(), logitsTensor.get_shape());
        logitsTensor.copy_to(raw_scores);
        raw_scoresPtr = raw_scores.data<float>();
        result.raw_scores = raw_scores;
    }

    std::vector<std::reference_wrapper<std::string>> predicted_labels;
    std::vector<float> predicted_scores;

    predicted_labels.reserve(hierarchical_info.num_multiclass_heads + hierarchical_info.num_multilabel_heads);
    predicted_scores.reserve(hierarchical_info.num_multiclass_heads + hierarchical_info.num_multilabel_heads);

    for (size_t i = 0; i < hierarchical_info.num_multiclass_heads; ++i) {
        const auto& logits_range = hierarchical_info.head_idx_to_logits_range[i];
        softmax(logitsPtr + logits_range.first, logitsPtr + logits_range.second);
        if (add_raw_scores) {
            softmax(raw_scoresPtr + logits_range.first, raw_scoresPtr + logits_range.second);
        }
        size_t j = fargmax(logitsPtr + logits_range.first, logitsPtr + logits_range.second);
        predicted_labels.push_back(hierarchical_info.all_groups[i][j]);
        predicted_scores.push_back(logitsPtr[logits_range.first + j]);
    }

    if (hierarchical_info.num_multilabel_heads) {
        const float* mlc_logitsPtr = logitsPtr + hierarchical_info.num_single_label_classes;

        for (size_t i = 0; i < hierarchical_info.num_multilabel_heads; ++i) {
            float score = sigmoid(mlc_logitsPtr[i]);
            if (score > confidence_threshold) {
                predicted_scores.push_back(score);
                predicted_labels.push_back(hierarchical_info.all_groups[hierarchical_info.num_multiclass_heads + i][0]);
            }
            if (add_raw_scores) {
                raw_scoresPtr[hierarchical_info.num_single_label_classes + i] = score;
            }
        }
    }

    auto resolved_labels = resolver->resolve_labels(predicted_labels, predicted_scores);

    result.topLabels.reserve(resolved_labels.size());
    for (const auto& label : resolved_labels) {
        result.topLabels.emplace_back(hierarchical_info.label_to_idx[label.first], label.first, label.second);
    }

    return result;

}