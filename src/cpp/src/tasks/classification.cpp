/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tasks/classification.h"

#include <openvino/op/softmax.hpp>
#include <openvino/op/topk.hpp>

#include "adapters/openvino_adapter.h"
#include "utils/preprocessing.h"
#include "utils/tensor.h"

namespace {
constexpr char indices_name[]{"indices"};
constexpr char raw_scores_name[]{"raw_scores"};
constexpr char scores_name[]{"scores"};
constexpr char saliency_map_name[]{"saliency_map"};
constexpr char feature_vector_name[]{"feature_vector"};

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
std::vector<std::string> get_non_xai_names(const std::vector<std::string>& outputs) {
    std::vector<std::string> outputNames;
    outputNames.reserve(std::max(1, int(outputs.size()) - 2));
    for (const auto& output : outputs) {
        if (output.find(saliency_map_name) != std::string::npos) {
            continue;
        }
        if (output.find(feature_vector_name) != std::string::npos) {
            continue;
        }
        outputNames.push_back(output);
    }
    return outputNames;
}

std::vector<size_t> get_non_xai_output_indices(const std::vector<ov::Output<ov::Node>>& outputs) {
    std::vector<size_t> outputIndices;
    outputIndices.reserve(std::max(1, int(outputs.size()) - 2));
    size_t idx = 0;
    for (const ov::Output<ov::Node>& output : outputs) {
        bool is_xai =
            output.get_names().count(saliency_map_name) > 0 || output.get_names().count(feature_vector_name) > 0;
        if (!is_xai) {
            outputIndices.push_back(idx);
        }
        ++idx;
    }
    return outputIndices;
}
}  // namespace

void Classification::serialize(std::shared_ptr<ov::Model>& ov_model) {
    if (utils::model_has_embedded_processing(ov_model)) {
        std::cout << "model already was serialized" << std::endl;
        return;
    }
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input  ------------------------------------------------------
    auto config = ov_model->has_rt_info("model_info") ? ov_model->get_rt_info<ov::AnyMap>("model_info") : ov::AnyMap{};
    std::string layout = "";
    layout = utils::get_from_any_maps("layout", config, {}, layout);
    auto inputsLayouts = utils::parseLayoutString(layout);

    if (ov_model->inputs().size() != 1) {
        throw std::logic_error("Classification model wrapper supports topologies with only 1 input");
    }
    const auto& input = ov_model->input();

    auto inputName = input.get_any_name();

    const ov::Shape& inputShape = input.get_partial_shape().get_max_shape();
    const ov::Layout& inputLayout = utils::getInputLayout(input, inputsLayouts);

    auto interpolation_mode = cv::INTER_LINEAR;
    utils::RESIZE_MODE resize_mode = utils::RESIZE_FILL;

    std::vector<float> scale_values;
    std::vector<float> mean_values;
    scale_values = utils::get_from_any_maps("scale_values", config, ov::AnyMap{}, scale_values);
    mean_values = utils::get_from_any_maps("mean_values", config, ov::AnyMap{}, mean_values);
    uint8_t pad_value = 0;
    bool reverse_input_channels = false;

    auto input_shape =
        ov::Shape{inputShape[ov::layout::width_idx(inputLayout)], inputShape[ov::layout::height_idx(inputLayout)]};

    ov_model = utils::embedProcessing(ov_model,
                                      inputName,
                                      inputLayout,
                                      resize_mode,
                                      interpolation_mode,
                                      input_shape,
                                      pad_value,
                                      reverse_input_channels,
                                      mean_values,
                                      scale_values);

    // --------------------------- Prepare output  -----------------------------------------------------
    if (ov_model->outputs().size() > 5) {
        throw std::logic_error("Classification model wrapper supports topologies with up to 4 outputs");
    }

    size_t topk = 1;
    topk = utils::get_from_any_maps("topk", config, {}, topk);
    std::vector<std::string> labels;
    labels = utils::get_from_any_maps("labels", config, ov::AnyMap{}, labels);

    auto non_xai_idx = get_non_xai_output_indices(ov_model->outputs());
    if (non_xai_idx.size() == 1) {
        const ov::Shape& outputShape = ov_model->outputs()[non_xai_idx[0]].get_partial_shape().get_max_shape();
        if (outputShape.size() != 2 && outputShape.size() != 4) {
            throw std::logic_error("Classification model wrapper supports topologies only with"
                                   " 2-dimensional or 4-dimensional output");
        }

        const ov::Layout outputLayout("NCHW");
        if (outputShape.size() == 4 && (outputShape[ov::layout::height_idx(outputLayout)] != 1 ||
                                        outputShape[ov::layout::width_idx(outputLayout)] != 1)) {
            throw std::logic_error("Classification model wrapper supports topologies only"
                                   " with 4-dimensional output which has last two dimensions of size 1");
        }

        size_t classesNum = outputShape[ov::layout::channels_idx(outputLayout)];
        if (topk > classesNum) {
            throw std::logic_error("The model provides " + std::to_string(classesNum) + " classes, but " +
                                   std::to_string(topk) + " labels are requested to be predicted");
        }
        if (classesNum != labels.size()) {
            throw std::logic_error("Model's number of classes and parsed labels must match (" +
                                   std::to_string(outputShape[1]) + " and " + std::to_string(labels.size()) + ')');
        }
    }

    bool multilabel;
    bool hierarchical;
    bool output_raw_scores;
    multilabel = utils::get_from_any_maps("multilabel", config, {}, multilabel);
    hierarchical = utils::get_from_any_maps("hierarchical", config, {}, hierarchical);
    output_raw_scores = utils::get_from_any_maps("output_raw_scores", config, {}, output_raw_scores);

    auto multiclass = !multilabel && !hierarchical;
    if (multiclass) {
        addOrFindSoftmaxAndTopkOutputs(ov_model, topk, output_raw_scores);
    }

    ov_model->set_rt_info(true, "model_info", "embedded_processing");
    ov_model->set_rt_info(input_shape[0], "model_info", "orig_width");
    ov_model->set_rt_info(input_shape[1], "model_info", "orig_height");
}

Classification Classification::create_model(const std::string& model_path,
                                            const ov::AnyMap& user_config,
                                            bool preload,
                                            const std::string& device) {
    auto adapter = std::make_shared<OpenVINOInferenceAdapter>();
    adapter->loadModel(model_path, device, user_config, false);

    std::string model_type;
    model_type = utils::get_from_any_maps("model_type", adapter->getModelConfig(), user_config, model_type);

    if (model_type.empty() || model_type != "Classification") {
        throw std::runtime_error("Incorrect or unsupported model_type, expected: Classification");
    }
    adapter->applyModelTransform(Classification::serialize);
    if (preload) {
        adapter->compileModel(device, user_config);
    }

    return Classification(adapter, user_config);
}

ClassificationResult Classification::infer(cv::Mat image) {
    return pipeline.infer(image);
}

std::vector<ClassificationResult> Classification::inferBatch(std::vector<cv::Mat> images) {
    return pipeline.inferBatch(images);
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

    auto saliency_map_iter = infResult.data.find(saliency_map_name);
    if (saliency_map_iter != infResult.data.end()) {
        result.saliency_map = std::move(saliency_map_iter->second);
        result.saliency_map = reorder_saliency_maps(result.saliency_map);
    }
    auto feature_vector_iter = infResult.data.find(feature_vector_name);
    if (feature_vector_iter != infResult.data.end()) {
        result.feature_vector = std::move(feature_vector_iter->second);
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

    auto logitsTensorName = get_non_xai_names(adapter->getOutputNames()).front();
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

ov::Tensor Classification::reorder_saliency_maps(const ov::Tensor& source_maps) {
    if (!hierarchical || source_maps.get_shape().size() == 1) {
        return source_maps;
    }

    auto reordered_maps = ov::Tensor(source_maps.get_element_type(), source_maps.get_shape());
    const std::uint8_t* source_maps_ptr = static_cast<std::uint8_t*>(source_maps.data());
    std::uint8_t* reordered_maps_ptr = static_cast<std::uint8_t*>(reordered_maps.data());

    size_t shape_offset = (source_maps.get_shape().size() == 4) ? 1 : 0;
    size_t map_byte_size = source_maps.get_element_type().size() * source_maps.get_shape()[shape_offset + 1] *
                           source_maps.get_shape()[shape_offset + 2];

    for (size_t i = 0; i < source_maps.get_shape()[shape_offset]; ++i) {
        size_t new_index = hierarchical_info.label_to_idx[hierarchical_info.logit_idx_to_label[i]];
        std::copy_n(source_maps_ptr + i * map_byte_size, map_byte_size, reordered_maps_ptr + new_index * map_byte_size);
    }

    return reordered_maps;
}

void Classification::addOrFindSoftmaxAndTopkOutputs(std::shared_ptr<ov::Model>& model,
                                                    size_t topk,
                                                    bool add_raw_scores) {
    auto nodes = model->get_ops();
    std::shared_ptr<ov::Node> softmaxNode;
    for (size_t i = 0; i < model->outputs().size(); ++i) {
        auto output_node = model->get_output_op(i)->input(0).get_source_output().get_node_shared_ptr();
        if (std::string(output_node->get_type_name()) == "Softmax") {
            softmaxNode = output_node;
        } else if (std::string(output_node->get_type_name()) == "TopK") {
            return;
        }
    }

    if (!softmaxNode) {
        auto logitsNode = model->get_output_op(0)->input(0).get_source_output().get_node();
        softmaxNode = std::make_shared<ov::op::v1::Softmax>(logitsNode->output(0), 1);
    }

    const auto k = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<size_t>{topk});
    std::shared_ptr<ov::Node> topkNode = std::make_shared<ov::op::v3::TopK>(softmaxNode,
                                                                            k,
                                                                            1,
                                                                            ov::op::v3::TopK::Mode::MAX,
                                                                            ov::op::v3::TopK::SortType::SORT_VALUES);

    auto indices = topkNode->output(0);
    auto scores = topkNode->output(1);
    ov::OutputVector outputs_vector;
    if (add_raw_scores) {
        auto raw_scores = softmaxNode->output(0);
        outputs_vector = {scores, indices, raw_scores};
    } else {
        outputs_vector = {scores, indices};
    }
    for (const ov::Output<ov::Node>& output : model->outputs()) {
        if (output.get_names().count(saliency_map_name) > 0 || output.get_names().count(feature_vector_name) > 0) {
            outputs_vector.push_back(output);
        }
    }

    auto source_rt_info =
        model->has_rt_info("model_info") ? model->get_rt_info<ov::AnyMap>("model_info") : ov::AnyMap{};
    model = std::make_shared<ov::Model>(outputs_vector, model->get_parameters(), "classification");

    // preserve extra model_info items
    for (const auto& k : source_rt_info) {
        model->set_rt_info(k.second, "model_info", k.first);
    }

    // manually set output tensors name for created topK node
    model->outputs()[0].set_names({indices_name});
    model->outputs()[1].set_names({scores_name});
    if (add_raw_scores) {
        model->outputs()[2].set_names({raw_scores_name});
    }

    // set output precisions
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    ppp.output(indices_name).tensor().set_element_type(ov::element::i32);
    ppp.output(scores_name).tensor().set_element_type(ov::element::f32);
    if (add_raw_scores) {
        ppp.output(raw_scores_name).tensor().set_element_type(ov::element::f32);
    }
    model = ppp.build();
}
