/*
// Copyright (C) 2020-2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "models/classification_model.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openvino/op/softmax.hpp>
#include <openvino/op/topk.hpp>
#include <openvino/openvino.hpp>

#include <nlohmann/json.hpp>

#include <utils/slog.hpp>

#include "models/results.h"
#include "models/input_data.h"

std::string ClassificationModel::ModelType = "Classification";

namespace {
constexpr char indices_name[]{"indices"};
constexpr char scores_name[]{"scores"};
constexpr char saliency_map_name[]{"saliency_map"};
constexpr char feature_vector_name[]{"feature_vector"};
constexpr char raw_scores_name[]{"raw_scores"};

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

void addOrFindSoftmaxAndTopkOutputs(std::shared_ptr<ov::Model>& model, size_t topk, bool add_raw_scores) {
    auto nodes = model->get_ops();
    std::shared_ptr<ov::Node> softmaxNode;
    for (size_t i = 0; i < model->outputs().size(); ++i) {
        auto output_node = model->get_output_op(i)->input(0).get_source_output().get_node_shared_ptr();
        if (std::string(output_node->get_type_name()) == "Softmax") {
            softmaxNode = output_node;
        }
        else if (std::string(output_node->get_type_name()) == "TopK") {
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
    }
    else {
        outputs_vector = {scores, indices};
    }
    for (const ov::Output<ov::Node>& output : model->outputs()) {
        if (output.get_names().count(saliency_map_name) > 0 || output.get_names().count(feature_vector_name) > 0) {
            outputs_vector.push_back(output);
        }
    }
    model = std::make_shared<ov::Model>(outputs_vector, model->get_parameters(), "classification");

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

std::vector<std::string> get_non_xai_names(const std::vector<ov::Output<ov::Node>>& outputs) {
    std::vector<std::string> outputNames;
    outputNames.reserve(std::max(1, int(outputs.size()) - 2));
    for (const ov::Output<ov::Node>& output : outputs) {
        if (output.get_names().count(saliency_map_name) > 0) {
            continue;
        } if (output.get_names().count(feature_vector_name) > 0) {
            continue;
        }
        outputNames.push_back(output.get_any_name());
    }
    return outputNames;
}

void append_xai_names(const std::vector<ov::Output<ov::Node>>& outputs, std::vector<std::string>& outputNames) {
    for (const ov::Output<ov::Node>& output : outputs) {
        if (output.get_names().count(saliency_map_name) > 0) {
            outputNames.emplace_back(saliency_map_name);
        } else if (output.get_names().count(feature_vector_name) > 0) {
            outputNames.push_back(feature_vector_name);
        }
    }
}
}

void ClassificationModel::init_from_config(const ov::AnyMap& top_priority, const ov::AnyMap& mid_priority) {
    topk = get_from_any_maps("topk", top_priority, mid_priority, topk);
    confidence_threshold = get_from_any_maps("confidence_threshold", top_priority, mid_priority, confidence_threshold);
    multilabel = get_from_any_maps("multilabel", top_priority, mid_priority, multilabel);
    output_raw_scores = get_from_any_maps("output_raw_scores", top_priority, mid_priority, output_raw_scores);
    hierarchical = get_from_any_maps("hierarchical", top_priority, mid_priority, hierarchical);
    hierarchical_config = get_from_any_maps("hierarchical_config", top_priority, mid_priority, hierarchical_config);
    if (hierarchical) {
        if (hierarchical_config.empty()) {
            throw std::runtime_error("Error: empty hierarchical classification config");
        }
        hierarchical_info = HierarchicalConfig(hierarchical_config);
        resolver = GreedyLabelsResolver(hierarchical_info);
    }
}

ClassificationModel::ClassificationModel(std::shared_ptr<ov::Model>& model, const ov::AnyMap& configuration)
        : ImageModel(model, configuration) {
    init_from_config(configuration, model->has_rt_info("model_info") ? model->get_rt_info<ov::AnyMap>("model_info") : ov::AnyMap{});
}

ClassificationModel::ClassificationModel(std::shared_ptr<InferenceAdapter>& adapter)
        : ImageModel(adapter) {
    init_from_config(adapter->getModelConfig(), ov::AnyMap{});
}

void ClassificationModel::updateModelInfo() {
    ImageModel::updateModelInfo();

    model->set_rt_info(ClassificationModel::ModelType, "model_info", "model_type");
    model->set_rt_info(topk, "model_info", "topk");
    model->set_rt_info(multilabel, "model_info", "multilabel");
    model->set_rt_info(hierarchical, "model_info", "hierarchical");
    model->set_rt_info(output_raw_scores, "model_info", "output_raw_scores");
    model->set_rt_info(confidence_threshold, "model_info", "confidence_threshold");
    model->set_rt_info(hierarchical_config, "model_info", "hierarchical_config");
}

std::unique_ptr<ClassificationModel> ClassificationModel::create_model(const std::string& modelFile, const ov::AnyMap& configuration, bool preload, const std::string& device) {
    auto core = ov::Core();
    std::shared_ptr<ov::Model> model = core.read_model(modelFile);

    // Check model_type in the rt_info, ignore configuration
    std::string model_type = ClassificationModel::ModelType;
    try {
        if (model->has_rt_info("model_info", "model_type")) {
            model_type = model->get_rt_info<std::string>("model_info", "model_type");
        }
    } catch (const std::exception&) {
        slog::warn << "Model type is not specified in the rt_info, use default model type: " << model_type << slog::endl;
    }

    if (model_type != ClassificationModel::ModelType) {
        throw std::runtime_error("Incorrect or unsupported model_type is provided in the model_info section: " + model_type);
    }

    std::unique_ptr<ClassificationModel> classifier{new ClassificationModel(model, configuration)};
    classifier->prepare();
    if (preload) {
        classifier->load(core, device);
    }
    return classifier;
}

std::unique_ptr<ClassificationModel> ClassificationModel::create_model(std::shared_ptr<InferenceAdapter>& adapter) {
    const ov::AnyMap& configuration = adapter->getModelConfig();
    auto model_type_iter = configuration.find("model_type");
    std::string model_type = ClassificationModel::ModelType;
    if (model_type_iter != configuration.end()) {
        model_type = model_type_iter->second.as<std::string>();
    }

    if (model_type != ClassificationModel::ModelType) {
        throw std::runtime_error("Incorrect or unsupported model_type is provided: " + model_type);
    }

    std::unique_ptr<ClassificationModel> classifier{new ClassificationModel(adapter)};
    return classifier;
}

std::unique_ptr<ResultBase> ClassificationModel::postprocess(InferenceResult& infResult) {
    std::unique_ptr<ResultBase> result;
    if (multilabel) {
        result = get_multilabel_predictions(infResult, output_raw_scores);
    } else if (hierarchical) {
        result = get_hierarchical_predictions(infResult, output_raw_scores);
    } else {
        result = get_multiclass_predictions(infResult, output_raw_scores);
    }

    ClassificationResult* cls_res = static_cast<ClassificationResult*>(result.get());
    auto saliency_map_iter = infResult.outputsData.find(saliency_map_name);
    if (saliency_map_iter != infResult.outputsData.end()) {
        cls_res->saliency_map = std::move(saliency_map_iter->second);
    }
    auto feature_vector_iter = infResult.outputsData.find(feature_vector_name);
    if (feature_vector_iter != infResult.outputsData.end()) {
        cls_res->feature_vector = std::move(feature_vector_iter->second);
    }
    return result;
}

std::unique_ptr<ResultBase> ClassificationModel::get_multilabel_predictions(InferenceResult& infResult, bool add_raw_scores) {
    const ov::Tensor& logitsTensor = infResult.outputsData.find(outputNames[0])->second;
    const float* logitsPtr = logitsTensor.data<float>();

    ClassificationResult* result = new ClassificationResult(infResult.frameId, infResult.metaData);
    auto retVal = std::unique_ptr<ResultBase>(result);

    auto raw_scores = ov::Tensor();
    float* raw_scoresPtr = nullptr;
    if (add_raw_scores) {
        raw_scores = ov::Tensor(logitsTensor.get_element_type(), logitsTensor.get_shape());
        raw_scoresPtr = raw_scores.data<float>();
        result->raw_scores = raw_scores;
    }

    result->topLabels.reserve(labels.size());
    for (size_t i = 0; i < labels.size(); ++i) {
        float score = sigmoid(logitsPtr[i]);
        if (score > confidence_threshold) {
            result->topLabels.emplace_back(i, labels[i], score);
        }
        if (add_raw_scores) {
            raw_scoresPtr[i] = score;
        }
    }

    return retVal;
}

std::unique_ptr<ResultBase> ClassificationModel::get_hierarchical_predictions(InferenceResult& infResult, bool add_raw_scores) {
    ClassificationResult* result = new ClassificationResult(infResult.frameId, infResult.metaData);

    const ov::Tensor& logitsTensor = infResult.outputsData.find(outputNames[0])->second;
    float* logitsPtr = logitsTensor.data<float>();

    auto raw_scores = ov::Tensor();
    float* raw_scoresPtr = nullptr;
    if (add_raw_scores) {
        raw_scores = ov::Tensor(logitsTensor.get_element_type(), logitsTensor.get_shape());
        logitsTensor.copy_to(raw_scores);
        raw_scoresPtr = raw_scores.data<float>();
        result->raw_scores = raw_scores;
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

    auto resolved_labels = resolver.resolve_labels(predicted_labels, predicted_scores);

    auto retVal = std::unique_ptr<ResultBase>(result);
    result->topLabels.reserve(resolved_labels.first.size());
    for (size_t i = 0; i < resolved_labels.first.size(); ++i) {
        result->topLabels.emplace_back(hierarchical_info.label_to_idx[resolved_labels.first[i]], resolved_labels.first[i], resolved_labels.second[i]);
    }

    return retVal;
}

std::unique_ptr<ResultBase> ClassificationModel::get_multiclass_predictions(InferenceResult& infResult, bool add_raw_scores) {
    const ov::Tensor& indicesTensor = infResult.outputsData.find(indices_name)->second;
    const int* indicesPtr = indicesTensor.data<int>();
    const ov::Tensor& scoresTensor = infResult.outputsData.find(scores_name)->second;
    const float* scoresPtr = scoresTensor.data<float>();

    ClassificationResult* result = new ClassificationResult(infResult.frameId, infResult.metaData);
    auto retVal = std::unique_ptr<ResultBase>(result);

    if (add_raw_scores) {
        const ov::Tensor& logitsTensor = infResult.outputsData.find(raw_scores_name)->second;
        result->raw_scores = ov::Tensor(logitsTensor.get_element_type(), logitsTensor.get_shape());
        logitsTensor.copy_to(result->raw_scores);
        result->raw_scores.set_shape(ov::Shape({result->raw_scores.get_size()}));
    }

    result->topLabels.reserve(scoresTensor.get_size());
    for (size_t i = 0; i < scoresTensor.get_size(); ++i) {
        int ind = indicesPtr[i];
        if (ind < 0 || ind >= static_cast<int>(labels.size())) {
            throw std::runtime_error("Invalid index for the class label is found during postprocessing");
        }
        result->topLabels.emplace_back(ind, labels[ind], scoresPtr[i]);
    }

    return retVal;
}

void ClassificationModel::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input  ------------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error("Classification model wrapper supports topologies with only 1 input");
    }
    const auto& input = model->input();
    inputNames.push_back(input.get_any_name());

    const ov::Shape& inputShape = input.get_partial_shape().get_max_shape();
    const ov::Layout& inputLayout = getInputLayout(input);

    if (!embedded_processing) {
        model = ImageModel::embedProcessing(model,
                                        inputNames[0],
                                        inputLayout,
                                        resizeMode,
                                        interpolationMode,
                                        ov::Shape{inputShape[ov::layout::width_idx(inputLayout)],
                                                  inputShape[ov::layout::height_idx(inputLayout)]},
                                        pad_value,
                                        reverse_input_channels,
                                        {},
                                        scale_values);

        ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
        model = ppp.build();
        useAutoResize = true; // temporal solution for classification
    }

    // --------------------------- Prepare output  -----------------------------------------------------
    if (model->outputs().size() > 5) {
        throw std::logic_error("Classification model wrapper supports topologies with up to 4 outputs");
    }

    if (model->outputs().size() == 1) {
        const ov::Shape& outputShape = model->output().get_partial_shape().get_max_shape();
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
        if (classesNum == labels.size() + 1) {
            labels.insert(labels.begin(), "other");
            slog::warn << "Inserted 'other' label as first." << slog::endl;
        } else if (classesNum != labels.size()) {
            throw std::logic_error("Model's number of classes and parsed labels must match (" +
                                std::to_string(outputShape[1]) + " and " + std::to_string(labels.size()) + ')');
        }
    }

    if (multilabel || hierarchical) {
        embedded_processing = true;
        outputNames = get_non_xai_names(model->outputs());
        append_xai_names(model->outputs(), outputNames);
        return;
    }

    if (!embedded_processing) {
        addOrFindSoftmaxAndTopkOutputs(model, topk, output_raw_scores);
    }
    embedded_processing = true;

    outputNames = {indices_name, scores_name};
    if (output_raw_scores) {
        outputNames.emplace_back("raw_scores");
    }
    append_xai_names(model->outputs(), outputNames);
}

std::unique_ptr<ClassificationResult> ClassificationModel::infer(const ImageInputData& inputData) {
    auto result = ModelBase::infer(static_cast<const InputData&>(inputData));
    return std::unique_ptr<ClassificationResult>(static_cast<ClassificationResult*>(result.release()));
}

HierarchicalConfig::HierarchicalConfig(const std::string& json_repr)  {
    nlohmann::json data = nlohmann::json::parse(json_repr);

    num_multilabel_heads = data.at("cls_heads_info").at("num_multilabel_classes");
    num_multiclass_heads = data.at("cls_heads_info").at("num_multiclass_heads");
    num_single_label_classes = data.at("cls_heads_info").at("num_single_label_classes");

    data.at("cls_heads_info").at("label_to_idx").get_to(label_to_idx);
    data.at("cls_heads_info").at("all_groups").get_to(all_groups);
    data.at("label_tree_edges").get_to(label_tree_edges);

    std::map<std::string, std::pair<int,int>> tmp_head_idx_to_logits_range;
    data.at("cls_heads_info").at("head_idx_to_logits_range").get_to(tmp_head_idx_to_logits_range);

    for (const auto& range_descr : tmp_head_idx_to_logits_range) {
        head_idx_to_logits_range[stoi(range_descr.first)] = range_descr.second;
    }
}

GreedyLabelsResolver::GreedyLabelsResolver(const HierarchicalConfig& config) :
    label_to_idx(config.label_to_idx),
    label_relations(config.label_tree_edges),
    label_groups(config.all_groups) {}

std::pair<std::vector<std::string>, std::vector<float>> GreedyLabelsResolver::resolve_labels(const std::vector<std::reference_wrapper<std::string>>& labels,
                                                                                             const std::vector<float>& scores) {
    std::map<std::string, float> label_to_prob;
    for (const auto& label_idx : label_to_idx) {
        label_to_prob[label_idx.first] = 0.f;
    }

    for (size_t i = 0; i < labels.size(); ++i) {
        label_to_prob[labels[i]] = scores[i];
    }

    std::vector<std::string> candidates;
    for (const auto& g : label_groups) {
        if (g.size() == 1) {
            candidates.push_back(g[0]);
        }
        else {
            float max_prob = 0.f;
            std::string max_label;
            for (const auto& lbl : g) {
                if (label_to_prob[lbl] > max_prob) {
                    max_prob = label_to_prob[lbl];
                    max_label = lbl;
                }
                if (max_label.size() > 0) {
                    candidates.push_back(max_label);
                }
            }
        }
    }
    std::vector<std::string> output_labels;
    std::vector<float> output_scores;

    for (const auto& lbl : candidates) {
        if (std::find(output_labels.begin(), output_labels.end(), lbl) != output_labels.end()) {
            continue;
        }
        auto labels_to_add = get_predecessors(lbl, candidates);
        for (const auto& new_lbl : labels_to_add) {
            if (std::find(output_labels.begin(), output_labels.end(), new_lbl) == output_labels.end()) {
                output_labels.push_back(new_lbl);
                output_scores.push_back(label_to_prob[new_lbl]);
            }
        }
    }

    return {output_labels, output_scores};
}

std::string GreedyLabelsResolver::get_parent(const std::string& label) {
    for (const auto& edge : label_relations) {
        if (label == edge.first) {
            return edge.second;
        }
    }
    return "";
}

std::vector<std::string> GreedyLabelsResolver::get_predecessors(const std::string& label, const std::vector<std::string>& candidates) {
    std::vector<std::string> predecessors;
    auto last_parent = get_parent(label);

    if (last_parent.size() == 0) {
        return {label};
    }
    while (last_parent.size() > 0) {
        if (std::find(candidates.begin(), candidates.end(), last_parent) == candidates.end()) {
            return {};
        }
        predecessors.push_back(last_parent);
        last_parent = get_parent(last_parent);

    }

    if (predecessors.size() > 0) {
        predecessors.push_back(label);
    }

    return predecessors;
}