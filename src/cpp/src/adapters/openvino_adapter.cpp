/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "adapters/openvino_adapter.h"

#include <openvino/openvino.hpp>
#include <stdexcept>
#include <vector>

#include "utils/config.h"

void OpenVINOInferenceAdapter::compileModel(const std::string& device, const ov::AnyMap& adapterConfig) {
    if (!model) {
        throw std::runtime_error("Model is not loaded");
    }
    size_t max_num_requests = 1;
    max_num_requests = utils::get_from_any_maps("max_num_requests", adapterConfig, {}, max_num_requests);

    ov::AnyMap customCompilationConfig(adapterConfig);
    if (max_num_requests != 1) {
        if (customCompilationConfig.find("PERFORMANCE_HINT") == customCompilationConfig.end()) {
            customCompilationConfig["PERFORMANCE_HINT"] = ov::hint::PerformanceMode::THROUGHPUT;
        }
        if (max_num_requests > 0) {
            if (customCompilationConfig.find("PERFORMANCE_HINT_NUM_REQUESTS") == customCompilationConfig.end()) {
                customCompilationConfig["PERFORMANCE_HINT_NUM_REQUESTS"] = ov::hint::num_requests(max_num_requests);
            }
        }
    } else {
        if (customCompilationConfig.find("PERFORMANCE_HINT") == customCompilationConfig.end()) {
            customCompilationConfig["PERFORMANCE_HINT"] = ov::hint::PerformanceMode::LATENCY;
        }
    }

    ov::Core core;
    compiledModel = core.compile_model(model, device, customCompilationConfig);
    asyncQueue = std::make_unique<AsyncInferQueue>(compiledModel, max_num_requests);
    initInputsOutputs();
}

void OpenVINOInferenceAdapter::loadModel(const std::string& modelPath,
                                         const std::string& device,
                                         const ov::AnyMap& adapterConfig,
                                         bool preCompile) {
    ov::Core core;
    model = core.read_model(modelPath);
    if (model->has_rt_info({"model_info"})) {
        modelConfig = model->get_rt_info<ov::AnyMap>("model_info");
    } else if (modelPath.find("onnx") != std::string::npos || modelPath.find("ONNX") != std::string::npos) {
        modelConfig = utils::get_config_from_onnx(modelPath);
        utils::add_ov_model_info(model, modelConfig);
    }
    if (preCompile) {
        compileModel(device, adapterConfig);
    }
}

void OpenVINOInferenceAdapter::applyModelTransform(std::function<void(std::shared_ptr<ov::Model>&)> t) {
    if (!model) {
        throw std::runtime_error("Model is not loaded");
    }
    t(model);
}

void OpenVINOInferenceAdapter::infer(const InferenceInput& input, InferenceOutput& output) {
    auto request = asyncQueue->operator[](asyncQueue->get_idle_request_id());
    for (const auto& [name, tensor] : input) {
        request.set_tensor(name, tensor);
    }
    for (const auto& [name, tensor] : output) {
        request.set_tensor(name, tensor);
    }
    request.infer();
    for (const auto& name : outputNames) {
        output[name] = request.get_tensor(name);
    }
}

InferenceOutput OpenVINOInferenceAdapter::infer(const InferenceInput& input) {
    auto request = asyncQueue->operator[](asyncQueue->get_idle_request_id());
    // Fill input blobs
    for (const auto& item : input) {
        request.set_tensor(item.first, item.second);
    }

    // Do inference
    request.infer();

    // Processing output blobs
    InferenceOutput output;
    for (const auto& item : outputNames) {
        output.emplace(item, request.get_tensor(item));
    }

    return output;
}

void OpenVINOInferenceAdapter::inferAsync(const InferenceInput& input, CallbackData callback_args) {
    asyncQueue->start_async(input, callback_args);
}

void OpenVINOInferenceAdapter::setCallback(std::function<void(ov::InferRequest, CallbackData)> callback) {
    asyncQueue->set_custom_callbacks(callback);
}

bool OpenVINOInferenceAdapter::isReady() {
    return asyncQueue->is_ready();
}

void OpenVINOInferenceAdapter::awaitAll() {
    asyncQueue->wait_all();
}

void OpenVINOInferenceAdapter::awaitAny() {
    asyncQueue->get_idle_request_id();
}

size_t OpenVINOInferenceAdapter::getNumAsyncExecutors() const {
    return asyncQueue->size();
}

ov::PartialShape OpenVINOInferenceAdapter::getInputShape(const std::string& inputName) const {
    return compiledModel.input(inputName).get_partial_shape();
}
ov::PartialShape OpenVINOInferenceAdapter::getOutputShape(const std::string& outputName) const {
    return compiledModel.output(outputName).get_partial_shape();
}

void OpenVINOInferenceAdapter::initInputsOutputs() {
    for (const auto& input : compiledModel.inputs()) {
        inputNames.push_back(input.get_any_name());
    }

    for (const auto& output : compiledModel.outputs()) {
        outputNames.push_back(output.get_any_name());
    }
}
ov::element::Type_t OpenVINOInferenceAdapter::getInputDatatype(const std::string& name) const {
    return compiledModel.input(name).get_element_type();
}
ov::element::Type_t OpenVINOInferenceAdapter::getOutputDatatype(const std::string& name) const {
    return compiledModel.output(name).get_element_type();
}

std::vector<std::string> OpenVINOInferenceAdapter::getInputNames() const {
    return inputNames;
}

std::vector<std::string> OpenVINOInferenceAdapter::getOutputNames() const {
    return outputNames;
}

const ov::AnyMap& OpenVINOInferenceAdapter::getModelConfig() const {
    return modelConfig;
}
