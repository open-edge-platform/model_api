/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "adapters/inference_adapter.h"
#include "utils/async_infer_queue.h"

class OpenVINOInferenceAdapter : public InferenceAdapter {
public:
    OpenVINOInferenceAdapter() = default;

    virtual InferenceOutput infer(const InferenceInput& input) override;
    virtual void infer(const InferenceInput& input, InferenceOutput& output) override;
    virtual void inferAsync(const InferenceInput& input, const CallbackData callback_args) override;
    virtual void setCallback(std::function<void(ov::InferRequest, const CallbackData)> callback);
    virtual bool isReady();
    virtual void awaitAll();
    virtual void awaitAny();
    virtual void loadModel(const std::string& modelPath,
                           const std::string& device = "",
                           const ov::AnyMap& adapterConfig = {},
                           bool preCompile = true) override;
    virtual void compileModel(const std::string& device = "", const ov::AnyMap& adapterConfig = {}) override;
    virtual size_t getNumAsyncExecutors() const;
    virtual ov::PartialShape getInputShape(const std::string& inputName) const override;
    virtual ov::PartialShape getOutputShape(const std::string& outputName) const override;
    virtual ov::element::Type_t getInputDatatype(const std::string& inputName) const override;
    virtual ov::element::Type_t getOutputDatatype(const std::string& outputName) const override;
    virtual std::vector<std::string> getInputNames() const override;
    virtual std::vector<std::string> getOutputNames() const override;
    virtual const ov::AnyMap& getModelConfig() const override;

    void applyModelTransform(std::function<void(std::shared_ptr<ov::Model>&)> t);

protected:
    void initInputsOutputs();

protected:
    // Depends on the implementation details but we should share the model state in this class
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::unique_ptr<AsyncInferQueue> asyncQueue;
    ov::AnyMap modelConfig;  // the content of model_info section of rt_info
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiledModel;
};
