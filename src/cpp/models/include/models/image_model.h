/*
 * Copyright (C) 2020-2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <stddef.h>

#include <memory>
#include <string>

#include "adapters/inference_adapter.h"
#include "models/input_data.h"
#include "models/results.h"
#include "utils/args_helper.hpp"
#include "utils/image_utils.h"
#include "utils/ocv_common.hpp"

namespace ov {
class InferRequest;
}  // namespace ov
struct InputData;
struct InternalModelData;

// ImageModel implements preprocess(), ImageModel's direct or indirect children are expected to implement prostprocess()
class ImageModel {
public:
    /// Constructor
    /// @param modelFile name of model to load
    /// @param useAutoResize - if true, image is resized by openvino
    /// @param layout - model input layout
    ImageModel(const std::string& modelFile,
               const std::string& resize_type,
               bool useAutoResize,
               const std::string& layout = "");

    ImageModel(std::shared_ptr<ov::Model>& model, const ov::AnyMap& configuration);
    ImageModel(std::shared_ptr<InferenceAdapter>& adapter, const ov::AnyMap& configuration = {});

    virtual std::shared_ptr<InternalModelData> preprocess(const InputData& inputData, InferenceInput& input);
    virtual std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) = 0;

    void load(ov::Core& core, const std::string& device, size_t num_infer_requests = 1);

    std::shared_ptr<ov::Model> prepare();

    virtual size_t getNumAsyncExecutors() const;
    virtual bool isReady();
    virtual void awaitAll();
    virtual void awaitAny();
    virtual void setCallback(
        std::function<void(std::unique_ptr<ResultBase>, const ov::AnyMap& callback_args)> callback);

    std::shared_ptr<ov::Model> getModel();
    std::shared_ptr<InferenceAdapter> getInferenceAdapter();

    static std::vector<std::string> loadLabels(const std::string& labelFilename);
    std::shared_ptr<ov::Model> embedProcessing(std::shared_ptr<ov::Model>& model,
                                               const std::string& inputName,
                                               const ov::Layout&,
                                               RESIZE_MODE resize_mode,
                                               const cv::InterpolationFlags interpolationMode,
                                               const ov::Shape& targetShape,
                                               uint8_t pad_value,
                                               bool brg2rgb,
                                               const std::vector<float>& mean,
                                               const std::vector<float>& scale,
                                               const std::type_info& dtype = typeid(int));
    virtual void inferAsync(const ImageInputData& inputData, const ov::AnyMap& callback_args = {});
    std::unique_ptr<ResultBase> inferImage(const ImageInputData& inputData);
    std::vector<std::unique_ptr<ResultBase>> inferBatchImage(const std::vector<ImageInputData>& inputData);

protected:
    RESIZE_MODE selectResizeMode(const std::string& resize_type);
    virtual void updateModelInfo();
    void init_from_config(const ov::AnyMap& top_priority, const ov::AnyMap& mid_priority);

    std::string getLabelName(size_t labelID) {
        return labelID < labels.size() ? labels[labelID] : std::string("Label #") + std::to_string(labelID);
    }

    std::vector<std::string> labels = {};
    bool useAutoResize = false;
    bool embedded_processing = false;  // flag in model_info that pre/postprocessing embedded

    size_t netInputHeight = 0;
    size_t netInputWidth = 0;
    cv::InterpolationFlags interpolationMode = cv::INTER_LINEAR;
    RESIZE_MODE resizeMode = RESIZE_FILL;
    uint8_t pad_value = 0;
    bool reverse_input_channels = false;
    std::vector<float> scale_values;
    std::vector<float> mean_values;

protected:
    virtual void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) = 0;

    InputTransform inputTransform = InputTransform();

    std::shared_ptr<ov::Model> model;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::string modelFile;
    std::shared_ptr<InferenceAdapter> inferenceAdapter;
    std::map<std::string, ov::Layout> inputsLayouts;
    ov::Layout getInputLayout(const ov::Output<ov::Node>& input);
    std::function<void(std::unique_ptr<ResultBase>, const ov::AnyMap&)> lastCallback;
};
