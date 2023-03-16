/*
// Copyright (C) 2021-2023 Intel Corporation
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

#include "models/image_model.h"

#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

#include <utils/image_utils.h>
#include <utils/ocv_common.hpp>
#include <adapters/inference_adapter.h>

#include "models/input_data.h"
#include "models/internal_model_data.h"

ImageModel::ImageModel(const std::string& modelFile,
               bool useAutoResize,
               const std::vector<std::string>& labels,
               const std::string& layout = "")
    : ModelBase(modelFile, layout),
      labels(labels),
      useAutoResize(useAutoResize) {}

ImageModel::ImageModel(std::shared_ptr<ov::Model>& model, const ov::AnyMap& configuration)
    : ModelBase(model, configuration) {
    auto layout_iter = configuration.find("layout");
    std::string layout;
    if (layout_iter != configuration.end()) {
        layout = layout_iter->second.as<std::string>();
        inputsLayouts = parseLayoutString(layout);
    }

    auto auto_resize_iter = configuration.find("auto_resize");
    if (auto_resize_iter != configuration.end()) {
        useAutoResize = auto_resize_iter->second.as<bool>();
    }

    auto labels_iter = configuration.find("labels");
    if (labels_iter == configuration.end()) {
        if (!model->has_rt_info<std::string>("model_info", "labels")) {
            throw std::runtime_error("Configuration or model rt_info should contain labels"); //TODO
        }
        labels = split(model->get_rt_info<std::string>("model_info", "labels"), ' ');
    } else {
        labels = labels_iter->second.as<std::vector<std::string>>();
    }
}

std::shared_ptr<InternalModelData> ImageModel::preprocess(const InputData& inputData, InferenceInput& input) {
    const auto& origImg = inputData.asRef<ImageInputData>().inputImage;
    auto img = inputTransform(origImg);

    if (!useAutoResize) {
        // /* Resize and copy data from the image to the input tensor */
        auto tensorShape = inferenceAdapter->getInputShape(inputNames[0]); // first input should be image
        const ov::Layout layout("NHWC");
        const size_t width = tensorShape[ov::layout::width_idx(layout)];
        const size_t height = tensorShape[ov::layout::height_idx(layout)];
        const size_t channels = tensorShape[ov::layout::channels_idx(layout)];
        if (static_cast<size_t>(img.channels()) != channels) {
            throw std::runtime_error("The number of channels for model input and image must match");
        }
        if (channels != 1 && channels != 3) {
            throw std::runtime_error("Unsupported number of channels");
        }
        img = resizeImageExt(img, width, height, resizeMode, interpolationMode);
    }
    input.emplace(inputNames[0], wrapMat2Tensor(img));
    return std::make_shared<InternalImageModelData>(origImg.cols, origImg.rows);
}
