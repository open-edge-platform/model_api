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

#include "models/segmentation_model.h"

#include <stddef.h>
#include <stdint.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include "models/internal_model_data.h"
#include "models/input_data.h"
#include "models/results.h"

std::unique_ptr<SegmentationModel> SegmentationModel::create_model(const std::string& modelFile, const ov::AnyMap& configuration, bool preload) {
    auto core = ov::Core();
    std::shared_ptr<ov::Model> model = core.read_model(modelFile);

    // Check model_type in the rt_info, ignore configuration
    std::string model_type = "Segmentation";
    try {
        if (model->has_rt_info("model_info", "model_type") ) {
            model_type = model->get_rt_info<std::string>("model_info", "model_type");
        }
    } catch (const std::exception& e) {
        slog::warn << "Model type is not specified in the rt_info, use default model type: " << model_type << slog::endl;
    }

    if (model_type != "Segmentation") {
        throw ov::Exception("Incorrect or unsupported model_type is provided in the model_info section: " + model_type);
    }

    std::unique_ptr<SegmentationModel> segmentor{new SegmentationModel(model, configuration)};
    segmentor->prepare();
    if (preload) {
        segmentor->load(core);
    }
    return segmentor;
}

std::unique_ptr<SegmentationModel> SegmentationModel::create_model(std::shared_ptr<InferenceAdapter>& adapter) {
    auto configuration = adapter->getModelConfig();
    auto model_type_iter = configuration.find("model_type");
    std::string model_type = "Segmentation";
    if (model_type_iter != configuration.end()) {
        model_type = model_type_iter->second.as<std::string>();
    }

    if (model_type != "Segmentation") {
        throw ov::Exception("Incorrect or unsupported model_type is provided: " + model_type);
    }

    std::unique_ptr<SegmentationModel> segmentor{new SegmentationModel(adapter)};
    return segmentor;
}

void SegmentationModel::updateModelInfo() {
    ImageModel::updateModelInfo();

    model->set_rt_info("segmentation", "model_info", "model_type");
}

void SegmentationModel::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output ---------------------------------------------
    // --------------------------- Prepare input  -----------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error("Segmentation model wrapper supports topologies with only 1 input");
    }
    const auto& input = model->input();
    inputNames.push_back(input.get_any_name());

    const ov::Layout& inputLayout = getInputLayout(input);
    const ov::Shape& inputShape = input.get_shape();
    if (inputShape.size() != 4 || inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
        throw std::logic_error("3-channel 4-dimensional model's input is expected");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout({"NHWC"});

    if (useAutoResize) {
        ppp.input().tensor().set_spatial_dynamic_shape();

        ppp.input()
            .preprocess()
            .convert_element_type(ov::element::f32)
            .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    }

    ppp.input().model().set_layout(inputLayout);
    model = ppp.build();
    // --------------------------- Prepare output  -----------------------------------------------------
    if (model->outputs().size() != 1) {
        throw std::logic_error("Segmentation model wrapper supports topologies with only 1 output");
    }

    const auto& output = model->output();
    outputNames.push_back(output.get_any_name());

    const ov::Shape& outputShape = output.get_shape();
    ov::Layout outputLayout("");
    switch (outputShape.size()) {
        case 3:
            outputLayout = "CHW";
            outChannels = 1;
            outHeight = static_cast<int>(outputShape[ov::layout::height_idx(outputLayout)]);
            outWidth = static_cast<int>(outputShape[ov::layout::width_idx(outputLayout)]);
            break;
        case 4:
            outputLayout = "NCHW";
            outChannels = static_cast<int>(outputShape[ov::layout::channels_idx(outputLayout)]);
            outHeight = static_cast<int>(outputShape[ov::layout::height_idx(outputLayout)]);
            outWidth = static_cast<int>(outputShape[ov::layout::width_idx(outputLayout)]);
            break;
        default:
            throw std::logic_error("Unexpected output tensor shape. Only 4D and 3D outputs are supported.");
    }
}

std::unique_ptr<ResultBase> SegmentationModel::postprocess(InferenceResult& infResult) {
    ImageResult* result = new ImageResult(infResult.frameId, infResult.metaData);
    const auto& inputImgSize = infResult.internalModelData->asRef<InternalImageModelData>();
    const auto& outTensor = infResult.getFirstOutputTensor();

    result->resultImage = cv::Mat(outHeight, outWidth, CV_8UC1);

    if (outChannels == 1 && outTensor.get_element_type() == ov::element::i32) {
        cv::Mat predictions(outHeight, outWidth, CV_32SC1, outTensor.data<int32_t>());
        predictions.convertTo(result->resultImage, CV_8UC1);
    } else if (outChannels == 1 && outTensor.get_element_type() == ov::element::i64) {
        cv::Mat predictions(outHeight, outWidth, CV_32SC1);
        const auto data = outTensor.data<int64_t>();
        for (size_t i = 0; i < predictions.total(); ++i) {
            reinterpret_cast<int32_t*>(predictions.data)[i] = int32_t(data[i]);
        }
        predictions.convertTo(result->resultImage, CV_8UC1);
    } else if (outTensor.get_element_type() == ov::element::f32) {
        const float* data = outTensor.data<float>();
        for (int rowId = 0; rowId < outHeight; ++rowId) {
            for (int colId = 0; colId < outWidth; ++colId) {
                int classId = 0;
                float maxProb = -1.0f;
                for (int chId = 0; chId < outChannels; ++chId) {
                    float prob = data[chId * outHeight * outWidth + rowId * outWidth + colId];
                    if (prob > maxProb) {
                        classId = chId;
                        maxProb = prob;
                    }
                }  // nChannels

                result->resultImage.at<uint8_t>(rowId, colId) = classId;
            }  // width
        }  // height
    }

    cv::resize(result->resultImage,
               result->resultImage,
               cv::Size(inputImgSize.inputImgWidth, inputImgSize.inputImgHeight),
               0,
               0,
               cv::INTER_NEAREST);

    return std::unique_ptr<ResultBase>(result);
}

std::unique_ptr<ImageResult> SegmentationModel::infer(const ImageInputData& inputData) {
    auto result = ModelBase::infer(static_cast<const InputData&>(inputData));
    return std::unique_ptr<ImageResult>(static_cast<ImageResult*>(result.release()));
}
