/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tasks/detection/ssd.h"

#include "utils/config.h"
#include "utils/math.h"
#include "utils/tensor.h"

constexpr char saliency_map_name[]{"saliency_map"};
constexpr char feature_vector_name[]{"feature_vector"};

NumAndStep NumAndStep::fromSingleOutput(const ov::Shape& shape) {
    const ov::Layout& layout("NCHW");
    if (shape.size() != 4) {
        throw std::logic_error("SSD single output must have 4 dimensions, but had " + std::to_string(shape.size()));
    }
    size_t detectionsNum = shape[ov::layout::height_idx(layout)];
    size_t objectSize = shape[ov::layout::width_idx(layout)];
    if (objectSize != 7) {
        throw std::logic_error("SSD single output must have 7 as a last dimension, but had " +
                               std::to_string(objectSize));
    }
    return {detectionsNum, objectSize};
}

NumAndStep NumAndStep::fromMultipleOutputs(const ov::Shape& boxesShape) {
    if (boxesShape.size() == 2) {
        ov::Layout boxesLayout = "NC";
        size_t detectionsNum = boxesShape[ov::layout::batch_idx(boxesLayout)];
        size_t objectSize = boxesShape[ov::layout::channels_idx(boxesLayout)];

        if (objectSize != 5) {
            throw std::logic_error("Incorrect 'boxes' output shape, [n][5] shape is required");
        }
        return {detectionsNum, objectSize};
    }
    if (boxesShape.size() == 3) {
        ov::Layout boxesLayout = "CHW";
        size_t detectionsNum = boxesShape[ov::layout::height_idx(boxesLayout)];
        size_t objectSize = boxesShape[ov::layout::width_idx(boxesLayout)];

        if (objectSize != 4 && objectSize != 5) {
            throw std::logic_error("Incorrect 'boxes' output shape, [b][n][{4 or 5}] shape is required");
        }
        return {detectionsNum, objectSize};
    }
    throw std::logic_error("Incorrect number of 'boxes' output dimensions, expected 2 or 3, but had " +
                           std::to_string(boxesShape.size()));
}

std::map<std::string, ov::Tensor> SSD::preprocess(cv::Mat image) {
    std::map<std::string, ov::Tensor> input = {};

    if (adapter->getInputNames().size() > 1) {
        ov::Tensor info{ov::element::i32, ov::Shape({1, 3})};
        int32_t* data = info.data<int32_t>();
        data[0] = input_shape.height;
        data[1] = input_shape.width;
        data[3] = 1;
        input.emplace(adapter->getInputNames()[1], std::move(info));
    }
    input.emplace(adapter->getInputNames()[0], utils::wrapMat2Tensor(image));

    return input;
}

cv::Size SSD::serialize(std::shared_ptr<ov::Model> ov_model) {
    auto output_mode = ov_model->outputs().size() > 1 ? SSDOutputMode::multi : SSDOutputMode::single;

    auto input_tensor = ov_model->inputs()[0];

    auto layout = ov::layout::get_layout(input_tensor);
    if (layout.empty()) {
        layout = utils::getLayoutFromShape(input_tensor.get_partial_shape());
    }

    auto interpolation_mode = cv::INTER_LINEAR;
    utils::RESIZE_MODE resize_mode = utils::RESIZE_FILL;

    auto shape = input_tensor.get_partial_shape().get_max_shape();

    auto input_shape = ov::Shape{shape[ov::layout::width_idx(layout)], shape[ov::layout::height_idx(layout)]};
    uint8_t pad_value = 0;

    auto config = ov_model->has_rt_info("model_info") ? ov_model->get_rt_info<ov::AnyMap>("model_info") : ov::AnyMap{};

    std::vector<float> scale_values;
    std::vector<float> mean_values;
    scale_values = utils::get_from_any_maps("scale_values", config, ov::AnyMap{}, scale_values);
    mean_values = utils::get_from_any_maps("mean_values", config, ov::AnyMap{}, mean_values);

    bool reverse_input_channels = false;

    ov_model = utils::embedProcessing(ov_model,
                                      input_tensor.get_any_name(),
                                      layout,
                                      resize_mode,
                                      interpolation_mode,
                                      input_shape,
                                      pad_value,
                                      reverse_input_channels,
                                      mean_values,
                                      scale_values);

    if (output_mode == SSDOutputMode::single) {
        prepareSingleOutput(ov_model);
    } else {
        // prepareMultipleOutputs(ov_model); //This does nothing from what I can see.
    }

    return cv::Size(input_shape[0], input_shape[1]);
}

void SSD::prepareSingleOutput(std::shared_ptr<ov::Model> ov_model) {
    const auto& output = ov_model->output();

    ov::preprocess::PrePostProcessor ppp(ov_model);
    ppp.output().tensor().set_element_type(ov::element::f32);
    ov_model = ppp.build();
}
void SSD::prepareMultipleOutputs(std::shared_ptr<ov::Model> ov_model) {
    const ov::OutputVector& outputs = ov_model->outputs();
    std::vector<std::string> output_names;
    for (auto& output : outputs) {
        const auto& tensorNames = output.get_names();
        for (const auto& name : tensorNames) {
            if (name.find("boxes") != std::string::npos) {
                output_names.push_back(name);
                break;
            } else if (name.find("labels") != std::string::npos) {
                output_names.push_back(name);
                break;
            } else if (name.find("scores") != std::string::npos) {
                output_names.push_back(name);
                break;
            }
        }
    }
    if (output_names.size() != 2 && output_names.size() != 3) {
        throw std::logic_error("SSD model wrapper must have 2 or 3 outputs, but had " +
                               std::to_string(output_names.size()));
    }
    std::sort(output_names.begin(), output_names.end());

    for (auto& name : output_names) {
        std::cout << "output name: " << name << std::endl;
    }

    // ov::preprocess::PrePostProcessor ppp(ov_model);

    // for (const auto& output_name : output_names) {
    //     if (output_name != "labels") { //TODO: Discover why this isnt needed in original?
    //         ppp.output(output_name).tensor().set_element_type(ov::element::f32);
    //     }
    // }
    // ov_model = ppp.build();
}

std::vector<std::string> SSD::filterOutXai(const std::vector<std::string>& names) {
    std::vector<std::string> filtered;
    std::copy_if(names.begin(), names.end(), std::back_inserter(filtered), [](const std::string& name) {
        return name != saliency_map_name && name != feature_vector_name;
    });
    return filtered;
}

DetectionResult SSD::postprocess(InferenceResult& infResult) {
    auto result = adapter->getOutputNames().size() > 1 ? postprocessMultipleOutputs(infResult)
                                                       : postprocessSingleOutput(infResult);

    {
        auto iter = infResult.data.find(feature_vector_name);
        if (iter != infResult.data.end()) {
            result.feature_vector = iter->second;
        }
    }

    {
        auto iter = infResult.data.find(saliency_map_name);
        if (iter != infResult.data.end()) {
            result.saliency_map = iter->second;
        }
    }

    return result;
}

DetectionResult SSD::postprocessSingleOutput(InferenceResult& infResult) {
    DetectionResult result;

    // WIP

    return result;
}
DetectionResult SSD::postprocessMultipleOutputs(InferenceResult& infResult) {
    const std::vector<std::string> namesWithoutXai = filterOutXai(adapter->getOutputNames());
    const float* boxes = infResult.data[namesWithoutXai[0]].data<float>();
    NumAndStep numAndStep = NumAndStep::fromMultipleOutputs(infResult.data[namesWithoutXai[0]].get_shape());
    const int64_t* labels = infResult.data[namesWithoutXai[1]].data<int64_t>();
    const float* scores = namesWithoutXai.size() > 2 ? infResult.data[namesWithoutXai[2]].data<float>() : nullptr;

    float floatInputImgWidth = float(infResult.inputImageSize.width),
          floatInputImgHeight = float(infResult.inputImageSize.height);
    float invertedScaleX = floatInputImgWidth / input_shape.width,
          invertedScaleY = floatInputImgHeight / input_shape.height;
    int padLeft = 0, padTop = 0;
    if (utils::RESIZE_KEEP_ASPECT == resize_mode || utils::RESIZE_KEEP_ASPECT_LETTERBOX == resize_mode) {
        invertedScaleX = invertedScaleY = std::max(invertedScaleX, invertedScaleY);
        if (utils::RESIZE_KEEP_ASPECT_LETTERBOX == resize_mode) {
            padLeft = (input_shape.width - int(std::round(floatInputImgWidth / invertedScaleX))) / 2;
            padTop = (input_shape.height - int(std::round(floatInputImgHeight / invertedScaleY))) / 2;
        }
    }

    // In models with scores stored in separate output coordinates are normalized to [0,1]
    // In other multiple-outputs models coordinates are normalized to [0,netInputWidth] and [0,netInputHeight]
    float widthScale = scores ? input_shape.width : 1.0f;
    float heightScale = scores ? input_shape.height : 1.0f;

    DetectionResult result;
    for (size_t i = 0; i < numAndStep.detectionsNum; i++) {
        float confidence = scores ? scores[i] : boxes[i * numAndStep.objectSize + 4];

        /** Filtering out objects with confidence < confidence_threshold probability **/
        if (confidence > confidence_threshold) {
            auto x = clamp_and_round((boxes[i * numAndStep.objectSize] * widthScale - padLeft) * invertedScaleX,
                                     0.f,
                                     floatInputImgWidth);
            auto y = clamp_and_round((boxes[i * numAndStep.objectSize + 1] * heightScale - padTop) * invertedScaleY,
                                     0.f,
                                     floatInputImgHeight);
            auto width = clamp_and_round((boxes[i * numAndStep.objectSize + 2] * widthScale - padLeft) * invertedScaleX,
                                         0.f,
                                         floatInputImgWidth) -
                         x;
            auto height =
                clamp_and_round((boxes[i * numAndStep.objectSize + 3] * heightScale - padTop) * invertedScaleY,
                                0.f,
                                floatInputImgHeight) -
                y;

            if (width * height >= box_area_threshold) {
                DetectedObject object;
                object.x = x;
                object.y = y;
                object.width = width;
                object.height = height;
                object.confidence = confidence;
                object.labelID = labels[i];
                object.label = this->labels[object.labelID];
                result.objects.push_back(object);
            }
        }
    }
    return result;
}
