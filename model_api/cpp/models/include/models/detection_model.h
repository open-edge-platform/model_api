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

#pragma once
#include <stddef.h>

#include <string>
#include <vector>

#include "models/image_model.h"

struct DetectionResult;
struct ImageInputData;
struct InferenceAdatper;

class DetectionModel : public ImageModel {
public:
    /// Constructor
    /// @param modelFile name of model to load
    /// @param confidenceThreshold - threshold to eliminate low-confidence detections.
    /// Any detected object with confidence lower than this threshold will be ignored.
    /// @param useAutoResize - if true, image will be resized by openvino.
    /// Otherwise, image will be preprocessed and resized using OpenCV routines.
    /// @param labels - array of labels for every class. If this array is empty or contains less elements
    /// than actual classes number, default "Label #N" will be shown for missing items.
    /// @param layout - model input layout
    DetectionModel(const std::string& modelFile,
                   float confidenceThreshold,
                   const std::string& resize_type,
                   bool useAutoResize,
                   const std::vector<std::string>& labels,
                   const std::string& layout = "");

    static std::unique_ptr<DetectionModel> create_model(const std::string& modelFile, std::shared_ptr<InferenceAdapter> adapter = nullptr, std::string model_type = "", const ov::AnyMap& configuration = {});

    static std::vector<std::string> loadLabels(const std::string& labelFilename);

    virtual std::unique_ptr<DetectionResult> infer(const ImageInputData& inputData);

    std::vector<std::string> labels;
protected:
    float confidenceThreshold;

    std::string getLabelName(int labelID) {
        return (size_t)labelID < labels.size() ? labels[labelID] : std::string("Label #") + std::to_string(labelID);
    }
};
