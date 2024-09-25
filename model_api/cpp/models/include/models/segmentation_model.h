/*
// Copyright (C) 2020-2024 Intel Corporation
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
#include <memory>
#include <string>
#include <vector>

#include "models/image_model.h"

namespace ov {
class Model;
}  // namespace ov
struct InferenceResult;
struct ResultBase;
struct ImageResult;
struct ImageResultWithSoftPrediction;
struct ImageInputData;
struct Contour;

class SegmentationModel : public ImageModel {
public:
    SegmentationModel(std::shared_ptr<ov::Model>& model, const ov::AnyMap& configuration);
    SegmentationModel(std::shared_ptr<InferenceAdapter>& adapter, const ov::AnyMap& configuration = {});

    static std::unique_ptr<SegmentationModel> create_model(const std::string& modelFile, const ov::AnyMap& configuration = {}, bool preload = true, const std::string& device = "AUTO");
    static std::unique_ptr<SegmentationModel> create_model(std::shared_ptr<InferenceAdapter>& adapter);

    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

    virtual std::unique_ptr<ImageResult> infer(const ImageInputData& inputData);
    virtual std::vector<std::unique_ptr<ImageResult>> inferBatch(const std::vector<ImageInputData>& inputImgs);

    static std::string ModelType;
    std::vector<Contour> getContours(const ImageResultWithSoftPrediction& imageResult);

protected:

    void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) override;
    void updateModelInfo() override;
    void init_from_config(const ov::AnyMap& top_priority, const ov::AnyMap& mid_priority);

    int blur_strength = -1;
    float soft_threshold = -std::numeric_limits<float>::infinity();
    bool return_soft_prediction = true;
};

cv::Mat create_hard_prediction_from_soft_prediction(const cv::Mat& soft_prediction, float soft_threshold, int blur_strength);
