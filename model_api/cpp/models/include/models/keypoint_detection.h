/*
// Copyright (C) 2024 Intel Corporation
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
struct KeypointDetectionResult;
struct ImageInputData;

class KeypointDetectionModel : public ImageModel {
public:
    KeypointDetectionModel(std::shared_ptr<ov::Model>& model, const ov::AnyMap& configuration);
    KeypointDetectionModel(std::shared_ptr<InferenceAdapter>& adapter, const ov::AnyMap& configuration = {});

    static std::unique_ptr<KeypointDetectionModel> create_model(const std::string& modelFile, const ov::AnyMap& configuration = {}, bool preload = true, const std::string& device = "AUTO");
    static std::unique_ptr<KeypointDetectionModel> create_model(std::shared_ptr<InferenceAdapter>& adapter);

    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

    virtual std::unique_ptr<KeypointDetectionResult> infer(const ImageInputData& inputData);
    virtual std::vector<std::unique_ptr<KeypointDetectionResult>> inferBatch(const std::vector<ImageInputData>& inputImgs);

    static std::string ModelType;

protected:

    void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) override;
    void updateModelInfo() override;
    void init_from_config(const ov::AnyMap& top_priority, const ov::AnyMap& mid_priority);
};
