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

#include <memory>
#include <string>
#include <vector>

#include "models/image_model.h"

namespace ov {
class Model;
}  // namespace ov
struct InferenceResult;
struct ClassificationResult;
struct ResultBase;
struct ImageInputData;

class ClassificationModel : public ImageModel {
public:
    ClassificationModel(std::shared_ptr<ov::Model>& model, const ov::AnyMap& configuration);
    ClassificationModel(std::shared_ptr<InferenceAdapter>& adapter);

    static std::unique_ptr<ClassificationModel> create_model(const std::string& modelFile, const ov::AnyMap& configuration = {}, bool preload = true);
    static std::unique_ptr<ClassificationModel> create_model(std::shared_ptr<InferenceAdapter>& adapter);

    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

    virtual std::unique_ptr<ClassificationResult> infer(const ImageInputData& inputData);
    static std::string ModelType;

protected:
    size_t topk = 1;

    void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) override;
    void updateModelInfo() override;
};
