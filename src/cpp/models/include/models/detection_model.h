/*
 * Copyright (C) 2020-2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "models/base_model.h"

struct DetectionResult;
struct ImageInputData;
struct InferenceAdatper;

class DetectionModel : public BaseModel {
public:
    DetectionModel(std::shared_ptr<ov::Model>& model, const ov::AnyMap& configuration);
    DetectionModel(std::shared_ptr<InferenceAdapter>& adapter, const ov::AnyMap& configuration = {});

    static std::unique_ptr<DetectionModel> create_model(const std::string& modelFile,
                                                        const ov::AnyMap& configuration = {},
                                                        std::string model_type = "",
                                                        bool preload = true,
                                                        const std::string& device = "AUTO");
    static std::unique_ptr<DetectionModel> create_model(std::shared_ptr<InferenceAdapter>& adapter);

    virtual std::unique_ptr<Scene> infer(const ImageInputData& inputData);
    virtual std::vector<std::unique_ptr<Scene>> inferBatch(const std::vector<ImageInputData>& inputImgs);

protected:
    float confidence_threshold = 0.5f;

    void updateModelInfo() override;
};
