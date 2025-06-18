/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/config.h"

#include <onnxruntime_cxx_api.h>

namespace {
std::vector<std::string> split(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> output;
    size_t start = 0;
    size_t end = 0;
    while ((end = str.find(delimiter, start)) != std::string::npos) {
        output.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
    }
    output.push_back(str.substr(start));
    return output;
}
}  // namespace

ov::AnyMap utils::get_config_from_onnx(const std::string& model_path) {
    ov::AnyMap config;
    if (model_path.find("onnx") != std::string::npos || model_path.find("ONNX") != std::string::npos) {
        Ort::Env env;
        Ort::SessionOptions ort_session_options;
        Ort::Session session = Ort::Session(env, model_path.c_str(), ort_session_options);
        Ort::AllocatorWithDefaultOptions ort_alloc;

        Ort::ModelMetadata model_metadata = session.GetModelMetadata();
        std::vector<Ort::AllocatedStringPtr> keys = model_metadata.GetCustomMetadataMapKeysAllocated(ort_alloc);
        for (const auto& key : keys) {
            std::vector<std::string> attr_names;
            if (key != nullptr) {
                const std::array<const char*, 1> list_names = {key.get()};

                Ort::AllocatedStringPtr values_search =
                    model_metadata.LookupCustomMetadataMapAllocated(list_names[0], ort_alloc);
                if (values_search != nullptr) {
                    const std::array<const char*, 1> value = {values_search.get()};
                    attr_names = split(std::string(list_names[0]), " ");
                    // only flat metadata is supported
                    if (attr_names.size() == 2 && attr_names[0] == "model_info")
                        config[attr_names[1]] = std::string(value[0]);
                }
            }
        }
    } else {
        throw std::runtime_error("Model is not ONNX, can't get config from it");
    }
    return config;
}
