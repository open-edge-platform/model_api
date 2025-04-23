/*
 * Copyright (C) 2020-2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <models/detection_model.h>
#include <models/input_data.h>
#include <models/results.h>

#include <filesystem>
#include <fstream>

using namespace std;

namespace {
string data() {
    // Get data from env var, not form cmd arg to stay aligned with Python version
    static const char* const data = getenv("DATA");
    EXPECT_NE(data, nullptr);
    return data;
}

string model_path(const char model_name[]) {
    return data() + "/ultralytics/" + model_name;
}

shared_ptr<DetectionModel> cached_model(const char model_name[]) {
    static const char* prev_arg;
    static shared_ptr<DetectionModel> prev_model;
    if (model_name == prev_arg) {
        return prev_model;
    } else {
        prev_arg = model_name;
        filesystem::path xml;
        for (auto const& dir_entry : filesystem::directory_iterator{model_path(model_name)}) {
            const filesystem::path& path = dir_entry.path();
            if (".xml" == path.extension()) {
                EXPECT_TRUE(xml.empty());
                xml = path;
            }
        }
        bool preload = true;
        prev_model = DetectionModel::create_model(xml.string(), {}, "", preload, "CPU");
        return prev_model;
    }
}

struct Param {
    const char* model_name;
    filesystem::path refpath;
};

class AccuracySuit : public testing::TestWithParam<Param> {};

TEST_P(AccuracySuit, TestDetector) {
    Param param = GetParam();
    ifstream file{param.refpath};
    stringstream ss;
    ss << file.rdbuf();
    EXPECT_EQ(ss.str(),
              string{*cached_model(param.model_name)
                          ->infer(cv::imread(data() + "/coco128/images/train2017/" + param.refpath.stem().string() +
                                             ".jpg"))->detection_result});
}

INSTANTIATE_TEST_SUITE_P(YOLOv8, AccuracySuit, testing::ValuesIn([] {
                             std::vector<Param> params;
                             for (const char* model_name : {"yolov5mu_openvino_model", "yolov8l_openvino_model"}) {
                                 vector<filesystem::path> refpaths;
                                 for (auto const& dir_entry :
                                      filesystem::directory_iterator{model_path(model_name) + "/ref/"}) {
                                     refpaths.push_back(dir_entry.path());
                                 }
                                 EXPECT_GT(refpaths.size(), 0);
                                 sort(refpaths.begin(), refpaths.end());
                                 for (const filesystem::path& refpath : refpaths) {
                                     params.push_back({model_name, refpath});
                                 }
                             }
                             return params;
                         }()));
}  // namespace
