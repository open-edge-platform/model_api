/*
 * Copyright (C) 2020-2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <stddef.h>
#include <tasks/classification.h>
#include <tasks/detection.h>
#include <tasks/results.h>
#include <tasks/semantic_segmentation.h>

#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

std::string PUBLIC_SCOPE_PATH = "../../tests/cpp/precommit/public_scope.json";
std::string DATA_DIR = "../data";
std::string MODEL_PATH_TEMPLATE = "public/%s/FP16/%s.xml";
std::string IMAGE_PATH = "coco128/images/train2017/000000000074.jpg";

struct ModelData {
    std::string name;
    std::string type;
};

class ModelParameterizedTest : public testing::TestWithParam<ModelData> {};

template <typename... Args>
std::string string_format(const std::string& fmt, Args... args) {
    size_t size = snprintf(nullptr, 0, fmt.c_str(), args...);
    std::string buf;
    buf.reserve(size + 1);
    buf.resize(size);
    snprintf(&buf[0], size + 1, fmt.c_str(), args...);
    return buf;
}

inline void from_json(const nlohmann::json& j, ModelData& test) {
    test.name = j.at("name").get<std::string>();
    test.type = j.at("type").get<std::string>();
}

std::vector<ModelData> GetTestData(const std::string& path) {
    std::ifstream input(path);
    nlohmann::json j;
    input >> j;
    return j;
}
TEST_P(ModelParameterizedTest, SynchronousInference) {
    cv::Mat image = cv::imread(DATA_DIR + "/" + IMAGE_PATH);
    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }

    std::string model_path;
    const std::string& name = GetParam().name;
    if (name.substr(name.size() - 4) == ".xml") {
        model_path = name;
    } else {
        model_path = string_format(MODEL_PATH_TEMPLATE, name.c_str(), name.c_str());
    }

    if ("DetectionModel" == GetParam().type) {
        bool preload = true;
        auto model = DetectionModel::create_model(DATA_DIR + "/" + model_path, {}, preload, "CPU");
        auto result = model.infer(image);
        EXPECT_GT(result.objects.size(), 0);
    } else if ("ClassificationModel" == GetParam().type) {
        bool preload = true;
        auto model = Classification::create_model(DATA_DIR + "/" + model_path, {}, preload, "CPU");
        auto result = model.infer(image);
        ASSERT_GT(result.topLabels.size(), 0);
        EXPECT_GT(result.topLabels.front().score, 0.0f);
    } else if ("SegmentationModel" == GetParam().type) {
        bool preload = true;
        auto model = SemanticSegmentation::create_model(DATA_DIR + "/" + model_path, {}, preload, "CPU");
        auto result = model.infer(image);
        ASSERT_GT(model.getContours(result).size(), 0);
    }
}

INSTANTIATE_TEST_SUITE_P(TestSanityPublic, ModelParameterizedTest, testing::ValuesIn(GetTestData(PUBLIC_SCOPE_PATH)));

class InputParser {
public:
    InputParser(int& argc, char** argv) {
        for (int i = 1; i < argc; ++i)
            this->tokens.push_back(std::string(argv[i]));
    }

    const std::string& getCmdOption(const std::string& option) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
            return *itr;
        }
        static const std::string empty_string("");
        return empty_string;
    }

    bool cmdOptionExists(const std::string& option) const {
        return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
    }

private:
    std::vector<std::string> tokens;
};

void print_help(const char* program_name) {
    std::cout << "Usage: " << program_name << " -p <path_to_public_scope.json> -d <path_to_data>" << std::endl;
}

int main(int argc, char** argv) {
    InputParser input(argc, argv);

    if (input.cmdOptionExists("-h")) {
        print_help(argv[0]);
        return 1;
    }
    const std::string& public_scope = input.getCmdOption("-p");
    if (!public_scope.empty()) {
        PUBLIC_SCOPE_PATH = public_scope;
    } else {
        print_help(argv[0]);
        return 1;
    }
    const std::string& data_dir = input.getCmdOption("-d");
    if (!data_dir.empty()) {
        DATA_DIR = data_dir;
    } else {
        print_help(argv[0]);
        return 1;
    }

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
