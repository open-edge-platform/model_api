/*
 * Copyright (C) 2020-2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
#include <adapters/openvino_adapter.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <tasks/classification.h>
#include <tasks/detection.h>
#include <tasks/results.h>

#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>

#include "utils/config.h"

using json = nlohmann::json;

std::string DATA_DIR = "../data";
std::string MODEL_PATH_TEMPLATE = "otx_models/%s.xml";
std::string IMAGE_PATH = "coco128/images/train2017/000000000074.jpg";

std::string TMP_MODEL_FILE = "tmp_model.xml";

struct ModelData {
    std::string name;
    ModelData(const std::string& name) : name(name) {}
};

class MockAdapter : public OpenVINOInferenceAdapter {
public:
    MockAdapter(const std::string& modelPath) : OpenVINOInferenceAdapter() {
        loadModel(modelPath, "CPU");
    }
};

class ClassificationModelParameterizedTest : public testing::TestWithParam<ModelData> {};

class SSDModelParameterizedTest : public testing::TestWithParam<ModelData> {};

class ClassificationModelParameterizedTestSaveLoad : public testing::TestWithParam<ModelData> {
protected:
    void TearDown() override {
        auto fileName = TMP_MODEL_FILE;
        std::remove(fileName.c_str());
        std::remove(fileName.replace(fileName.end() - 4, fileName.end(), ".bin").c_str());
    }
};

class DetectionModelParameterizedTestSaveLoad : public ClassificationModelParameterizedTestSaveLoad {};

template <typename... Args>
std::string string_format(const std::string& fmt, Args... args) {
    size_t size = snprintf(nullptr, 0, fmt.c_str(), args...);
    std::string buf;
    buf.reserve(size + 1);
    buf.resize(size);
    snprintf(&buf[0], size + 1, fmt.c_str(), args...);
    return buf;
}

TEST_P(ClassificationModelParameterizedTest, TestClassificationDefaultConfig) {
    auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
    bool preload = true;
    auto model = Classification::create_model(DATA_DIR + "/" + model_path, {}, preload, "CPU");

    auto config = model.adapter->getModelConfig();

    std::string model_type;
    model_type = utils::get_from_any_maps("model_type", config, {}, model_type);
    EXPECT_EQ(model_type, "Classification");

    bool embedded_processing;
    embedded_processing = utils::get_from_any_maps("embedded_processing", config, {}, embedded_processing);
    EXPECT_TRUE(embedded_processing);
}

TEST_P(ClassificationModelParameterizedTest, TestClassificationCustomConfig) {
    GTEST_SKIP() << "Classification config tests fail on CI";
    auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
    std::vector<std::string> mock_labels;
    size_t num_classes = 1000;
    for (size_t i = 0; i < num_classes; i++) {
        mock_labels.push_back(std::to_string(i));
    }
    ov::AnyMap configuration = {{"layout", "data:HWC"}, {"resize_type", "fit_to_window"}, {"labels", mock_labels}};
    bool preload = true;
    auto model = Classification::create_model(DATA_DIR + "/" + model_path, configuration, preload, "CPU");

    auto config = model.adapter->getModelConfig();
    std::string layout;
    layout = utils::get_from_any_maps("layout", config, {}, layout);
    EXPECT_EQ(layout, configuration.at("layout").as<std::string>());

    std::string resize_type;
    resize_type = utils::get_from_any_maps("resize_type", config, {}, resize_type);
    EXPECT_EQ(resize_type, configuration.at("resize_type").as<std::string>());

    std::vector<std::string> labels;
    labels = utils::get_from_any_maps("labels", config, {}, labels);
    for (size_t i = 0; i < num_classes; i++) {
        EXPECT_EQ(labels[i], mock_labels[i]);
    }
}

// TEST_P(ClassificationModelParameterizedTestSaveLoad, TestClassificationCorrectnessAfterSaveLoad) {
//     cv::Mat image = cv::imread(DATA_DIR + "/" + IMAGE_PATH);
//     if (!image.data) {
//         throw std::runtime_error{"Failed to read the image"};
//     }
//
//     auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
//     std::cout << model_path << "\n";
//     bool preload = true;
//     auto model = Classification::create_model(DATA_DIR + "/" + model_path, {}, preload, "CPU");
//
//     auto ov_model = model->getModel();
//     ov::serialize(ov_model, TMP_MODEL_FILE);
//
//     auto result = model->infer(image)->topLabels;
//
//     auto model_restored = ClassificationModel::create_model(TMP_MODEL_FILE, {}, preload, "CPU");
//     auto result_data = model_restored->infer(image);
//     auto result_restored = result_data->topLabels;
//
//     EXPECT_EQ(result_restored[0].id, result[0].id);
//     EXPECT_EQ(result_restored[0].score, result[0].score);
// }
//
// TEST_P(ClassificationModelParameterizedTestSaveLoad, TestClassificationCorrectnessAfterSaveLoadWithAdapter) {
//     cv::Mat image = cv::imread(DATA_DIR + "/" + IMAGE_PATH);
//     if (!image.data) {
//         throw std::runtime_error{"Failed to read the image"};
//     }
//
//     auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
//     bool preload = true;
//     auto model = ClassificationModel::create_model(DATA_DIR + "/" + model_path, {}, preload, "CPU");
//     auto ov_model = model->getModel();
//     ov::serialize(ov_model, TMP_MODEL_FILE);
//     auto result = model->infer(image)->topLabels;
//
//     std::shared_ptr<InferenceAdapter> adapter = std::make_shared<MockAdapter>(TMP_MODEL_FILE);
//     auto model_restored = ClassificationModel::create_model(adapter);
//     auto result_data = model_restored->infer(image);
//     auto result_restored = result_data->topLabels;
//
//     EXPECT_EQ(result_restored[0].id, result[0].id);
//     EXPECT_EQ(result_restored[0].score, result[0].score);
// }

TEST_P(SSDModelParameterizedTest, TestDetectionDefaultConfig) {
    auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
    bool preload = true;
    auto model = DetectionModel::create_model(DATA_DIR + "/" + model_path, {}, preload, "CPU");

    auto config = model.algorithm->adapter->getModelConfig();

    std::string model_type;
    model_type = utils::get_from_any_maps("model_type", config, {}, model_type);
    EXPECT_EQ(model_type, "ssd");

    bool embedded_processing;
    embedded_processing = utils::get_from_any_maps("embedded_processing", config, {}, embedded_processing);
    EXPECT_TRUE(embedded_processing);
}

TEST_P(SSDModelParameterizedTest, TestDetectionCustomConfig) {
    GTEST_SKIP() << "Detection config tests fail on CI";
    auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
    std::vector<std::string> mock_labels;
    size_t num_classes = 80;
    for (size_t i = 0; i < num_classes; i++) {
        mock_labels.push_back(std::to_string(i));
    }
    ov::AnyMap configuration = {{"layout", "data:HWC"}, {"resize_type", "fit_to_window"}, {"labels", mock_labels}};
    bool preload = true;
    auto model = DetectionModel::create_model(DATA_DIR + "/" + model_path, configuration, preload, "CPU");

    auto config = model.algorithm->adapter->getModelConfig();
    std::string layout;
    layout = utils::get_from_any_maps("layout", config, {}, layout);
    EXPECT_EQ(layout, configuration.at("layout").as<std::string>());

    std::string resize_type;
    resize_type = utils::get_from_any_maps("resize_type", config, {}, resize_type);
    EXPECT_EQ(resize_type, configuration.at("resize_type").as<std::string>());

    std::vector<std::string> labels;
    labels = utils::get_from_any_maps("labels", config, {}, labels);
    for (size_t i = 0; i < num_classes; i++) {
        EXPECT_EQ(labels[i], mock_labels[i]);
    }
}

// TEST_P(DetectionModelParameterizedTestSaveLoad, TestDetctionCorrectnessAfterSaveLoad) {
//     cv::Mat image = cv::imread(DATA_DIR + "/" + IMAGE_PATH);
//     if (!image.data) {
//         throw std::runtime_error{"Failed to read the image"};
//     }
//
//     auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
//     bool preload = true;
//     auto model = DetectionModel::create_model(DATA_DIR + "/" + model_path, {}, preload, "CPU");
//
//     auto ov_model = model->getModel();
//     ov::serialize(ov_model, TMP_MODEL_FILE);
//
//     auto result = model->infer(image)->objects;
//
//     image = cv::imread(DATA_DIR + "/" + IMAGE_PATH);
//     if (!image.data) {
//         throw std::runtime_error{"Failed to read the image"};
//     }
//     auto model_restored = DetectionModel::create_model(TMP_MODEL_FILE, {}, "", preload, "CPU");
//     auto result_data = model_restored->infer(image);
//     auto result_restored = result_data->objects;
//
//     ASSERT_EQ(result.size(), result_restored.size());
//
//     for (size_t i = 0; i < result.size(); i++) {
//         ASSERT_EQ(result[i].x, result_restored[i].x);
//         ASSERT_EQ(result[i].y, result_restored[i].y);
//         ASSERT_EQ(result[i].width, result_restored[i].width);
//         ASSERT_EQ(result[i].height, result_restored[i].height);
//     }
// }
//
// TEST_P(DetectionModelParameterizedTestSaveLoad, TestDetctionCorrectnessAfterSaveLoadWithAdapter) {
//     cv::Mat image = cv::imread(DATA_DIR + "/" + IMAGE_PATH);
//     if (!image.data) {
//         throw std::runtime_error{"Failed to read the image"};
//     }
//
//     auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
//     bool preload = true;
//     auto model = DetectionModel::create_model(DATA_DIR + "/" + model_path, {}, "", preload, "CPU");
//     auto ov_model = model->getModel();
//     ov::serialize(ov_model, TMP_MODEL_FILE);
//     auto result = model->infer(image)->objects;
//
//     image = cv::imread(DATA_DIR + "/" + IMAGE_PATH);
//     if (!image.data) {
//         throw std::runtime_error{"Failed to read the image"};
//     }
//
//     std::shared_ptr<InferenceAdapter> adapter = std::make_shared<MockAdapter>(TMP_MODEL_FILE);
//     auto model_restored = DetectionModel::create_model(adapter);
//     auto result_data = model_restored->infer(image);
//     auto result_restored = result_data->objects;
//
//     ASSERT_EQ(result.size(), result_restored.size());
//
//     for (size_t i = 0; i < result.size(); i++) {
//         ASSERT_EQ(result[i].x, result_restored[i].x);
//         ASSERT_EQ(result[i].y, result_restored[i].y);
//         ASSERT_EQ(result[i].width, result_restored[i].width);
//         ASSERT_EQ(result[i].height, result_restored[i].height);
//     }
// }

INSTANTIATE_TEST_SUITE_P(ClassificationTestInstance,
                         ClassificationModelParameterizedTest,
                         ::testing::Values(ModelData("mlc_mobilenetv3_large_voc")));
// INSTANTIATE_TEST_SUITE_P(ClassificationTestInstance,
//                          ClassificationModelParameterizedTestSaveLoad,
//                          ::testing::Values(ModelData("mlc_mobilenetv3_large_voc")));
INSTANTIATE_TEST_SUITE_P(SSDTestInstance,
                         SSDModelParameterizedTest,
                         ::testing::Values(ModelData("detection_model_with_xai_head")));
// INSTANTIATE_TEST_SUITE_P(SSDTestInstance,
//                          DetectionModelParameterizedTestSaveLoad,
//                          ::testing::Values(ModelData("detection_model_with_xai_head")));

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
    std::cout << "Usage: " << program_name << "-d <path_to_data>" << std::endl;
}

int main(int argc, char** argv) {
    InputParser input(argc, argv);

    if (input.cmdOptionExists("-h")) {
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
