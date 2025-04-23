/*
 * Copyright (C) 2020-2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
#include <adapters/openvino_adapter.h>
#include <gtest/gtest.h>
#include <models/anomaly_model.h>
#include <models/classification_model.h>
#include <models/detection_model.h>
#include <models/input_data.h>
#include <models/instance_segmentation.h>
#include <models/keypoint_detection.h>
#include <models/results.h>
#include <models/segmentation_model.h>
#include <stddef.h>
#include <tilers/detection.h>
#include <tilers/instance_segmentation.h>
#include <tilers/semantic_segmentation.h>

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

std::string PUBLIC_SCOPE_PATH = "../../tests/cpp/accuracy/public_scope.json";
std::string DATA_DIR = "../data";
std::string MODEL_PATH_TEMPLATE = "public/%s/FP16/%s.xml";

struct TestData {
    std::string image;
    std::vector<std::string> reference;
};

struct ModelData {
    std::string name;
    std::string type;
    std::vector<TestData> testData;
    std::string tiler;
    cv::Size input_res = cv::Size(0, 0);
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
    for (auto& item : j.at("test_data")) {
        TestData data;
        data.image = item.at("image").get<std::string>();
        for (auto& ref : item.at("reference")) {
            data.reference.push_back(ref.get<std::string>());
        }
        test.testData.push_back(data);
    }
    if (j.contains("tiler")) {
        test.tiler = j.at("tiler").get<std::string>();
    }
    if (j.contains("input_res")) {
        auto res = j.at("input_res").get<std::string>();
        res.erase(std::remove(res.begin(), res.end(), '('), res.end());
        res.erase(std::remove(res.begin(), res.end(), ')'), res.end());
        test.input_res.width = std::stoi(res.substr(0, res.find(',')));
        res.erase(0, res.find(',') + 1);
        test.input_res.height = std::stoi(res);
    }
}

namespace {
std::vector<ModelData> GetTestData(const std::string& path) {
    std::ifstream input(path);
    nlohmann::json j;
    input >> j;
    return j;
}

template <typename Type>
std::vector<std::shared_ptr<Type>> create_models(const std::string& model_path) {
    bool preload = true;
    std::vector<std::shared_ptr<Type>> models{Type::create_model(model_path, {}, preload, "CPU")};
    if (std::string::npos != model_path.find("/serialized/")) {
        static ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        std::shared_ptr<InferenceAdapter> adapter = std::make_shared<OpenVINOInferenceAdapter>();
        adapter->loadModel(model, core, "CPU");
        models.push_back(Type::create_model(adapter));
    }
    return models;
}

template <>
std::vector<std::shared_ptr<DetectionModel>> create_models(const std::string& model_path) {
    bool preload = true;
    std::vector<std::shared_ptr<DetectionModel>> models{
        DetectionModel::create_model(model_path, {}, "", preload, "CPU")};
    if (std::string::npos != model_path.find("/serialized/")) {
        static ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        std::shared_ptr<InferenceAdapter> adapter = std::make_shared<OpenVINOInferenceAdapter>();
        adapter->loadModel(model, core, "CPU");
        models.push_back(DetectionModel::create_model(adapter));
    }
    return models;
}
}  // namespace

TEST_P(ModelParameterizedTest, AccuracyTest) {
    auto modelData = GetParam();

    std::string modelPath;
    const std::string& name = modelData.name;
    if (name.find(".onnx") != std::string::npos) {
        GTEST_SKIP() << "ONNX models are not supported in C++ implementation";
    }
    if (name.find("action_cls_xd3_kinetic") != std::string::npos) {
        GTEST_SKIP() << "ActionClassificationModel is not supported in C++ implementation";
    }
    if (name.find("mobilenet_v3_large_hc_cf") != std::string::npos) {
        GTEST_SKIP() << "mobilenet_v3_large_hc_cf fails in OV 2025.0";
    }
    if (name.find("anomaly_padim_bottle_mvtec") != std::string::npos) {
        GTEST_SKIP() << "anomaly_padim_bottle_mvtec fails in OV 2025.0";
    }
    if (name.find("sam_vit_b") != std::string::npos) {
        GTEST_SKIP() << "SAM-based models are not supported in C++ implementation";
    }

    if (name.substr(name.size() - 4) == ".xml") {
        modelPath = DATA_DIR + '/' + name;
    } else {
        modelPath = DATA_DIR + '/' + string_format(MODEL_PATH_TEMPLATE, name.c_str(), name.c_str());
    }
    const std::string& basename = modelPath.substr(modelPath.find_last_of("/\\") + 1);
    for (const std::string& modelXml : {modelPath, DATA_DIR + "/serialized/" + basename}) {
        if (modelData.type == "DetectionModel") {
            for (const std::shared_ptr<DetectionModel>& model : create_models<DetectionModel>(modelXml)) {
                for (size_t i = 0; i < modelData.testData.size(); i++) {
                    ASSERT_EQ(modelData.testData[i].reference.size(), 1);
                    auto imagePath = DATA_DIR + "/" + modelData.testData[i].image;

                    cv::Mat image = cv::imread(imagePath);
                    if (!image.data) {
                        throw std::runtime_error{"Failed to read the image"};
                    }

                    std::unique_ptr<Scene> result;
                    if (modelData.tiler == "DetectionTiler") {
                        auto tiler = DetectionTiler(std::move(model), {});
                        if (modelData.input_res.height > 0 && modelData.input_res.width > 0) {
                            cv::resize(image, image, modelData.input_res);
                        }
                        result = tiler.run(image);
                    } else {
                        result = model->infer(image);
                    }
                    EXPECT_EQ(std::string{*result->detection_result}, modelData.testData[i].reference[0]);
                }
            }
        } else if (modelData.type == "ClassificationModel") {
            for (const std::shared_ptr<ClassificationModel>& model : create_models<ClassificationModel>(modelXml)) {
                for (size_t i = 0; i < modelData.testData.size(); i++) {
                    ASSERT_EQ(modelData.testData[i].reference.size(), 1);
                    auto imagePath = DATA_DIR + "/" + modelData.testData[i].image;

                    cv::Mat image = cv::imread(imagePath);
                    if (!image.data) {
                        throw std::runtime_error{"Failed to read the image"};
                    }
                    auto result = model->infer(image);
                    EXPECT_EQ(std::string{*result->classification_result}, modelData.testData[i].reference[0]);
                }
            }
        } else if (modelData.type == "SegmentationModel") {
            for (const std::shared_ptr<SegmentationModel>& model : create_models<SegmentationModel>(modelXml)) {
                for (size_t i = 0; i < modelData.testData.size(); i++) {
                    ASSERT_EQ(modelData.testData[i].reference.size(), 1);
                    auto imagePath = DATA_DIR + "/" + modelData.testData[i].image;

                    cv::Mat image = cv::imread(imagePath);
                    if (!image.data) {
                        throw std::runtime_error{"Failed to read the image"};
                    }

                    std::unique_ptr<Scene> pred;
                    if (modelData.tiler == "SemanticSegmentationTiler") {
                        auto tiler = SemanticSegmentationTiler(std::move(model), {});
                        if (modelData.input_res.height > 0 && modelData.input_res.width > 0) {
                            cv::resize(image, image, modelData.input_res);
                        }
                        pred = tiler.run(image);
                    } else {
                        pred = model->infer(image);
                    }

                    ImageResultWithSoftPrediction* soft = dynamic_cast<ImageResultWithSoftPrediction*>(pred->image_result.get());
                    if (soft) {
                        const std::vector<Contour>& contours = model->getContours(*soft);
                        std::stringstream ss;
                        ss << *soft << "; ";
                        for (const Contour& contour : contours) {
                            ss << contour << ", ";
                        }
                        ASSERT_EQ(ss.str(), modelData.testData[i].reference[0]);
                    } else {
                        ASSERT_EQ(std::string{*pred->image_result}, modelData.testData[i].reference[0]);
                    }
                }
            }
        } else if (modelData.type == "MaskRCNNModel") {
            for (const std::shared_ptr<MaskRCNNModel>& model : create_models<MaskRCNNModel>(modelXml)) {
                for (size_t i = 0; i < modelData.testData.size(); i++) {
                    ASSERT_EQ(modelData.testData[i].reference.size(), 1);
                    auto imagePath = DATA_DIR + "/" + modelData.testData[i].image;

                    cv::Mat image = cv::imread(imagePath);
                    if (!image.data) {
                        throw std::runtime_error{"Failed to read the image"};
                    }

                    std::unique_ptr<Scene> result;
                    if (modelData.tiler == "InstanceSegmentationTiler") {
                        auto tiler = InstanceSegmentationTiler(std::move(model), {});
                        if (modelData.input_res.height > 0 && modelData.input_res.width > 0) {
                            cv::resize(image, image, modelData.input_res);
                        }
                        result = tiler.run(image);
                    } else {
                        result = model->infer(image);
                    }

                    const std::vector<SegmentedObjectWithRects>& withRects =
                        add_rotated_rects(result->instance_segmentation_result->segmentedObjects);
                    std::stringstream ss;
                    for (const SegmentedObjectWithRects& obj : withRects) {
                        ss << obj << "; ";
                    }
                    size_t filled = 0;
                    for (const cv::Mat_<std::uint8_t>& cls_map : result->instance_segmentation_result->saliency_map) {
                        if (cls_map.data) {
                            ++filled;
                        }
                    }
                    ss << filled << "; ";
                    try {
                        ss << result->instance_segmentation_result->feature_vector.get_shape();
                    } catch (ov::Exception&) {
                        ss << "[0]";
                    }
                    ss << "; ";
                    try {
                        // getContours() assumes each instance generates only one contour.
                        // That doesn't hold for some models
                        for (const Contour& contour : getContours(result->instance_segmentation_result->segmentedObjects)) {
                            ss << contour << "; ";
                        }
                    } catch (const std::runtime_error&) {
                    }
                    EXPECT_EQ(ss.str(), modelData.testData[i].reference[0]);
                }
            }
        } else if (modelData.type == "AnomalyDetection") {
            for (const std::shared_ptr<AnomalyModel>& model : create_models<AnomalyModel>(modelXml)) {
                for (size_t i = 0; i < modelData.testData.size(); i++) {
                    ASSERT_EQ(modelData.testData[i].reference.size(), 1);
                    auto imagePath = DATA_DIR + "/" + modelData.testData[i].image;

                    cv::Mat image = cv::imread(imagePath);
                    if (!image.data) {
                        throw std::runtime_error{"Failed to read the image"};
                    }
                    auto result = model->infer(image);
                    EXPECT_EQ(std::string{*result->anomaly_result}, modelData.testData[i].reference[0]);
                }
            }
        } else if (modelData.type == "KeypointDetectionModel") {
            for (const std::shared_ptr<KeypointDetectionModel>& model :
                 create_models<KeypointDetectionModel>(modelXml)) {
                for (size_t i = 0; i < modelData.testData.size(); i++) {
                    if (i == 0) {
                        GTEST_SKIP() << "OV gives different results on unpreprocessed keypoint model";
                    }
                    ASSERT_EQ(modelData.testData[i].reference.size(), 1);
                    auto imagePath = DATA_DIR + "/" + modelData.testData[i].image;

                    cv::Mat image = cv::imread(imagePath);
                    if (!image.data) {
                        throw std::runtime_error{"Failed to read the image"};
                    }
                    auto result = model->infer(image);
                    EXPECT_EQ(std::string{(*result->keypoint_detection_result).poses[0]}, modelData.testData[i].reference[0]);
                }
            }
        }

        else {
            throw std::runtime_error("Unknown model type: " + modelData.type);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(TestAccuracyPublic, ModelParameterizedTest, testing::ValuesIn(GetTestData(PUBLIC_SCOPE_PATH)));

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
