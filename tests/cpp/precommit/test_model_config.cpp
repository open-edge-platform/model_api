#include <stddef.h>

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <cstdio>

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>

#include <gtest/gtest.h>

#include <models/classification_model.h>
#include <models/detection_model.h>
#include <models/detection_model_ssd.h>
#include <models/input_data.h>
#include <models/results.h>
#include <adapters/openvino_adapter.h>

using json = nlohmann::json;

std::string DATA_DIR = "../data";
std::string MODEL_PATH_TEMPLATE = "public/%s/FP16/%s.xml";
std::string IMAGE_PATH = "coco128/images/train2017/000000000074.jpg";

std::string TMP_MODEL_FILE = "tmp_model.xml";

struct ModelData {
    std::string name;
    ModelData(const std::string& name)
        : name(name) {}
};

class MockAdapter : public OpenVINOInferenceAdapter {
    public:
        MockAdapter(const std::string& modelPath)
            : OpenVINOInferenceAdapter() {
            auto core = ov::Core();
            auto model = core.read_model(modelPath);
            loadModel(model, core, "AUTO");
        }
};

class ClassificationModelParameterizedTest : public testing::TestWithParam<ModelData> {
};

class SSDModelParameterizedTest : public testing::TestWithParam<ModelData> {
};

class ClassificationModelParameterizedTestSaveLoad : public testing::TestWithParam<ModelData> {
    protected:
        void TearDown() override {
            auto fileName = TMP_MODEL_FILE;
            std::remove(fileName.c_str());
            std::remove(fileName.replace(fileName.end() - 4, fileName.end(), ".bin").c_str());
        }
};

class DetectionModelParameterizedTestSaveLoad : public ClassificationModelParameterizedTestSaveLoad {
};

template<typename... Args>
std::string string_format(const std::string &fmt, Args... args) {
    size_t size = snprintf(nullptr, 0, fmt.c_str(), args...);
    std::string buf;
    buf.reserve(size + 1);
    buf.resize(size);
    snprintf(&buf[0], size + 1, fmt.c_str(), args...);
    return buf;
}

TEST_P(ClassificationModelParameterizedTest, TestClassificationDefaultConfig) {
    auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
    auto model = ClassificationModel::create_model(DATA_DIR + "/" + model_path);

    auto ov_model = model->getModel();

    EXPECT_EQ(ov_model->get_rt_info<std::string>("model_info", "model_type"), ClassificationModel::ModelType);

    auto embedded_processing = ov_model->get_rt_info<bool>("model_info", "embedded_processing");
    EXPECT_TRUE(embedded_processing);

    SUCCEED();
}

TEST_P(ClassificationModelParameterizedTest, TestClassificationCustomConfig) {
    auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
    std::vector<std::string> mock_labels;
    size_t num_classes = 1000;
    for (size_t i = 0; i < num_classes; i++) {
        mock_labels.push_back(std::to_string(i));
    }
    ov::AnyMap configuration = {
        {"layout", "data:HWC"},
        {"resize_type", "standard"}, //fit_to_window
        {"labels", mock_labels}
    };
    auto model = ClassificationModel::create_model(DATA_DIR + "/" + model_path, configuration);

    auto ov_model = model->getModel();

    auto layout = ov_model->get_rt_info<std::string>("model_info", "layout");
    EXPECT_EQ(layout, configuration.at("layout").as<std::string>());

    auto resize_type = ov_model->get_rt_info<std::string>("model_info", "resize_type");
    EXPECT_EQ(resize_type, configuration.at("resize_type").as<std::string>());

    auto labels = split(ov_model->get_rt_info<std::string>("model_info", "labels"), ' ');
    for (size_t i = 0; i < num_classes; i++) {
        EXPECT_EQ(labels[i], mock_labels[i]);
    }

    SUCCEED();
}

TEST_P(ClassificationModelParameterizedTestSaveLoad, TestClassificationCorrectnessAfterSaveLoad) {
    cv::Mat image = cv::imread(DATA_DIR + "/" + IMAGE_PATH);
    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }

    auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
    auto model = ClassificationModel::create_model(DATA_DIR + "/" + model_path);

    auto ov_model = model->getModel();
    ov::serialize(ov_model, TMP_MODEL_FILE);
    
    auto result = model->infer(image)->topLabels;

    auto model_restored = ClassificationModel::create_model(TMP_MODEL_FILE);
    auto result_data = model_restored->infer(image);
    auto result_restored = result_data->topLabels;

    EXPECT_EQ(result_restored[0].id, result[0].id);
    EXPECT_EQ(result_restored[0].score, result[0].score);

    SUCCEED();
}

TEST_P(ClassificationModelParameterizedTestSaveLoad, TestClassificationCorrectnessAfterSaveLoadWithAdapter) {
    cv::Mat image = cv::imread(DATA_DIR + "/" + IMAGE_PATH);
    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }

    auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
    auto model = ClassificationModel::create_model(DATA_DIR + "/" + model_path);
    auto ov_model = model->getModel();
    ov::serialize(ov_model, TMP_MODEL_FILE);
    auto result = model->infer(image)->topLabels;

    std::shared_ptr<InferenceAdapter> adapter = std::make_shared<MockAdapter>(TMP_MODEL_FILE);
    auto model_restored = ClassificationModel::create_model(adapter);
    auto result_data = model_restored->infer(image);
    auto result_restored = result_data->topLabels;

    EXPECT_EQ(result_restored[0].id, result[0].id);
    EXPECT_EQ(result_restored[0].score, result[0].score);

    SUCCEED();
}

TEST_P(SSDModelParameterizedTest, TestDetectionDefaultConfig) {
    auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
    auto model = DetectionModel::create_model(DATA_DIR + "/" + model_path);

    auto ov_model = model->getModel();

    EXPECT_EQ(ov_model->get_rt_info<std::string>("model_info", "model_type"), ModelSSD::ModelType);
    
    auto embedded_processing = ov_model->get_rt_info<bool>("model_info", "embedded_processing");
    EXPECT_TRUE(embedded_processing);

    SUCCEED();
}

TEST_P(SSDModelParameterizedTest, TestDetectionCustomConfig) {
    auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
    std::vector<std::string> mock_labels;
    size_t num_classes = 80;
    for (size_t i = 0; i < num_classes; i++) {
        mock_labels.push_back(std::to_string(i));
    }
    ov::AnyMap configuration = {
        {"layout", "data:HWC"},
        {"resize_type", "standard"}, //fit_to_window
        {"labels", mock_labels}
    };
    auto model = DetectionModel::create_model(DATA_DIR + "/" + model_path, configuration);

    auto ov_model = model->getModel();

    auto layout = ov_model->get_rt_info<std::string>("model_info", "layout");
    EXPECT_EQ(layout, configuration.at("layout").as<std::string>());

    auto resize_type = ov_model->get_rt_info<std::string>("model_info", "resize_type");
    EXPECT_EQ(resize_type, configuration.at("resize_type").as<std::string>());

    auto labels = split(ov_model->get_rt_info<std::string>("model_info", "labels"), ' ');
    for (size_t i = 0; i < num_classes; i++) {
        EXPECT_EQ(labels[i], mock_labels[i]);
    }

    SUCCEED();
}

TEST_P(DetectionModelParameterizedTestSaveLoad, TestDetctionCorrectnessAfterSaveLoad) {
    cv::Mat image = cv::imread(DATA_DIR + "/" + IMAGE_PATH);
    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }

    auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
    auto model = DetectionModel::create_model(DATA_DIR + "/" + model_path);

    auto ov_model = model->getModel();
    ov::serialize(ov_model, TMP_MODEL_FILE);
    
    auto result = model->infer(image)->objects;

    image = cv::imread(DATA_DIR + "/" + IMAGE_PATH);
    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }
    auto model_restored = DetectionModel::create_model(TMP_MODEL_FILE);
    auto result_data = model_restored->infer(image);
    auto result_restored = result_data->objects;

    ASSERT_EQ(result.size(), result_restored.size());

    for (size_t i = 0; i < result.size(); i++) {
        ASSERT_EQ(result[i].x, result_restored[i].x);
        ASSERT_EQ(result[i].y, result_restored[i].y);
        ASSERT_EQ(result[i].width, result_restored[i].width);
        ASSERT_EQ(result[i].height, result_restored[i].height);
    }

    SUCCEED();
}

TEST_P(DetectionModelParameterizedTestSaveLoad, TestDetctionCorrectnessAfterSaveLoadWithAdapter) {
    cv::Mat image = cv::imread(DATA_DIR + "/" + IMAGE_PATH);
    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }

    auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
    auto model = DetectionModel::create_model(DATA_DIR + "/" + model_path);
    auto ov_model = model->getModel();
    ov::serialize(ov_model, TMP_MODEL_FILE);
    auto result = model->infer(image)->objects;

    image = cv::imread(DATA_DIR + "/" + IMAGE_PATH);
    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }

    std::shared_ptr<InferenceAdapter> adapter = std::make_shared<MockAdapter>(TMP_MODEL_FILE);
    auto model_restored = DetectionModel::create_model(adapter);
    auto result_data = model_restored->infer(image);
    auto result_restored = result_data->objects;

    ASSERT_EQ(result.size(), result_restored.size());

    for (size_t i = 0; i < result.size(); i++) {
        ASSERT_EQ(result[i].x, result_restored[i].x);
        ASSERT_EQ(result[i].y, result_restored[i].y);
        ASSERT_EQ(result[i].width, result_restored[i].width);
        ASSERT_EQ(result[i].height, result_restored[i].height);
    }

    SUCCEED();
}


INSTANTIATE_TEST_SUITE_P(ClassificationTestInstance, ClassificationModelParameterizedTest, ::testing::Values(ModelData("efficientnet-b0-pytorch")));
INSTANTIATE_TEST_SUITE_P(ClassificationTestInstance, ClassificationModelParameterizedTestSaveLoad, ::testing::Values(ModelData("efficientnet-b0-pytorch")));
INSTANTIATE_TEST_SUITE_P(SSDTestInstance, SSDModelParameterizedTest, ::testing::Values(ModelData("ssdlite_mobilenet_v2"), ModelData("ssd_mobilenet_v1_fpn_coco")));
INSTANTIATE_TEST_SUITE_P(SSDTestInstance, DetectionModelParameterizedTestSaveLoad, ::testing::Values(ModelData("ssdlite_mobilenet_v2"), ModelData("ssd_mobilenet_v1_fpn_coco")));

class InputParser{
    public:
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }

        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }

        bool cmdOptionExists(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};

void print_help(const char* program_name)
{
    std::cout << "Usage: " << program_name << "-d <path_to_data>" << std::endl;
}

int main(int argc, char **argv)
{
    InputParser input(argc, argv);

    if(input.cmdOptionExists("-h")){
        print_help(argv[0]);
        return 1;
    }

    const std::string &data_dir = input.getCmdOption("-d");
    if (!data_dir.empty()){
        DATA_DIR = data_dir;
    }
    else{
        print_help(argv[0]);
        return 1;
    }

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
