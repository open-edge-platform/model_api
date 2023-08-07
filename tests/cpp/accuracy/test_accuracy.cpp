#include <stddef.h>

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <fstream>

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>

#include <gtest/gtest.h>

#include <models/classification_model.h>
#include <models/detection_model.h>
#include <models/input_data.h>
#include <models/instance_segmentation.h>
#include <models/results.h>
#include <models/segmentation_model.h>
#include <adapters/openvino_adapter.h>
#include <tilers/detection.h>
#include <tilers/instance_segmentation.h>

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

class ModelParameterizedTest : public testing::TestWithParam<ModelData> {
};

template<typename... Args>
std::string string_format(const std::string &fmt, Args... args)
{
    size_t size = snprintf(nullptr, 0, fmt.c_str(), args...);
    std::string buf;
    buf.reserve(size + 1);
    buf.resize(size);
    snprintf(&buf[0], size + 1, fmt.c_str(), args...);
    return buf;
}

inline void from_json(const nlohmann::json& j, ModelData& test)
{
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
std::vector<ModelData> GetTestData(const std::string& path)
{
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
    std::vector<std::shared_ptr<DetectionModel>> models{DetectionModel::create_model(model_path, {}, "", preload, "CPU")};
    if (std::string::npos != model_path.find("/serialized/")) {
        static ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        std::shared_ptr<InferenceAdapter> adapter = std::make_shared<OpenVINOInferenceAdapter>();
        adapter->loadModel(model, core, "CPU");
        models.push_back(DetectionModel::create_model(adapter));
    }
    return models;
}
}

TEST_P(ModelParameterizedTest, AccuracyTest)
{
    auto modelData = GetParam();

    std::string modelPath;
    const std::string& name = modelData.name;
    if (name.find(".onnx") != std::string::npos) {
        GTEST_SKIP() << "ONNX models are not supported in C++ implementation";
    }

    if (name.substr(name.size() - 4) == ".xml") {
        modelPath = DATA_DIR + '/' + name;
    } else {
        modelPath = DATA_DIR + '/' + string_format(MODEL_PATH_TEMPLATE, name.c_str(), name.c_str());
    }
    const std::string& basename = modelPath.substr(modelPath.find_last_of("/\\") + 1);
    for (const std::string& modelXml: {modelPath, DATA_DIR + "/serialized/" + basename}) {
        if (modelData.type == "DetectionModel") {
            for (const std::shared_ptr<DetectionModel>& model : create_models<DetectionModel>(modelXml)) {
                for (size_t i = 0; i < modelData.testData.size(); i++) {
                    ASSERT_EQ(modelData.testData[i].reference.size(), 1);
                    auto imagePath = DATA_DIR + "/" + modelData.testData[i].image;

                    cv::Mat image = cv::imread(imagePath);
                    if (!image.data) {
                        throw std::runtime_error{"Failed to read the image"};
                    }

                    std::unique_ptr<ResultBase> result;
                    if (modelData.tiler == "DetectionTiler") {
                        auto tiler = DetectionTiler(std::move(model), {});
                        if (modelData.input_res.height > 0 && modelData.input_res.width > 0) {
                            cv::resize(image, image, modelData.input_res);
                        }
                        result = tiler.run(image);
                    }
                    else {
                        result = model->infer(image);
                    }
                    EXPECT_EQ(std::string{*static_cast<DetectionResult*>(result.get())}, modelData.testData[i].reference[0]);
                }
            }
        }
        else if (modelData.type == "ClassificationModel") {
            for (const std::shared_ptr<ClassificationModel>& model : create_models<ClassificationModel>(modelXml)) {
                for (size_t i = 0; i < modelData.testData.size(); i++) {
                    ASSERT_EQ(modelData.testData[i].reference.size(), 1);
                    auto imagePath = DATA_DIR + "/" + modelData.testData[i].image;

                    cv::Mat image = cv::imread(imagePath);
                    if (!image.data) {
                        throw std::runtime_error{"Failed to read the image"};
                    }
                    auto result = model->infer(image);
                    EXPECT_EQ(std::string{*result}, modelData.testData[i].reference[0]);
                }
            }
        }
        else if (modelData.type == "SegmentationModel") {
            for (const std::shared_ptr<SegmentationModel>& model : create_models<SegmentationModel>(modelXml)) {
                for (size_t i = 0; i < modelData.testData.size(); i++) {
                    ASSERT_EQ(modelData.testData[i].reference.size(), 1);
                    auto imagePath = DATA_DIR + "/" + modelData.testData[i].image;

                    cv::Mat image = cv::imread(imagePath);
                    if (!image.data) {
                        throw std::runtime_error{"Failed to read the image"};
                    }

                    std::unique_ptr<ImageResult> pred = model->infer(image);
                    ImageResultWithSoftPrediction* soft = dynamic_cast<ImageResultWithSoftPrediction*>(pred.get());
                    if (soft) {
                        const std::vector<Contour>& contours = model->getContours(*soft);
                        std::stringstream ss;
                        ss << *soft << "; ";
                        for (const Contour& contour : contours) {
                            ss << contour << ", ";
                        }
                        ASSERT_EQ(ss.str(), modelData.testData[i].reference[0]);
                    } else {
                        ASSERT_EQ(std::string{*pred}, modelData.testData[i].reference[0]);
                    }
                }
            }
        }
        else if (modelData.type == "MaskRCNNModel") {
            for (const std::shared_ptr<MaskRCNNModel>& model : create_models<MaskRCNNModel>(modelXml)) {
                for (size_t i = 0; i < modelData.testData.size(); i++) {
                    ASSERT_EQ(modelData.testData[i].reference.size(), 1);
                    auto imagePath = DATA_DIR + "/" + modelData.testData[i].image;

                    cv::Mat image = cv::imread(imagePath);
                    if (!image.data) {
                        throw std::runtime_error{"Failed to read the image"};
                    }

                    std::unique_ptr<ResultBase> result;
                    if (modelData.tiler == "InstanceSegmentationTiler") {
                        auto tiler = InstanceSegmentationTiler(std::move(model), {});
                        if (modelData.input_res.height > 0 && modelData.input_res.width > 0) {
                            cv::resize(image, image, modelData.input_res);
                        }
                        result = tiler.run(image);
                    } else {
                        result = model->infer(image);
                    }
                    auto res = static_cast<InstanceSegmentationResult*>(result.get());

                    const std::vector<SegmentedObjectWithRects>& withRects = add_rotated_rects(res->segmentedObjects);
                    std::stringstream ss;
                    for (const SegmentedObjectWithRects& obj : withRects) {
                        ss << obj << "; ";
                    }
                    size_t filled = 0;
                    for (const cv::Mat_<std::uint8_t>& cls_map : res->saliency_map) {
                        if (cls_map.data) {
                            ++filled;
                        }
                    }
                    ss << filled << "; ";
                    try {
                        ss << res->feature_vector.get_shape();
                    } catch (ov::Exception&) {
                        ss << "[0]";
                    }
                    ss << "; ";
                    try {
                        // getContours() assumes each instance generates only one contour.
                        // That doesn't hold for some models
                        for (const Contour& contour : getContours(res->segmentedObjects)) {
                            ss << contour << "; ";
                        }
                    } catch (const std::runtime_error&) {}
                    EXPECT_EQ(ss.str(), modelData.testData[i].reference[0]);
                }
            }
        }
        else if (modelData.type == "AnomalyDetection"){
            // TODO: will be resolved in the C++ PR of the anomaly detection model
        }
        else {
            throw std::runtime_error("Unknown model type: " + modelData.type);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(TestAccuracyPublic, ModelParameterizedTest, testing::ValuesIn(GetTestData(PUBLIC_SCOPE_PATH)));

class InputParser{
    public:
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }

        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr = std::find(this->tokens.begin(), this->tokens.end(), option);
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
    std::cout << "Usage: " << program_name << " -p <path_to_public_scope.json> -d <path_to_data>" << std::endl;
}

int main(int argc, char **argv)
{
    InputParser input(argc, argv);

    if(input.cmdOptionExists("-h")){
        print_help(argv[0]);
        return 1;
    }
    const std::string &public_scope = input.getCmdOption("-p");
    if (!public_scope.empty()){
        PUBLIC_SCOPE_PATH = public_scope;
    }
    else{
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
