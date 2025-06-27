#include <gtest/gtest.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <thread>

#include "matchers.h"
#include "tasks/anomaly.h"
#include "tasks/classification.h"
#include "tasks/detection.h"
#include "tasks/instance_segmentation.h"
#include "tasks/semantic_segmentation.h"

std::string PUBLIC_SCOPE_PATH = "../public_scope.json";
std::string DATA_DIR = "../data";

struct TestData {
    std::string image;
    std::vector<std::string> reference;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TestData, image, reference);

cv::Mat load_image(const std::string& path, bool use_tiling, cv::Size size) {
    cv::Mat image = cv::imread(path);
    if (use_tiling) {
        cv::resize(image, image, size);
    }
    return image;
}

struct ModelData {
    std::string name;
    std::string type;
    std::vector<TestData> test_data;
    std::string tiler = "";
    cv::Size input_res = cv::Size(0, 0);
};

void from_json(const nlohmann::json& nlohmann_json_j, ModelData& nlohmann_json_t) {
    nlohmann_json_t.name = nlohmann_json_j["name"];
    nlohmann_json_t.type = nlohmann_json_j["type"];
    nlohmann_json_t.test_data = nlohmann_json_j["test_data"];
    nlohmann_json_t.tiler = (nlohmann_json_j.contains("tiler") ? nlohmann_json_j["tiler"] : "");
    if (nlohmann_json_j.contains("input_res")) {
        auto res = nlohmann_json_j.at("input_res").get<std::string>();
        res.erase(std::remove(res.begin(), res.end(), '('), res.end());
        res.erase(std::remove(res.begin(), res.end(), ')'), res.end());
        nlohmann_json_t.input_res.width = std::stoi(res.substr(0, res.find(',')));
        res.erase(0, res.find(',') + 1);
        nlohmann_json_t.input_res.height = std::stoi(res);
    }
}

void to_json(nlohmann::json& nlohmann_json_j, const ModelData& nlohmann_json_t) {
    nlohmann_json_j["name"] = nlohmann_json_t.name;
    nlohmann_json_j["type"] = nlohmann_json_t.type;
    nlohmann_json_j["test_data"] = nlohmann_json_t.test_data;
    if (!nlohmann_json_t.tiler.empty()) {
        nlohmann_json_j["tiler"] = nlohmann_json_t.tiler;
    }
    if (!nlohmann_json_t.input_res.empty()) {
        nlohmann_json_j["input_res"] = "(" + std::to_string(nlohmann_json_t.input_res.width) + "," +
                                       std::to_string(nlohmann_json_t.input_res.width) + ")";
    }
}

void PrintTo(const ModelData& param, std::ostream* os) {
    nlohmann::json d = param;

    *os << "TestCase(name=" << param.name << ",\n"
        << "expected=" << param.test_data[0].reference[0] << ",\n"
        << "input=" << d.dump(2);
    *os << "\n)";
}

class ModelParameterizedTest : public testing::TestWithParam<ModelData> {
public:
    struct PrintToStringParamName {
        std::string operator()(const testing::TestParamInfo<ModelData>& info) const {
            auto data = info.param;
            return std::to_string(info.index) + "_" + data.type;
        }
    };
};

TEST_P(ModelParameterizedTest, AccuracyTest) {
    auto data = GetParam();
    auto model_path = DATA_DIR + '/' + data.name;

    auto use_tiling = !data.input_res.empty();
    if (data.type == "DetectionModel") {
        auto model = DetectionModel::load(model_path, {{"tiling", use_tiling}});

        for (auto& test_data : data.test_data) {
            std::string image_path = DATA_DIR + '/' + test_data.image;
            auto image = load_image(image_path, use_tiling, data.input_res);
            auto result = model.infer(image);
            EXPECT_EQ(std::string{result}, test_data.reference[0]);
        }

    } else if (data.type == "SegmentationModel") {
        auto model = SemanticSegmentation::load(model_path, {{"tiling", use_tiling}});

        for (auto& test_data : data.test_data) {
            std::string image_path = DATA_DIR + '/' + test_data.image;
            auto image = load_image(image_path, use_tiling, data.input_res);
            auto result = model.infer(image);

            EXPECT_EQ(format_test_output_to_string(model, result), test_data.reference[0]);
        }
    } else if (data.type == "MaskRCNNModel") {
        auto model = InstanceSegmentation::load(model_path, {{"tiling", use_tiling}});

        for (auto& test_data : data.test_data) {
            std::string image_path = DATA_DIR + '/' + test_data.image;
            auto image = load_image(image_path, use_tiling, data.input_res);
            auto result = model.infer(image);

            EXPECT_EQ(format_test_output_to_string(model, result), test_data.reference[0]);
        }
    } else if (data.type == "ClassificationModel") {
        auto model = Classification::load(model_path);
        for (auto& test_data : data.test_data) {
            std::string image_path = DATA_DIR + '/' + test_data.image;
            auto image = load_image(image_path, use_tiling, data.input_res);
            auto result = model.infer(image);
            EXPECT_EQ(std::string{result}, test_data.reference[0]);
        }
    } else if (data.type == "AnomalyDetection") {
        auto model = Anomaly::load(model_path);

        for (auto& test_data : data.test_data) {
            std::string image_path = DATA_DIR + '/' + test_data.image;
            auto image = load_image(image_path, use_tiling, data.input_res);
            auto result = model.infer(image);

            EXPECT_EQ(std::string{result}, test_data.reference[0]);
        }
    } else {
        FAIL() << "No implementation for model type " << data.type;
    }
}

TEST_P(ModelParameterizedTest, SerializedAccuracyTest) {
    auto data = GetParam();

    const std::string& basename = data.name.substr(data.name.find_last_of("/\\") + 1);
    auto model_path = DATA_DIR + "/serialized/" + basename;
    auto use_tiling = !data.input_res.empty();
    if (data.type == "DetectionModel") {
        auto model = DetectionModel::load(model_path, {{"tiling", use_tiling}});
        for (auto& test_data : data.test_data) {
            std::string image_path = DATA_DIR + '/' + test_data.image;
            auto image = load_image(image_path, use_tiling, data.input_res);
            auto result = model.infer(image);
            EXPECT_EQ(std::string{result}, test_data.reference[0]);
        }
    } else if (data.type == "SegmentationModel") {
        auto model = SemanticSegmentation::load(model_path, {{"tiling", use_tiling}});

        for (auto& test_data : data.test_data) {
            std::string image_path = DATA_DIR + '/' + test_data.image;
            auto image = load_image(image_path, use_tiling, data.input_res);
            auto result = model.infer(image);

            EXPECT_EQ(format_test_output_to_string(model, result), test_data.reference[0]);
        }
    } else if (data.type == "MaskRCNNModel") {
        auto model = InstanceSegmentation::load(model_path, {{"tiling", use_tiling}});

        for (auto& test_data : data.test_data) {
            std::string image_path = DATA_DIR + '/' + test_data.image;
            auto image = load_image(image_path, use_tiling, data.input_res);
            auto result = model.infer(image);

            EXPECT_EQ(format_test_output_to_string(model, result), test_data.reference[0]);
        }
    } else if (data.type == "ClassificationModel") {
        auto model = Classification::load(model_path);
        for (auto& test_data : data.test_data) {
            std::string image_path = DATA_DIR + '/' + test_data.image;
            auto image = load_image(image_path, use_tiling, data.input_res);
            auto result = model.infer(image);

            EXPECT_EQ(std::string{result}, test_data.reference[0]);
        }
    } else if (data.type == "AnomalyDetection") {
        auto model = Anomaly::load(model_path);
        for (auto& test_data : data.test_data) {
            std::string image_path = DATA_DIR + '/' + test_data.image;
            auto image = load_image(image_path, use_tiling, data.input_res);
            auto result = model.infer(image);

            EXPECT_EQ(std::string{result}, test_data.reference[0]);
        }
    } else {
        FAIL() << "No implementation for model type " << data.type;
    }
}

TEST_P(ModelParameterizedTest, AccuracyTestBatch) {
    auto data = GetParam();

    const std::string& basename = data.name.substr(data.name.find_last_of("/\\") + 1);
    auto model_path = DATA_DIR + "/serialized/" + basename;

    auto use_tiling = !data.input_res.empty();
    if (data.type == "DetectionModel") {
        auto model = DetectionModel::load(model_path, {{"tiling", use_tiling}});

        for (auto& test_data : data.test_data) {
            std::string image_path = DATA_DIR + '/' + test_data.image;
            auto image = load_image(image_path, use_tiling, data.input_res);
            auto result = model.inferBatch({image});
            ASSERT_EQ(result.size(), 1);
            EXPECT_EQ(std::string{result[0]}, test_data.reference[0]);
        }
    } else if (data.type == "SegmentationModel") {
        auto model = SemanticSegmentation::load(model_path, {{"tiling", use_tiling}});

        for (auto& test_data : data.test_data) {
            std::string image_path = DATA_DIR + '/' + test_data.image;
            auto image = load_image(image_path, use_tiling, data.input_res);
            auto result = model.inferBatch({image});

            EXPECT_EQ(format_test_output_to_string(model, result[0]), test_data.reference[0]);
        }
    } else if (data.type == "MaskRCNNModel") {
        auto model = InstanceSegmentation::load(model_path, {{"tiling", use_tiling}});

        for (auto& test_data : data.test_data) {
            std::string image_path = DATA_DIR + '/' + test_data.image;
            auto image = load_image(image_path, use_tiling, data.input_res);
            auto result = model.inferBatch({image});

            ASSERT_EQ(result.size(), 1);
            EXPECT_EQ(format_test_output_to_string(model, result[0]), test_data.reference[0]);
        }
    } else if (data.type == "ClassificationModel") {
        auto model = Classification::load(model_path);

        for (auto& test_data : data.test_data) {
            std::string image_path = DATA_DIR + '/' + test_data.image;
            auto image = load_image(image_path, use_tiling, data.input_res);
            auto result = model.inferBatch({image});

            ASSERT_EQ(result.size(), 1);
            EXPECT_EQ(std::string{result[0]}, test_data.reference[0]);
        }
    } else if (data.type == "AnomalyDetection") {
        auto model = Anomaly::load(model_path);

        for (auto& test_data : data.test_data) {
            std::string image_path = DATA_DIR + '/' + test_data.image;
            auto image = load_image(image_path, use_tiling, data.input_res);
            auto result = model.inferBatch({image});

            ASSERT_EQ(result.size(), 1);
            EXPECT_EQ(std::string{result[0]}, test_data.reference[0]);
        }
    } else {
        FAIL() << "No implementation for model type " << data.type;
    }
}

std::vector<ModelData> GetTestData(const std::string& path) {
    std::ifstream input(path);
    nlohmann::json j;
    input >> j;
    return j;
}

INSTANTIATE_TEST_SUITE_P(TestAccuracy,
                         ModelParameterizedTest,
                         testing::ValuesIn(GetTestData(PUBLIC_SCOPE_PATH)),
                         [](const ::testing::TestParamInfo<ModelData>& info) {
                             return std::to_string(info.index) + "_" + info.param.type;  // So test name will be "case1"
                         });

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
