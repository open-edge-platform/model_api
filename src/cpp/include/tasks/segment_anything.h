#ifndef VISUAL_PROMPTING_H_
#define VISUAL_PROMPTING_H_

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>

#include "utils/config.h"
#include "utils/preprocessing.h"
#include "adapters/inference_adapter.h"
#include "utils/vision_pipeline.h"

class MaskPredictor {
public:
    std::shared_ptr<InferenceAdapter> adapter;
    ov::Tensor image_encodings;
    cv::Size inputImageSize;
    utils::RESIZE_MODE resize_mode;

    MaskPredictor() {}
    MaskPredictor(std::shared_ptr<InferenceAdapter> adapter, ov::Tensor image_encodings, cv::Size inputImageSize, utils::RESIZE_MODE resize_mode):
        adapter(adapter), image_encodings(image_encodings), inputImageSize(inputImageSize), resize_mode(resize_mode) {

        cv::Size input_size(1024, 1024); //TODO: Remove this hardcoded input
        float scaleX = input_size.width / (float)inputImageSize.width;
        float scaleY = input_size.height / (float)inputImageSize.height;
        float s = std::min(scaleX, scaleY);
        float sx = s;
        float sy = s;

        resize_transform = {
            sx, 0, 0,
            0, sy, 0,
            0, 0, 1,
        };
    }

    std::vector<cv::Mat> infer(std::vector<cv::Point> positive, std::vector<cv::Point> negative = {});

    cv::Matx33f resize_transform;

    cv::Point transform(cv::Point);
};

class SegmentAnything {
public:

    VisionPipeline<MaskPredictor> pipeline;

    std::shared_ptr<InferenceAdapter> encoder_adapter;
    std::shared_ptr<InferenceAdapter> predictor_adapter;

    SegmentAnything(std::shared_ptr<InferenceAdapter> encoder_adapter, std::shared_ptr<InferenceAdapter> predictor_adapter, const ov::AnyMap& user_config) : encoder_adapter(encoder_adapter), predictor_adapter(predictor_adapter) {
        pipeline = VisionPipeline<MaskPredictor>(
            encoder_adapter,
            [&](cv::Mat image) {
                return preprocess(image);
            },
            [&](InferenceResult result) {
                return postprocess(result);
            }
        );
    }

    static SegmentAnything create_model(const std::string& encoder_model_path,
                                        const std::string& predictor_model_path,
                                        const ov::AnyMap& user_config = {},
                                        bool preload = true,
                                        const std::string& device = "AUTO");


    void static serialize(std::shared_ptr<ov::Model>& ov_model);

    std::map<std::string, ov::Tensor> preprocess(cv::Mat);
    MaskPredictor postprocess(InferenceResult& infResult);
    MaskPredictor infer(cv::Mat image);
};


#endif // VISUAL_PROMPTING_H_
