#ifndef VISUAL_PROMPTING_H_
#define VISUAL_PROMPTING_H_

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>

#include "utils/config.h"
#include "adapters/inference_adapter.h"
#include "utils/vision_pipeline.h"

class MaskPredictor {
public:
    std::shared_ptr<InferenceAdapter> adapter;
    ov::Tensor image_encodings;
    cv::Size inputImageSize;

    MaskPredictor() {}
    MaskPredictor(std::shared_ptr<InferenceAdapter> adapter, ov::Tensor image_encodings, cv::Size inputImageSize): adapter(adapter), image_encodings(image_encodings), inputImageSize(inputImageSize) {

    }

    void infer(std::vector<cv::Point> positive, std::vector<cv::Point> negative = {});
};

class VisualPrompting {
public:

    VisionPipeline<MaskPredictor> pipeline;

    std::shared_ptr<InferenceAdapter> encoder_adapter;
    std::shared_ptr<InferenceAdapter> predictor_adapter;

    VisualPrompting(std::shared_ptr<InferenceAdapter> encoder_adapter, std::shared_ptr<InferenceAdapter> predictor_adapter, const ov::AnyMap& user_config) : encoder_adapter(encoder_adapter), predictor_adapter(predictor_adapter) {
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

    static VisualPrompting create_model(const std::string& encoder_model_path,
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
