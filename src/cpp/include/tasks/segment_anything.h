#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>

#include "utils/config.h"
#include "utils/preprocessing.h"
#include "adapters/inference_adapter.h"
#include "utils/vision_pipeline.h"
#include "tasks/segment_anything/mask_predictor.h"

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
        auto model_config = encoder_adapter->getModelConfig();
        input_shape.width = utils::get_from_any_maps("orig_width", user_config, model_config, input_shape.width);
        input_shape.height = utils::get_from_any_maps("orig_height", user_config, model_config, input_shape.width);

    }

    cv::Size input_shape;

    static SegmentAnything create_model(const std::string& encoder_model_path,
                                        const std::string& predictor_model_path,
                                        const ov::AnyMap& user_config = {},
                                        bool preload = true,
                                        const std::string& device = "AUTO");


    void static serialize(std::shared_ptr<ov::Model>& ov_model);

    std::map<std::string, ov::Tensor> preprocess(cv::Mat);
    MaskPredictor postprocess(InferenceResult& infResult);
    MaskPredictor infer(cv::Mat image);
private:
    utils::RESIZE_MODE resize_mode = utils::RESIZE_KEEP_ASPECT;
};
