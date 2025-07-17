#include "tasks/segment_anything.h"
#include <openvino/core/model.hpp>
#include "adapters/openvino_adapter.h"

#include "utils/preprocessing.h"
#include "utils/tensor.h"
#include "utils/tiling.h"

SegmentAnything SegmentAnything::create_model(const std::string& encoder_model_path,
    const std::string& predictor_model_path,
    const ov::AnyMap& user_config,
    bool preload,
    const std::string& device) {


    auto encoder_adapter = std::make_shared<OpenVINOInferenceAdapter>();
    auto predictor_adapter = std::make_shared<OpenVINOInferenceAdapter>();
    encoder_adapter->loadModel(encoder_model_path, device, user_config, false);
    encoder_adapter->applyModelTransform(SegmentAnything::serialize);

    if (preload) {
        encoder_adapter->compileModel(device, user_config);
    }

    predictor_adapter->loadModel(predictor_model_path, device, user_config, true);
    return SegmentAnything(encoder_adapter, predictor_adapter, user_config);
}

void SegmentAnything::serialize(std::shared_ptr<ov::Model>& ov_model) {
    if (utils::model_has_embedded_processing(ov_model)) {
        return;
    }

    auto input = ov_model->inputs().front();

    auto layout = ov::layout::get_layout(input);
    if (layout.empty()) {
        layout = utils::getLayoutFromShape(input.get_partial_shape());
    }


    const ov::Shape& shape = input.get_partial_shape().get_max_shape();

    auto interpolation_mode = cv::INTER_LINEAR;
    uint8_t pad_value = 0;
    bool reverse_input_channels = false;

    utils::RESIZE_MODE resize_mode = utils::RESIZE_KEEP_ASPECT;
    std::vector<float> scale_values = {58.395, 57.12, 57.375};
    std::vector<float> mean_values = {123.675, 116.28, 103.53};

    if (ov_model->has_rt_info("model_info")) {
        std::cout << "getting config from model config" << std::endl;
        auto config = ov_model->get_rt_info<ov::AnyMap>("model_info");
        reverse_input_channels =
            utils::get_from_any_maps("reverse_input_channels", config, ov::AnyMap{}, reverse_input_channels);
        scale_values = utils::get_from_any_maps("scale_values", config, ov::AnyMap{}, scale_values);
        mean_values = utils::get_from_any_maps("mean_values", config, ov::AnyMap{}, mean_values);
    }


    auto input_shape = ov::Shape{shape[ov::layout::width_idx(layout)], shape[ov::layout::height_idx(layout)]};

    ov_model = utils::embedProcessing(ov_model,
                                      input.get_any_name(),
                                      layout,
                                      resize_mode,
                                      interpolation_mode,
                                      input_shape,
                                      pad_value,
                                      reverse_input_channels,
                                      mean_values,
                                      scale_values);
    ov_model->set_rt_info(true, "model_info", "embedded_processing");
    ov_model->set_rt_info(input_shape[0], "model_info", "orig_width");
    ov_model->set_rt_info(input_shape[1], "model_info", "orig_height");
}


MaskPredictor  SegmentAnything::infer(cv::Mat image) {
    return pipeline.infer(image);
}

std::map<std::string, ov::Tensor> SegmentAnything::preprocess(cv::Mat image) {
    std::map<std::string, ov::Tensor> input = {};
    input.emplace(encoder_adapter->getInputNames()[0], utils::wrapMat2Tensor(image));
    return input;
}

MaskPredictor SegmentAnything::postprocess(InferenceResult& infResult) {
    auto tensorName = encoder_adapter->getOutputNames().front();
    return MaskPredictor(predictor_adapter, std::move(infResult.data[tensorName]), infResult.inputImageSize, input_shape, resize_mode);
}
