#include "tasks/segment_anything.h"
#include <openvino/core/model.hpp>
#include "adapters/openvino_adapter.h"

#include "utils/preprocessing.h"
#include "utils/tensor.h"
#include "utils/tiling.h"

using utils::RESIZE_KEEP_ASPECT;

std::vector<cv::Mat> MaskPredictor::infer(std::vector<cv::Point> positive, std::vector<cv::Point> negative) {
    const std::string image_embeddings_tensor_name = "image_embeddings";
    const std::string point_coords_tensor_name = "point_coords";
    const std::string point_labels_tensor_name = "point_labels";
    const std::string orig_image_tensor_name = "orig_im_size";

    std::map<std::string, ov::Tensor> tensors;


    tensors[image_embeddings_tensor_name] = image_encodings;

    std::vector<float> point_coord_data;
    std::vector<float> point_label_data;
    point_coord_data.reserve(positive.size() * 2);
    point_label_data.reserve(positive.size());

    for (size_t i = 0; i < positive.size(); i++) {
        auto transformed = transform(positive[i]);
        point_coord_data.push_back(transformed.x);
        point_coord_data.push_back(transformed.y);
        point_label_data.push_back(1);
    }
    std::vector<float> orig_image_size = {(float)inputImageSize.height, (float)inputImageSize.width};
    std::vector<float> mask_input(256 * 256);


    tensors[point_coords_tensor_name] = ov::Tensor(ov::element::f32, ov::Shape({positive.size(), 1, 2}), point_coord_data.data());
    tensors[point_labels_tensor_name] = ov::Tensor(ov::element::f32, ov::Shape({positive.size(), 1}), point_label_data.data());
    tensors[orig_image_tensor_name] = ov::Tensor(ov::element::f32, ov::Shape({2}), orig_image_size.data());
    std::vector<float> has_mask_input = {0};
    tensors["has_mask_input"] = ov::Tensor(ov::element::f32, ov::Shape({1}), has_mask_input.data());
    tensors["mask_input"] = ov::Tensor(ov::element::f32, ov::Shape({1, 1, 256, 256}), mask_input.data());


    InferenceResult result;
    result.data = adapter->infer(tensors);
    auto tensorName = adapter->getOutputNames().front();
    auto predictions = result.data[tensorName];

    float* data = predictions.data<float>();

    // Step 2: Get tensor shape
    ov::Shape shape = predictions.get_shape(); // [1, 4, 2048, 1365]
    size_t num_masks = shape[1];
    size_t height = shape[2];
    size_t width = shape[3];

    // Step 3: Convert each mask to cv::Mat
    std::vector<cv::Mat> mask_mats;
    for (size_t i = 0; i < num_masks; ++i) {
        // Calculate offset into the data array
        size_t offset = i * height * width;

        // Wrap the raw data into a cv::Mat (no data copy)
        cv::Mat mask(height, width, CV_32F, data + offset);

        // Optional: Clone if you need to store it independently
        mask_mats.push_back(mask.clone());
    }
    return mask_mats;
}

cv::Point MaskPredictor::transform(cv::Point input) {
    cv::Vec3f vector(input.x, input.y, 1);
    auto scaled = resize_transform * vector;
    return cv::Point(scaled[0], scaled[1]);
}

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
    //predictor_adapter->applyModelTransform([](std::shared_ptr<ov::Model>& model) {
    //    ov::set_batch(model, 1);
    //});

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
    utils::RESIZE_MODE resize_mode = RESIZE_KEEP_ASPECT;
    uint8_t pad_value = 0;
    bool reverse_input_channels = false;

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

    return MaskPredictor(predictor_adapter, std::move(infResult.data[tensorName]), infResult.inputImageSize, utils::RESIZE_KEEP_ASPECT);
}
