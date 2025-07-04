/*
 * Copyright (C) 2024-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tasks/keypoint_detection.h"

#include "adapters/openvino_adapter.h"
#include "utils/config.h"
#include "utils/tensor.h"

namespace {

void colArgMax(const cv::Mat& src,
               cv::Mat& dst_locs,
               cv::Mat& dst_values,
               bool apply_softmax = false,
               float eps = 1e-6f) {
    dst_locs = cv::Mat::zeros(src.rows, 1, CV_32S);
    dst_values = cv::Mat::zeros(src.rows, 1, CV_32F);

    for (int row = 0; row < src.rows; ++row) {
        const float* ptr_row = src.ptr<float>(row);
        int max_val_idx = 0;
        float max_val = ptr_row[0];
        for (int col = 1; col < src.cols; ++col) {
            if (ptr_row[col] > max_val) {
                max_val_idx = col;
                dst_locs.at<int>(row) = max_val_idx;
                max_val = ptr_row[col];
            }
        }

        if (apply_softmax) {
            float sum = 0.0f;
            for (int col = 0; col < src.cols; ++col) {
                sum += exp(ptr_row[col] - max_val);
            }
            dst_values.at<float>(row) = exp(ptr_row[max_val_idx] - max_val) / (sum + eps);
        } else {
            dst_values.at<float>(row) = max_val;
        }
    }
}

KeypointDetectionResult decode_simcc(const cv::Mat& simcc_x,
                                     const cv::Mat& simcc_y,
                                     const cv::Point2f& extra_scale = cv::Point2f(1.f, 1.f),
                                     const cv::Point2i& extra_offset = cv::Point2f(0.f, 0.f),
                                     bool apply_softmax = false,
                                     float simcc_split_ratio = 2.0f,
                                     float decode_beta = 150.0f,
                                     float sigma = 6.0f) {
    cv::Mat x_locs, max_val_x;
    std::cout << cv::sum(simcc_x) << "\n";
    std::cout << cv::sum(simcc_y) << "\n";
    colArgMax(simcc_x, x_locs, max_val_x, false);

    cv::Mat y_locs, max_val_y;
    colArgMax(simcc_y, y_locs, max_val_y, false);

    if (apply_softmax) {
        cv::Mat tmp_locs;
        colArgMax(decode_beta * sigma * simcc_x, tmp_locs, max_val_x, true);
        colArgMax(decode_beta * sigma * simcc_y, tmp_locs, max_val_y, true);
    }

    std::vector<cv::Point2f> keypoints(x_locs.rows);
    cv::Mat scores = cv::Mat::zeros(x_locs.rows, 1, CV_32F);
    for (int i = 0; i < x_locs.rows; ++i) {
        keypoints[i] = cv::Point2f((x_locs.at<int>(i) - extra_offset.x) * extra_scale.x,
                                   (y_locs.at<int>(i) - extra_offset.y) * extra_scale.y) /
                       simcc_split_ratio;
        scores.at<float>(i) = std::min(max_val_x.at<float>(i), max_val_y.at<float>(i));

        if (scores.at<float>(i) <= 0.f) {
            keypoints[i] = cv::Point2f(-1.f, -1.f);
        }
    }

    return {std::move(keypoints), scores};
}

}  // namespace

KeypointDetection KeypointDetection::create_model(const std::string& model_path,
                                                  const ov::AnyMap& user_config,
                                                  bool preload,
                                                  const std::string& device) {
    auto adapter = std::make_shared<OpenVINOInferenceAdapter>();
    adapter->loadModel(model_path, device, user_config, false);

    std::string model_type;
    model_type = utils::get_from_any_maps("model_type", user_config, adapter->getModelConfig(), model_type);

    if (model_type.empty() || model_type != "keypoint_detection") {
        throw std::runtime_error("Incorrect or unsupported model_type, expected: keypoint_detection");
    }
    adapter->applyModelTransform(KeypointDetection::serialize);
    if (preload) {
        adapter->compileModel(device, user_config);
    }

    return KeypointDetection(adapter, user_config);
}

void KeypointDetection::serialize(std::shared_ptr<ov::Model>& ov_model) {
    if (utils::model_has_embedded_processing(ov_model)) {
        std::cout << "model already was serialized" << std::endl;
        return;
    }
    if (ov_model->inputs().size() != 1) {
        throw std::logic_error("KeypointDetection model wrapper supports topologies with only 1 input");
    }
    const auto& input = ov_model->input();
    auto config = ov_model->has_rt_info("model_info") ? ov_model->get_rt_info<ov::AnyMap>("model_info") : ov::AnyMap{};
    std::string layout = "";
    layout = utils::get_from_any_maps("layout", config, {}, layout);
    auto inputsLayouts = utils::parseLayoutString(layout);
    const ov::Layout& inputLayout = utils::getInputLayout(input, inputsLayouts);
    const ov::Shape& inputShape = input.get_partial_shape().get_max_shape();
    if (inputShape.size() != 4 || inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
        throw std::logic_error("3-channel 4-dimensional model's input is expected");
    }

    auto interpolation_mode = cv::INTER_LINEAR;
    utils::RESIZE_MODE resize_mode = utils::RESIZE_MODE::RESIZE_FILL;
    resize_mode = utils::get_from_any_maps("resize_type", config, ov::AnyMap{}, resize_mode);

    std::vector<float> scale_values;
    std::vector<float> mean_values;
    scale_values = utils::get_from_any_maps("scale_values", config, ov::AnyMap{}, scale_values);
    mean_values = utils::get_from_any_maps("mean_values", config, ov::AnyMap{}, mean_values);
    uint8_t pad_value = 0;
    pad_value = utils::get_from_any_maps<unsigned>("pad_value", config, ov::AnyMap{}, pad_value);
    bool reverse_input_channels = false;
    reverse_input_channels =
        utils::get_from_any_maps("reverse_input_channels", config, ov::AnyMap{}, reverse_input_channels);

    cv::Size input_shape(inputShape[ov::layout::width_idx(inputLayout)],
                         inputShape[ov::layout::height_idx(inputLayout)]);

    ov_model = utils::embedProcessing(
        ov_model,
        input.get_any_name(),
        inputLayout,
        resize_mode,
        interpolation_mode,
        ov::Shape{static_cast<size_t>(input_shape.width), static_cast<size_t>(input_shape.height)},
        pad_value,
        reverse_input_channels,
        mean_values,
        scale_values);

    // --------------------------- Check output  -----------------------------------------------------

    if (ov_model->outputs().size() != 2) {
        throw std::logic_error(std::string{"KeypointDetection model wrapper supports topologies with 2 outputs"});
    }

    ov_model->set_rt_info(true, "model_info", "embedded_processing");
    ov_model->set_rt_info(input_shape.width, "model_info", "orig_width");
    ov_model->set_rt_info(input_shape.height, "model_info", "orig_height");
}

std::map<std::string, ov::Tensor> KeypointDetection::preprocess(cv::Mat image) {
    std::map<std::string, ov::Tensor> input = {};
    input.emplace(adapter->getInputNames()[0], utils::wrapMat2Tensor(image));
    return input;
}

KeypointDetectionResult KeypointDetection::postprocess(InferenceResult& infResult) {
    auto outputNames = adapter->getOutputNames();

    const ov::Tensor& pred_x_tensor = infResult.data.find(outputNames[0])->second;
    size_t shape_offset = pred_x_tensor.get_shape().size() == 3 ? 1 : 0;
    auto pred_x_mat = cv::Mat(cv::Size(static_cast<int>(pred_x_tensor.get_shape()[shape_offset + 1]),
                                       static_cast<int>(pred_x_tensor.get_shape()[shape_offset])),
                              CV_32F,
                              pred_x_tensor.data(),
                              pred_x_tensor.get_strides()[shape_offset]);

    const ov::Tensor& pred_y_tensor = infResult.data.find(outputNames[1])->second;
    shape_offset = pred_y_tensor.get_shape().size() == 3 ? 1 : 0;
    auto pred_y_mat = cv::Mat(cv::Size(static_cast<int>(pred_y_tensor.get_shape()[shape_offset + 1]),
                                       static_cast<int>(pred_y_tensor.get_shape()[shape_offset])),
                              CV_32F,
                              pred_y_tensor.data(),
                              pred_y_tensor.get_strides()[shape_offset]);

    float inverted_scale_x = static_cast<float>(infResult.inputImageSize.width) / input_shape.width,
          inverted_scale_y = static_cast<float>(infResult.inputImageSize.height) / input_shape.height;

    int pad_left = 0, pad_top = 0;
    if (utils::RESIZE_MODE::RESIZE_KEEP_ASPECT == resize_mode ||
        utils::RESIZE_MODE::RESIZE_KEEP_ASPECT_LETTERBOX == resize_mode) {
        inverted_scale_x = inverted_scale_y = std::max(inverted_scale_x, inverted_scale_y);
        if (utils::RESIZE_MODE::RESIZE_KEEP_ASPECT_LETTERBOX == resize_mode) {
            pad_left =
                (input_shape.width -
                 static_cast<int>(std::round(static_cast<float>(infResult.inputImageSize.width) / inverted_scale_x))) /
                2;
            pad_top =
                (input_shape.height -
                 static_cast<int>(std::round(static_cast<float>(infResult.inputImageSize.height) / inverted_scale_y))) /
                2;
        }
    }

    return decode_simcc(pred_x_mat,
                        pred_y_mat,
                        {inverted_scale_x, inverted_scale_y},
                        {pad_left, pad_top},
                        apply_softmax);
}

KeypointDetectionResult KeypointDetection::infer(cv::Mat image) {
    return pipeline.infer(image);
}

std::vector<KeypointDetectionResult> KeypointDetection::inferBatch(std::vector<cv::Mat> images) {
    return pipeline.inferBatch(images);
}
