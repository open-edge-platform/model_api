/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tasks/semantic_segmentation.h"

#include <opencv2/core.hpp>

#include "adapters/openvino_adapter.h"
#include "utils/config.h"
#include "utils/tensor.h"

namespace {
constexpr char feature_vector_name[]{"feature_vector"};
cv::Mat get_activation_map(const cv::Mat& features) {
    double min_soft_score, max_soft_score;
    cv::minMaxLoc(features, &min_soft_score, &max_soft_score);
    double factor = 255.0 / (max_soft_score - min_soft_score + 1e-12);

    cv::Mat int_act_map;
    features.convertTo(int_act_map, CV_8U, factor, -min_soft_score * factor);
    return int_act_map;
}

void normalize_soft_prediction(cv::Mat& soft_prediction, const cv::Mat& normalize_factor) {
    float* data = soft_prediction.ptr<float>(0);
    const int num_classes = soft_prediction.channels();
    const size_t step_rows = soft_prediction.step[0] / sizeof(float);
    const size_t step_cols = soft_prediction.step[1] / sizeof(float);

    for (int y = 0; y < soft_prediction.rows; ++y) {
        for (int x = 0; x < soft_prediction.cols; ++x) {
            int weight = normalize_factor.at<int>(y, x);
            if (weight > 0) {
                for (int c = 0; c < num_classes; ++c) {
                    data[y * step_rows + x * step_cols + c] /= weight;
                }
            }
        }
    }
}
}  // namespace

SemanticSegmentation SemanticSegmentation::load(const std::string& model_path, const ov::AnyMap& configuration) {
    auto core = ov::Core();
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    if (model->has_rt_info("model_info", "model_type")) {
        std::cout << "has model type in info: " << model->get_rt_info<std::string>("model_info", "model_type")
                  << std::endl;
    } else {
        throw std::runtime_error("Incorrect or unsupported model_type");
    }

    if (utils::model_has_embedded_processing(model)) {
        std::cout << "model already was serialized" << std::endl;
    } else {
        SemanticSegmentation::serialize(model);
    }
    auto adapter = std::make_shared<OpenVINOInferenceAdapter>();
    adapter->loadModel(model, core, "AUTO");
    return SemanticSegmentation(adapter, configuration);
}

void SemanticSegmentation::serialize(std::shared_ptr<ov::Model>& ov_model) {
    if (ov_model->inputs().size() != 1) {
        throw std::logic_error("Segmentation model wrapper supports topologies with only 1 input");
    }
    const auto& input = ov_model->input();
    // inputNames.push_back(input.get_any_name());

    auto layout = ov::layout::get_layout(input);
    if (layout.empty()) {
        layout = utils::getLayoutFromShape(input.get_partial_shape());
    }
    const ov::Shape& shape = input.get_partial_shape().get_max_shape();
    if (shape.size() != 4 || shape[ov::layout::channels_idx(layout)] != 3) {
        throw std::logic_error("3-channel 4-dimensional model's input is expected");
    }
    if (ov_model->outputs().size() > 2) {
        throw std::logic_error("Segmentation model wrapper supports topologies with 1 or 2 outputs");
    }

    std::string out_name;
    for (ov::Output<ov::Node>& output : ov_model->outputs()) {
        const std::unordered_set<std::string>& out_names = output.get_names();
        if (out_names.find(feature_vector_name) == out_names.end()) {
            if (out_name.empty()) {
                out_name = output.get_any_name();
            } else {
                throw std::runtime_error(std::string{"Only "} + feature_vector_name +
                                         " and 1 other output are allowed");
            }
        }
    }
    if (out_name.empty()) {
        throw std::runtime_error("No output containing segmentation masks found");
    }

    auto interpolation_mode = cv::INTER_LINEAR;
    utils::RESIZE_MODE resize_mode = utils::RESIZE_FILL;
    uint8_t pad_value = 0;
    bool reverse_input_channels = false;

    auto config = ov_model->has_rt_info("model_info") ? ov_model->get_rt_info<ov::AnyMap>("model_info") : ov::AnyMap{};

    std::vector<float> scale_values;
    std::vector<float> mean_values;
    scale_values = utils::get_from_any_maps("scale_values", config, ov::AnyMap{}, scale_values);
    mean_values = utils::get_from_any_maps("mean_values", config, ov::AnyMap{}, mean_values);

    ov_model =
        utils::embedProcessing(ov_model,
                               input.get_any_name(),
                               layout,
                               resize_mode,
                               interpolation_mode,
                               ov::Shape{shape[ov::layout::width_idx(layout)], shape[ov::layout::height_idx(layout)]},
                               pad_value,
                               reverse_input_channels,
                               mean_values,
                               scale_values);

    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(ov_model);
    ov::Layout out_layout = utils::getLayoutFromShape(ov_model->output(out_name).get_partial_shape());
    ppp.output(out_name).model().set_layout(out_layout);
    ppp.output(out_name).tensor().set_element_type(ov::element::f32);
    if (ov::layout::has_channels(out_layout)) {
        ppp.output(out_name).tensor().set_layout("NCHW");
    } else {
        // deeplabv3
        ppp.output(out_name).tensor().set_layout("NHW");
    }
    ov_model = ppp.build();

    cv::Size input_shape(shape[ov::layout::width_idx(layout)], shape[ov::layout::height_idx(layout)]);
    ov_model->set_rt_info(input_shape.width, "model_info", "orig_width");
    ov_model->set_rt_info(input_shape.height, "model_info", "orig_height");
}

std::map<std::string, ov::Tensor> SemanticSegmentation::preprocess(cv::Mat image) {
    std::map<std::string, ov::Tensor> input = {};
    input.emplace(adapter->getInputNames()[0], utils::wrapMat2Tensor(image));
    return input;
}

SemanticSegmentationResult SemanticSegmentation::postprocess(InferenceResult& infResult) {
    auto outputNames = adapter->getOutputNames();
    const auto& outputName = outputNames[0] == feature_vector_name ? outputNames[1] : outputNames[0];
    const auto& outTensor = infResult.data[outputName];
    const ov::Shape& outputShape = outTensor.get_shape();
    const ov::Layout& outputLayout = utils::getLayoutFromShape(outputShape);
    size_t outChannels =
        ov::layout::has_channels(outputLayout) ? outputShape[ov::layout::channels_idx(outputLayout)] : 1;
    int outHeight = static_cast<int>(outputShape[ov::layout::height_idx(outputLayout)]);
    int outWidth = static_cast<int>(outputShape[ov::layout::width_idx(outputLayout)]);
    cv::Mat soft_prediction;
    if (outChannels == 1 && outTensor.get_element_type() == ov::element::i32) {
        cv::Mat predictions(outHeight, outWidth, CV_32SC1, outTensor.data<int32_t>());
        predictions.convertTo(soft_prediction, CV_8UC1);
    } else if (outChannels == 1 && outTensor.get_element_type() == ov::element::i64) {
        cv::Mat predictions(outHeight, outWidth, CV_32SC1);
        const auto data = outTensor.data<int64_t>();
        for (size_t i = 0; i < predictions.total(); ++i) {
            reinterpret_cast<int32_t*>(predictions.data)[i] = int32_t(data[i]);
        }
        predictions.convertTo(soft_prediction, CV_8UC1);
    } else if (outTensor.get_element_type() == ov::element::f32) {
        float* data = outTensor.data<float>();
        std::vector<cv::Mat> channels;
        for (size_t c = 0; c < outTensor.get_shape()[1]; ++c) {
            channels.emplace_back(cv::Size{outWidth, outHeight}, CV_32FC1, data + c * outHeight * outWidth);
        }
        cv::merge(channels, soft_prediction);
    }

    cv::Mat hard_prediction =
        create_hard_prediction_from_soft_prediction(soft_prediction, soft_threshold, blur_strength);

    cv::resize(hard_prediction, hard_prediction, infResult.inputImageSize, 0.0, 0.0, cv::INTER_NEAREST);

    SemanticSegmentationResult result;
    result.resultImage = hard_prediction;
    if (return_soft_prediction) {
        cv::resize(soft_prediction, soft_prediction, infResult.inputImageSize, 0.0, 0.0, cv::INTER_NEAREST);
        result.soft_prediction = soft_prediction;
        auto iter = infResult.data.find(feature_vector_name);
        if (infResult.data.end() != iter) {
            result.saliency_map = get_activation_map(soft_prediction);
            result.feature_vector = iter->second;
        }
    }

    return result;
}

std::vector<Contour> SemanticSegmentation::getContours(const SemanticSegmentationResult& result) {
    if (result.soft_prediction.empty()) {
        throw std::runtime_error{"Cannot get contours from semantic segmentation result without soft prediction"};
    }
    if (result.soft_prediction.channels() == 1) {
        throw std::runtime_error{"Cannot get contours from soft prediction with 1 layer"};
    }

    std::vector<Contour> combined_contours = {};
    cv::Mat label_index_map;
    cv::Mat current_label_soft_prediction;
    for (int index = 1; index < result.soft_prediction.channels(); index++) {
        cv::extractChannel(result.soft_prediction, current_label_soft_prediction, index);
        cv::inRange(result.resultImage,
                    cv::Scalar(index, index, index),
                    cv::Scalar(index, index, index),
                    label_index_map);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(label_index_map, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        std::string label = getLabelName(index - 1);

        for (unsigned int i = 0; i < contours.size(); i++) {
            cv::Mat mask = cv::Mat::zeros(result.resultImage.rows, result.resultImage.cols, result.resultImage.type());
            cv::drawContours(mask, contours, i, 255, -1);
            float probability = (float)cv::mean(current_label_soft_prediction, mask)[0];
            combined_contours.push_back({label, probability, contours[i]});
        }
    }

    return combined_contours;
}

SemanticSegmentationResult SemanticSegmentation::infer(cv::Mat image) {
    return pipeline->infer(image);
}

std::vector<SemanticSegmentationResult> SemanticSegmentation::inferBatch(std::vector<cv::Mat> images) {
    return pipeline->inferBatch(images);
}

cv::Mat SemanticSegmentation::create_hard_prediction_from_soft_prediction(cv::Mat soft_prediction,
                                                                          float threshold,
                                                                          int blur_strength) {
    if (soft_prediction.channels() == 1) {
        return soft_prediction;
    }

    cv::Mat soft_prediction_blurred = soft_prediction.clone();

    bool applyBlurAndSoftThreshold = (blur_strength > -1 && soft_threshold < std::numeric_limits<float>::infinity());
    if (applyBlurAndSoftThreshold) {
        cv::blur(soft_prediction_blurred, soft_prediction_blurred, cv::Size{blur_strength, blur_strength});
    }

    cv::Mat hard_prediction{cv::Size{soft_prediction_blurred.cols, soft_prediction_blurred.rows}, CV_8UC1};
    for (int i = 0; i < soft_prediction_blurred.rows; ++i) {
        for (int j = 0; j < soft_prediction_blurred.cols; ++j) {
            float max_prob = -std::numeric_limits<float>::infinity();
            if (applyBlurAndSoftThreshold) {
                max_prob = soft_threshold;
            }

            uint8_t max_id = 0;
            for (int c = 0; c < soft_prediction_blurred.channels(); ++c) {
                float prob = ((float*)soft_prediction_blurred.ptr(i, j))[c];
                if (prob > max_prob) {
                    max_prob = prob;
                    max_id = c;
                }
            }
            hard_prediction.at<uint8_t>(i, j) = max_id;
        }
    }
    return hard_prediction;
}

SemanticSegmentationResult SemanticSegmentation::postprocess_tile(SemanticSegmentationResult tile, const cv::Rect&) {
    return tile;
}

SemanticSegmentationResult SemanticSegmentation::merge_tiling_results(
    const std::vector<SemanticSegmentationResult>& tiles_results,
    const cv::Size& image_size,
    const std::vector<cv::Rect>& tile_coords,
    const utils::TilingInfo& tiling_info) {
    auto first = tiles_results.front();
    cv::Mat voting_mask(cv::Size(image_size.width, image_size.height), CV_32SC1, cv::Scalar(0));
    cv::Mat merged_soft_prediction(cv::Size(image_size.width, image_size.height),
                                   CV_32FC(first.soft_prediction.channels()),
                                   cv::Scalar(0));

    for (size_t i = 0; i < tiles_results.size(); ++i) {
        voting_mask(tile_coords[i]) += 1;
        merged_soft_prediction(tile_coords[i]) += tiles_results[i].soft_prediction;
    }

    normalize_soft_prediction(merged_soft_prediction, voting_mask);

    SemanticSegmentationResult result;
    result.resultImage =
        create_hard_prediction_from_soft_prediction(merged_soft_prediction, soft_threshold, blur_strength);
    ;
    if (return_soft_prediction) {
        result.soft_prediction = merged_soft_prediction;
    }
    return result;
}
