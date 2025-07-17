/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "utils/config.h"

namespace utils {

std::shared_ptr<ov::Model> embedProcessing(std::shared_ptr<ov::Model>& model,
                                           const std::string& inputName,
                                           const ov::Layout&,
                                           RESIZE_MODE resize_mode,
                                           const cv::InterpolationFlags interpolationMode,
                                           const ov::Shape& targetShape,
                                           uint8_t pad_value,
                                           bool brg2rgb,
                                           const std::vector<float>& mean,
                                           const std::vector<float>& scale,
                                           const std::type_info& dtype = typeid(int));

ov::preprocess::PostProcessSteps::CustomPostprocessOp createResizeGraph(RESIZE_MODE resizeMode,
                                                                        const ov::Shape& size,
                                                                        const cv::InterpolationFlags interpolationMode,
                                                                        uint8_t pad_value);

ov::Output<ov::Node> resizeImageGraph(const ov::Output<ov::Node>& input,
                                      const ov::Shape& size,
                                      bool keep_aspect_ratio,
                                      const cv::InterpolationFlags interpolationMode,
                                      uint8_t pad_value);

ov::Output<ov::Node> fitToWindowLetterBoxGraph(const ov::Output<ov::Node>& input,
                                               const ov::Shape& size,
                                               const cv::InterpolationFlags interpolationMode,
                                               uint8_t pad_value);

ov::Output<ov::Node> cropResizeGraph(const ov::Output<ov::Node>& input,
                                     const ov::Shape& size,
                                     const cv::InterpolationFlags interpolationMode);

}  // namespace utils
