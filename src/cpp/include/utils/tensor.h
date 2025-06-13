/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace utils {
static inline ov::Tensor wrapMat2Tensor(const cv::Mat& mat) {
    auto matType = mat.type() & CV_MAT_DEPTH_MASK;
    if (matType != CV_8U && matType != CV_32F) {
        throw std::runtime_error("Unsupported mat type for wrapping");
    }
    bool isMatFloat = matType == CV_32F;

    const size_t channels = mat.channels();
    const size_t height = mat.rows;
    const size_t width = mat.cols;

    const size_t strideH = mat.step.buf[0];
    const size_t strideW = mat.step.buf[1];

    const bool isDense = !isMatFloat
                             ? (strideW == channels && strideH == channels * width)
                             : (strideW == channels * sizeof(float) && strideH == channels * width * sizeof(float));
    if (!isDense) {
        throw std::runtime_error("Doesn't support conversion from not dense cv::Mat");
    }
    auto precision = isMatFloat ? ov::element::f32 : ov::element::u8;
    struct SharedMatAllocator {
        const cv::Mat mat;
        void* allocate(size_t bytes, size_t) {
            return bytes <= mat.rows * mat.step[0] ? mat.data : nullptr;
        }
        void deallocate(void*, size_t, size_t) {}
        bool is_equal(const SharedMatAllocator& other) const noexcept {
            return this == &other;
        }
    };
    return ov::Tensor(precision, ov::Shape{1, height, width, channels}, SharedMatAllocator{mat});
}

}  // namespace utils
