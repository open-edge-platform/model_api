/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>

inline float clamp_and_round(float val, float min, float max) {
    return std::round(std::max(min, std::min(max, val)));
};

template <typename T, std::size_t N>
constexpr std::size_t arraySize(const T (&)[N]) noexcept {
    return N;
}

template <typename T>
T clamp(T value, T low, T high) {
    return value < low ? low : (value > high ? high : value);
}
