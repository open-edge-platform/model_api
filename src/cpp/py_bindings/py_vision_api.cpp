/*
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nanobind/nanobind.h>

namespace nb = nanobind;

void init_base_modules(nb::module_& m);
void init_classification(nb::module_& m);
void init_segmentation(nb::module_& m);
void init_instance_segmentation(nb::module_& m);
void init_keypoint_detection(nb::module_& m);
void init_anomaly_detection(nb::module_& m);
void init_detection(nb::module_& m);

NB_MODULE(py_model_api, m) {
    m.doc() = "Nanobind binding for OpenVINO Vision API library";
    init_base_modules(m);
    init_classification(m);
    init_keypoint_detection(m);
    init_segmentation(m);
    init_instance_segmentation(m);
    init_anomaly_detection(m);
    init_detection(m);
}
