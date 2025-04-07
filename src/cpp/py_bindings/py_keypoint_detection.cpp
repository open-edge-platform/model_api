/*
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "models/keypoint_detection.h"
#include "models/results.h"
#include "py_utils.hpp"

namespace pyutils = vision::nanobind::utils;

void init_keypoint_detection(nb::module_& m) {
    nb::class_<KeypointDetectionModel, ImageModel>(m, "KeypointDetectionModel")
        .def_static(
            "create_model",
            [](const std::string& model_path,
               const std::map<std::string, nb::object>& configuration,
               bool preload,
               const std::string& device) {
                auto ov_any_config = ov::AnyMap();
                for (const auto& item : configuration) {
                    ov_any_config[item.first] = pyutils::py_object_to_any(item.second, item.first);
                }

                return KeypointDetectionModel::create_model(model_path, ov_any_config, preload, device);
            },
            nb::arg("model_path"),
            nb::arg("configuration") = ov::AnyMap({}),
            nb::arg("preload") = true,
            nb::arg("device") = "AUTO")

        .def("__call__",
             [](KeypointDetectionModel& self, const nb::ndarray<>& input) {
                 return self.infer(pyutils::wrap_np_mat(input));
             })
        .def("infer_batch",
             [](KeypointDetectionModel& self, const std::vector<nb::ndarray<>> inputs) {
                 std::vector<ImageInputData> input_mats;
                 input_mats.reserve(inputs.size());

                 for (const auto& input : inputs) {
                     input_mats.push_back(pyutils::wrap_np_mat(input));
                 }

                 return self.inferBatch(input_mats);
             })
        .def_prop_ro_static("__model__", [](nb::object) {
            return KeypointDetectionModel::ModelType;
        });

    nb::class_<KeypointDetectionResult, ResultBase>(m, "KeypointDetectionResult")
        .def(nb::init<int64_t, std::shared_ptr<MetaData>>(), nb::arg("frameId") = -1, nb::arg("metaData") = nullptr)
        .def_prop_ro(
            "keypoints",
            [](const KeypointDetectionResult& result) {
                if (!result.poses.empty()) {
                    return nb::ndarray<float, nb::numpy, nb::c_contig>(
                        const_cast<void*>(static_cast<const void*>(result.poses[0].keypoints.data())),
                        {result.poses[0].keypoints.size(), 2});
                }
                return nb::ndarray<float, nb::numpy, nb::c_contig>();
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "scores",
            [](const KeypointDetectionResult& result) {
                if (!result.poses.empty()) {
                    return nb::ndarray<float, nb::numpy, nb::c_contig>(
                        const_cast<void*>(static_cast<const void*>(result.poses[0].scores.data())),
                        {result.poses[0].scores.size()});
                }
                return nb::ndarray<float, nb::numpy, nb::c_contig>();
            },
            nb::rv_policy::reference_internal);
}
