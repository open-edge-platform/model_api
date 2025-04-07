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

#include "models/anomaly_model.h"
#include "models/results.h"
#include "py_utils.hpp"

namespace pyutils = vision::nanobind::utils;

void init_anomaly_detection(nb::module_& m) {
    nb::class_<AnomalyModel, ImageModel>(m, "AnomalyDetection")
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

                return AnomalyModel::create_model(model_path, ov_any_config, preload, device);
            },
            nb::arg("model_path"),
            nb::arg("configuration") = ov::AnyMap({}),
            nb::arg("preload") = true,
            nb::arg("device") = "AUTO")

        .def("__call__",
             [](AnomalyModel& self, const nb::ndarray<>& input) {
                 return self.infer(pyutils::wrap_np_mat(input));
             })
        .def("infer_batch",
             [](AnomalyModel& self, const std::vector<nb::ndarray<>> inputs) {
                 std::vector<ImageInputData> input_mats;
                 input_mats.reserve(inputs.size());

                 for (const auto& input : inputs) {
                     input_mats.push_back(pyutils::wrap_np_mat(input));
                 }

                 return self.inferBatch(input_mats);
             })
        .def_prop_ro_static("__model__", [](nb::object) {
            return AnomalyModel::ModelType;
        });

    nb::class_<AnomalyResult, ResultBase>(m, "AnomalyResult")
        .def(nb::init<int64_t, std::shared_ptr<MetaData>>(), nb::arg("frameId") = -1, nb::arg("metaData") = nullptr)
        .def_prop_ro(
            "anomaly_map",
            [](AnomalyResult& r) {
                return nb::ndarray<uint8_t, nb::numpy, nb::c_contig>(r.anomaly_map.data,
                                                                     {static_cast<size_t>(r.anomaly_map.rows),
                                                                      static_cast<size_t>(r.anomaly_map.cols),
                                                                      static_cast<size_t>(r.anomaly_map.channels())});
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "pred_boxes",
            [](AnomalyResult& r) {
                return nb::ndarray<int, nb::numpy, nb::c_contig>(r.pred_boxes.data(), {r.pred_boxes.size(), 4});
            },
            nb::rv_policy::reference_internal)
        .def_ro("pred_label", &AnomalyResult::pred_label)
        .def_prop_ro(
            "pred_mask",
            [](AnomalyResult& r) {
                return nb::ndarray<uint8_t, nb::numpy, nb::c_contig>(r.pred_mask.data,
                                                                     {static_cast<size_t>(r.pred_mask.rows),
                                                                      static_cast<size_t>(r.pred_mask.cols),
                                                                      static_cast<size_t>(r.pred_mask.channels())});
            },
            nb::rv_policy::reference_internal)
        .def_ro("pred_score", &AnomalyResult::pred_score);
}
