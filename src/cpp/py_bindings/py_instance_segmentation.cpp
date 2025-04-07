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

#include "models/instance_segmentation.h"
#include "models/results.h"
#include "py_utils.hpp"

namespace pyutils = vision::nanobind::utils;

void init_instance_segmentation(nb::module_& m) {
    nb::class_<MaskRCNNModel, ImageModel>(m, "MaskRCNNModel")
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

                return MaskRCNNModel::create_model(model_path, ov_any_config, preload, device);
            },
            nb::arg("model_path"),
            nb::arg("configuration") = ov::AnyMap({}),
            nb::arg("preload") = true,
            nb::arg("device") = "AUTO")

        .def("__call__",
             [](MaskRCNNModel& self, const nb::ndarray<>& input) {
                 return self.infer(pyutils::wrap_np_mat(input));
             })
        .def("infer_batch",
             [](MaskRCNNModel& self, const std::vector<nb::ndarray<>> inputs) {
                 std::vector<ImageInputData> input_mats;
                 input_mats.reserve(inputs.size());

                 for (const auto& input : inputs) {
                     input_mats.push_back(pyutils::wrap_np_mat(input));
                 }

                 return self.inferBatch(input_mats);
             })
        .def_prop_ro_static("__model__", [](nb::object) {
            return MaskRCNNModel::ModelType;
        });

    nb::class_<InstanceSegmentationResult, ResultBase>(m, "InstanceSegmentationResult")
        .def(nb::init<int64_t, std::shared_ptr<MetaData>>(), nb::arg("frameId") = -1, nb::arg("metaData") = nullptr)
        .def_prop_ro(
            "feature_vector",
            [](InstanceSegmentationResult& r) {
                if (!r.feature_vector) {
                    return nb::ndarray<float, nb::numpy, nb::c_contig>();
                }

                return nb::ndarray<float, nb::numpy, nb::c_contig>(r.feature_vector.data(),
                                                                   r.feature_vector.get_shape().size(),
                                                                   r.feature_vector.get_shape().data());
            },
            nb::rv_policy::reference_internal);
}
