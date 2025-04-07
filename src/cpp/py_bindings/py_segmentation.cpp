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

#include "models/results.h"
#include "models/segmentation_model.h"
#include "py_utils.hpp"

namespace pyutils = vision::nanobind::utils;

void init_segmentation(nb::module_& m) {
    nb::class_<SegmentationModel, ImageModel>(m, "SegmentationModel")
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

                return SegmentationModel::create_model(model_path, ov_any_config, preload, device);
            },
            nb::arg("model_path"),
            nb::arg("configuration") = ov::AnyMap({}),
            nb::arg("preload") = true,
            nb::arg("device") = "AUTO")

        .def("__call__",
             [](SegmentationModel& self, const nb::ndarray<>& input) {
                 return self.infer(pyutils::wrap_np_mat(input));
             })
        .def("infer_batch",
             [](SegmentationModel& self, const std::vector<nb::ndarray<>> inputs) {
                 std::vector<ImageInputData> input_mats;
                 input_mats.reserve(inputs.size());

                 for (const auto& input : inputs) {
                     input_mats.push_back(pyutils::wrap_np_mat(input));
                 }

                 return self.inferBatch(input_mats);
             })
        .def_prop_ro_static("__model__", [](nb::object) {
            return SegmentationModel::ModelType;
        });

    nb::class_<ImageResult, ResultBase>(m, "ImageResult")
        .def(nb::init<int64_t, std::shared_ptr<MetaData>>(), nb::arg("frameId") = -1, nb::arg("metaData") = nullptr)
        .def_prop_ro(
            "resultImage",
            [](ImageResult& r) {
                return nb::ndarray<uint8_t, nb::numpy, nb::c_contig>(r.resultImage.data,
                                                                     {static_cast<size_t>(r.resultImage.rows),
                                                                      static_cast<size_t>(r.resultImage.cols),
                                                                      static_cast<size_t>(r.resultImage.channels())});
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "feature_vector",
            [](ResultBase& r) {
                ImageResultWithSoftPrediction ir = r.asRef<ImageResultWithSoftPrediction>();
                if (!ir.feature_vector) {
                    return nb::ndarray<float, nb::numpy, nb::c_contig>();
                }

                return nb::ndarray<float, nb::numpy, nb::c_contig>(ir.feature_vector.data(),
                                                                   ir.feature_vector.get_shape().size(),
                                                                   ir.feature_vector.get_shape().data());
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "soft_prediction",
            [](ResultBase& r) {
                ImageResultWithSoftPrediction ir = r.asRef<ImageResultWithSoftPrediction>();
                return nb::ndarray<float, nb::numpy, nb::c_contig>(
                    ir.soft_prediction.data,
                    {static_cast<size_t>(ir.soft_prediction.rows),
                     static_cast<size_t>(ir.soft_prediction.cols),
                     static_cast<size_t>(ir.soft_prediction.channels())});
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "saliency_map",
            [](ResultBase& r) {
                ImageResultWithSoftPrediction ir = r.asRef<ImageResultWithSoftPrediction>();
                return nb::ndarray<float, nb::numpy, nb::c_contig>(ir.saliency_map.data,
                                                                   {static_cast<size_t>(ir.saliency_map.rows),
                                                                    static_cast<size_t>(ir.saliency_map.cols),
                                                                    static_cast<size_t>(ir.saliency_map.channels())});
            },
            nb::rv_policy::reference_internal);
}
