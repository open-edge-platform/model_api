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

#include "models/detection_model.h"
#include "models/results.h"
#include "py_utils.hpp"

namespace pyutils = vision::nanobind::utils;

void init_detection(nb::module_& m) {
    nb::class_<DetectionModel, ImageModel>(m, "DetectionModel")
        .def_static(
            "create_model",
            [](const std::string& model_path,
               const std::map<std::string, nb::object>& configuration,
               std::string model_type,
               bool preload,
               const std::string& device) {
                auto ov_any_config = ov::AnyMap();
                for (const auto& item : configuration) {
                    ov_any_config[item.first] = pyutils::py_object_to_any(item.second, item.first);
                }

                return DetectionModel::create_model(model_path, ov_any_config, model_type, preload, device);
            },
            nb::arg("model_path"),
            nb::arg("configuration") = ov::AnyMap({}),
            nb::arg("model_type") = "",
            nb::arg("preload") = true,
            nb::arg("device") = "AUTO")

        .def("__call__",
             [](DetectionModel& self, const nb::ndarray<>& input) {
                 return self.infer(pyutils::wrap_np_mat(input));
             })
        .def("infer_batch", [](DetectionModel& self, const std::vector<nb::ndarray<>> inputs) {
            std::vector<ImageInputData> input_mats;
            input_mats.reserve(inputs.size());

            for (const auto& input : inputs) {
                input_mats.push_back(pyutils::wrap_np_mat(input));
            }

            return self.inferBatch(input_mats);
        });

    nb::class_<DetectionResult, ResultBase>(m, "DetectionResult")
        .def(nb::init<>())
        .def_prop_ro(
            "saliency_map",
            [](DetectionResult& r) {
                if (!r.saliency_map) {
                    return nb::ndarray<float, nb::numpy, nb::c_contig>();
                }

                return nb::ndarray<float, nb::numpy, nb::c_contig>(r.saliency_map.data(),
                                                                   r.saliency_map.get_shape().size(),
                                                                   r.saliency_map.get_shape().data());
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "feature_vector",
            [](DetectionResult& r) {
                if (!r.feature_vector) {
                    return nb::ndarray<float, nb::numpy, nb::c_contig>();
                }

                return nb::ndarray<float, nb::numpy, nb::c_contig>(r.feature_vector.data(),
                                                                   r.feature_vector.get_shape().size(),
                                                                   r.feature_vector.get_shape().data());
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "label_names",
            [](DetectionResult& r) {
                std::vector<std::string> labels;
                std::transform(r.objects.begin(),
                               r.objects.end(),
                               std::back_inserter(labels),
                               [](const DetectedObject& obj) {
                                   return obj.label;
                               });

                return labels;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "scores",
            [](DetectionResult& r) {
                std::vector<float> scores;
                std::transform(r.objects.begin(),
                               r.objects.end(),
                               std::back_inserter(scores),
                               [](const DetectedObject& obj) {
                                   return obj.confidence;
                               });
                return nb::ndarray<float, nb::numpy, nb::c_contig>(scores.data(), {scores.size()}).cast();
            },
            nb::rv_policy::move)
        .def_prop_ro(
            "labels",
            [](DetectionResult& r) {
                std::vector<size_t> labels;
                std::transform(r.objects.begin(),
                               r.objects.end(),
                               std::back_inserter(labels),
                               [](const DetectedObject& obj) {
                                   return obj.labelID;
                               });
                return nb::ndarray<float, nb::numpy, nb::c_contig>(labels.data(), {labels.size()}).cast();
            },
            nb::rv_policy::move)
        .def_prop_ro(
            "bboxes",
            [](DetectionResult& r) {
                std::vector<cv::Rect2f> bboxes;
                std::transform(r.objects.begin(),
                               r.objects.end(),
                               std::back_inserter(bboxes),
                               [](const DetectedObject& obj) {
                                   return cv::Rect2f(obj.x, obj.y, obj.width, obj.height);
                               });
                return nb::ndarray<float, nb::numpy, nb::c_contig>(bboxes.data(), {bboxes.size(), 4}).cast();
            },
            nb::rv_policy::move);
}
