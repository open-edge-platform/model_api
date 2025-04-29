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

using ScoresOutput = nb::ndarray<float, nb::numpy, nb::c_contig>;
using LabelsOutput = nb::ndarray<size_t, nb::numpy, nb::c_contig>;

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
            nb::rv_policy::reference_internal)
        .def_prop_ro("label_names",
                     [](InstanceSegmentationResult& r) {
                         size_t labels_count = static_cast<size_t>(r.segmentedObjects.size());
                         std::vector<std::string> labels(labels_count);

                         for (size_t i = 0; i < labels_count; ++i) {
                             labels[i] = r.segmentedObjects[i].label;
                         }

                         return labels;
                     })
        .def_prop_ro("labels",
                     [](InstanceSegmentationResult& r) {
                         size_t labels_count = static_cast<size_t>(r.segmentedObjects.size());
                         std::vector<size_t> labels(labels_count);

                         for (size_t i = 0; i < labels_count; ++i) {
                             labels[i] = r.segmentedObjects[i].labelID;
                         }

                         return LabelsOutput(labels.data(), {labels_count}).cast();
                     })
        .def_prop_ro("scores",
                     [](InstanceSegmentationResult& r) {
                         size_t scores_count = static_cast<size_t>(r.segmentedObjects.size());
                         std::vector<float> scores(scores_count);

                         for (size_t i = 0; i < scores_count; ++i) {
                             scores[i] = r.segmentedObjects[i].confidence;
                         }

                         return ScoresOutput(scores.data(), {scores_count}).cast();
                     })
        .def_prop_ro("bboxes",
                     [](InstanceSegmentationResult& r) {
                         size_t boxes_count = static_cast<size_t>(r.segmentedObjects.size());
                         std::vector<std::vector<int>> boxes(boxes_count);

                         for (size_t i = 0; i < boxes_count; ++i) {
                             std::vector<int> box(4);
                             box[0] = r.segmentedObjects[i].tl().x;
                             box[1] = r.segmentedObjects[i].tl().y;
                             box[2] = r.segmentedObjects[i].br().x;
                             box[3] = r.segmentedObjects[i].br().y;
                             boxes[i] = box;
                         }

                         return boxes;
                     })
        .def_prop_ro("masks",
                     [](InstanceSegmentationResult& r) {
                         size_t elements_count = static_cast<size_t>(r.segmentedObjects.size());
                         std::vector<std::vector<std::vector<int>>> masks(elements_count);

                         for (size_t i = 0; i < elements_count; ++i) {
                             int rows = r.segmentedObjects[i].mask.rows;
                             int cols = r.segmentedObjects[i].mask.cols;

                             std::vector<std::vector<int>> mask(rows, std::vector<int>(cols));

                             for (int row = 0; row < rows; ++row) {
                                 for (int col = 0; col < cols; ++col) {
                                     mask[row][col] = r.segmentedObjects[i].mask.at<uint8_t>(row, col);
                                 }
                             }

                             masks[i] = mask;
                         }

                         return masks;
                     })
        .def_prop_ro(
            "saliency_map",
            [](InstanceSegmentationResult& r) {
                if (r.saliency_map.empty()) {
                    return nb::ndarray<uint8_t, nb::numpy, nb::c_contig>();
                }
                int rows = r.saliency_map[0].rows;
                int cols = r.saliency_map[0].cols;
                int num_matrices = r.saliency_map.size();

                return nb::ndarray<uint8_t, nb::numpy, nb::c_contig>(
                    &r.saliency_map,
                    {static_cast<size_t>(num_matrices), static_cast<size_t>(rows), static_cast<size_t>(cols)});
            },
            nb::rv_policy::reference_internal);
}
