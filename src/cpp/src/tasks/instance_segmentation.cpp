/*
 * Copyright (C) 2020-2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tasks/instance_segmentation.h"

#include "adapters/openvino_adapter.h"
#include "utils/config.h"
#include "utils/math.h"
#include "utils/nms.h"
#include "utils/preprocessing.h"
#include "utils/tensor.h"

constexpr char saliency_map_name[]{"saliency_map"};
constexpr char feature_vector_name[]{"feature_vector"};

void append_xai_names(const std::vector<ov::Output<ov::Node>>& outputs, std::vector<std::string>& outputNames) {
    for (const ov::Output<ov::Node>& output : outputs) {
        if (output.get_names().count(saliency_map_name) > 0) {
            outputNames.emplace_back(saliency_map_name);
        } else if (output.get_names().count(feature_vector_name) > 0) {
            outputNames.push_back(feature_vector_name);
        }
    }
}

cv::Rect expand_box(const cv::Rect2f& box, float scale) {
    float w_half = box.width * 0.5f * scale, h_half = box.height * 0.5f * scale;
    const cv::Point2f& center = (box.tl() + box.br()) * 0.5f;
    return {cv::Point(int(center.x - w_half), int(center.y - h_half)),
            cv::Point(int(center.x + w_half), int(center.y + h_half))};
}

std::vector<cv::Mat_<std::uint8_t>> average_and_normalize(const std::vector<std::vector<cv::Mat>>& saliency_maps) {
    std::vector<cv::Mat_<std::uint8_t>> aggregated;
    aggregated.reserve(saliency_maps.size());
    for (const std::vector<cv::Mat>& per_object_maps : saliency_maps) {
        if (per_object_maps.empty()) {
            aggregated.emplace_back();
        } else {
            cv::Mat_<double> saliency_map{per_object_maps.front().size()};
            for (const cv::Mat& per_object_map : per_object_maps) {
                if (saliency_map.size != per_object_map.size) {
                    throw std::runtime_error("saliency_maps must have same size");
                }
                if (per_object_map.channels() != 1) {
                    throw std::runtime_error("saliency_maps must have one channel");
                }
                if (per_object_map.type() != CV_8U) {
                    throw std::runtime_error("saliency_maps must have type CV_8U");
                }
            }
            for (int row = 0; row < saliency_map.rows; ++row) {
                for (int col = 0; col < saliency_map.cols; ++col) {
                    std::uint8_t max_val = 0;
                    for (const cv::Mat& per_object_map : per_object_maps) {
                        max_val = std::max(max_val, per_object_map.at<std::uint8_t>(row, col));
                    }
                    saliency_map.at<double>(row, col) = max_val;
                }
            }
            double min, max;
            cv::minMaxLoc(saliency_map, &min, &max);
            cv::Mat_<std::uint8_t> converted;
            saliency_map.convertTo(converted, CV_8U, 255.0 / (max + 1e-12));
            aggregated.push_back(std::move(converted));
        }
    }
    return aggregated;
}

struct Lbm {
    ov::Tensor labels, boxes, masks;
};

Lbm filterTensors(const std::map<std::string, ov::Tensor>& infResult) {
    Lbm lbm;
    for (const auto& pair : infResult) {
        if (pair.first == saliency_map_name || pair.first == feature_vector_name) {
            continue;
        }
        switch (pair.second.get_shape().size()) {
        case 2:
            lbm.labels = pair.second;
            break;
        case 3:
            lbm.boxes = pair.second;
            break;
        case 4:
            lbm.masks = pair.second;
            break;
        case 0:
            break;
        default:
            throw std::runtime_error("Unexpected result: " + pair.first);
        }
    }
    return lbm;
}

cv::Mat segm_postprocess(const SegmentedObject& box, const cv::Mat& unpadded, int im_h, int im_w) {
    cv::Mat raw_cls_mask;
    cv::copyMakeBorder(unpadded, raw_cls_mask, 1, 1, 1, 1, cv::BORDER_CONSTANT, {0});
    cv::Rect extended_box = expand_box(box, float(raw_cls_mask.cols) / (raw_cls_mask.cols - 2));

    int w = std::max(extended_box.width + 1, 1);
    int h = std::max(extended_box.height + 1, 1);
    int x0 = clamp(extended_box.x, 0, im_w);
    int y0 = clamp(extended_box.y, 0, im_h);
    int x1 = clamp(extended_box.x + extended_box.width + 1, 0, im_w);
    int y1 = clamp(extended_box.y + extended_box.height + 1, 0, im_h);

    cv::Mat resized;
    cv::resize(raw_cls_mask, resized, {w, h});
    cv::Mat im_mask(cv::Size{im_w, im_h}, CV_8UC1, cv::Scalar{0});
    im_mask(cv::Rect{x0, y0, x1 - x0, y1 - y0})
        .setTo(1,
               resized({cv::Point(x0 - extended_box.x, y0 - extended_box.y),
                        cv::Point(x1 - extended_box.x, y1 - extended_box.y)}) > 0.5f);
    return im_mask;
}

void InstanceSegmentation::serialize(std::shared_ptr<ov::Model>& ov_model) {
    if (ov_model->inputs().size() != 1) {
        throw std::logic_error("MaskRCNNModel model wrapper supports topologies with only 1 input");
    }
    const auto& input = ov_model->input();
    auto config = ov_model->has_rt_info("model_info") ? ov_model->get_rt_info<ov::AnyMap>("model_info") : ov::AnyMap{};
    std::string layout = "";
    layout = utils::get_from_any_maps("layout", config, {}, layout);
    auto inputsLayouts = utils::parseLayoutString(layout);
    const ov::Layout& inputLayout = utils::getInputLayout(input, inputsLayouts);
    const ov::Shape& inputShape = input.get_partial_shape().get_max_shape();
    if (inputShape.size() != 4 || inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
        throw std::logic_error("3-channel 4-dimensional model's input is expected");
    }

    auto interpolation_mode = cv::INTER_LINEAR;
    utils::RESIZE_MODE resize_mode;
    resize_mode = utils::get_from_any_maps("resize_type", config, ov::AnyMap{}, resize_mode);

    std::vector<float> scale_values;
    std::vector<float> mean_values;
    scale_values = utils::get_from_any_maps("scale_values", config, ov::AnyMap{}, scale_values);
    mean_values = utils::get_from_any_maps("mean_values", config, ov::AnyMap{}, mean_values);
    uint8_t pad_value = 0;
    bool reverse_input_channels = false;
    reverse_input_channels =
        utils::get_from_any_maps("reverse_input_channels", config, ov::AnyMap{}, reverse_input_channels);

    ov_model = utils::embedProcessing(
        ov_model,
        input.get_any_name(),
        inputLayout,
        resize_mode,
        interpolation_mode,
        ov::Shape{inputShape[ov::layout::width_idx(inputLayout)], inputShape[ov::layout::height_idx(inputLayout)]},
        pad_value,
        reverse_input_channels,
        mean_values,
        scale_values);

    cv::Size input_shape(inputShape[ov::layout::width_idx(inputLayout)],
                         inputShape[ov::layout::height_idx(inputLayout)]);

    // --------------------------- Prepare output  -----------------------------------------------------
    struct NameRank {
        std::string name;
        size_t rank;
    };
    std::vector<NameRank> filtered;
    filtered.reserve(3);
    for (ov::Output<ov::Node>& output : ov_model->outputs()) {
        const std::unordered_set<std::string>& out_names = output.get_names();
        if (out_names.find(saliency_map_name) == out_names.end() &&
            out_names.find(feature_vector_name) == out_names.end()) {
            filtered.push_back({output.get_any_name(), output.get_partial_shape().get_max_shape().size()});
        }
    }
    if (filtered.size() != 3 && filtered.size() != 4) {
        throw std::logic_error(std::string{"MaskRCNNModel model wrapper supports topologies with "} +
                               saliency_map_name + ", " + feature_vector_name + " and 3 or 4 other outputs");
    }

    ov_model->set_rt_info(input_shape.width, "model_info", "orig_width");
    ov_model->set_rt_info(input_shape.height, "model_info", "orig_height");
}

InstanceSegmentation InstanceSegmentation::load(const std::string& model_path, const ov::AnyMap& configuration) {
    auto core = ov::Core();
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    if (model->has_rt_info("model_info", "model_type")) {
        std::cout << "has model type in info: " << model->get_rt_info<std::string>("model_info", "model_type")
                  << std::endl;
    } else {
        throw std::runtime_error("Incorrect or unsupported model_type");
    }

    if (utils::model_has_embedded_processing(model)) {
        std::cout << "model already was serialized" << std::endl;
    } else {
        serialize(model);
    }
    auto adapter = std::make_shared<OpenVINOInferenceAdapter>();
    adapter->loadModel(model, core, "AUTO");
    return InstanceSegmentation(adapter, configuration);
}

InstanceSegmentationResult InstanceSegmentation::infer(cv::Mat image) {
    return pipeline->infer(image);
}

std::vector<InstanceSegmentationResult> InstanceSegmentation::inferBatch(std::vector<cv::Mat> images) {
    return pipeline->inferBatch(images);
}

std::map<std::string, ov::Tensor> InstanceSegmentation::preprocess(cv::Mat image) {
    std::map<std::string, ov::Tensor> input = {};
    input.emplace(adapter->getInputNames()[0], utils::wrapMat2Tensor(image));
    return input;
}

InstanceSegmentationResult InstanceSegmentation::postprocess(InferenceResult& infResult) {
    float floatInputImgWidth = float(infResult.inputImageSize.width),
          floatInputImgHeight = float(infResult.inputImageSize.height);
    float invertedScaleX = floatInputImgWidth / input_shape.width,
          invertedScaleY = floatInputImgHeight / input_shape.height;

    std::cout << "got an inf result with image: " << infResult.inputImageSize << std::endl;
    std::cout << "resize mode: " << resize_mode << std::endl;
    int padLeft = 0, padTop = 0;
    if (utils::RESIZE_KEEP_ASPECT == resize_mode || utils::RESIZE_KEEP_ASPECT_LETTERBOX == resize_mode) {
        std::cout << "using some other resize mode..." << std::endl;
        invertedScaleX = invertedScaleY = std::max(invertedScaleX, invertedScaleY);
        if (utils::RESIZE_KEEP_ASPECT_LETTERBOX == resize_mode) {
            padLeft = (input_shape.width - int(std::round(floatInputImgWidth / invertedScaleX))) / 2;
            padTop = (input_shape.height - int(std::round(floatInputImgHeight / invertedScaleY))) / 2;
        }
    }
    const Lbm& lbm = filterTensors(infResult.data);
    const int64_t* const labels_tensor_ptr = lbm.labels.data<int64_t>();
    const float* const boxes = lbm.boxes.data<float>();
    size_t objectSize = lbm.boxes.get_shape().back();
    float* const masks = lbm.masks.data<float>();
    const cv::Size& masks_size{int(lbm.masks.get_shape()[3]), int(lbm.masks.get_shape()[2])};
    InstanceSegmentationResult result;
    std::vector<std::vector<cv::Mat>> saliency_maps;
    auto outputNames = adapter->getOutputNames();
    bool has_feature_vector_name =
        std::find(outputNames.begin(), outputNames.end(), feature_vector_name) != outputNames.end();

    if (has_feature_vector_name) {
        if (labels.empty()) {
            throw std::runtime_error("Can't get number of classes because labels are empty");
        }
        saliency_maps.resize(labels.size());
    }

    for (size_t i = 0; i < lbm.labels.get_size(); ++i) {
        float confidence = boxes[i * objectSize + 4];
        if (confidence <= confidence_threshold && !has_feature_vector_name) {
            continue;
        }
        SegmentedObject obj;

        obj.confidence = confidence;
        obj.labelID = labels_tensor_ptr[i] + 1;
        if (!labels.empty() && obj.labelID >= labels.size()) {
            continue;
        }
        obj.label = getLabelName(obj.labelID);

        obj.x = clamp(round((boxes[i * objectSize + 0] - padLeft) * invertedScaleX), 0.f, floatInputImgWidth);
        obj.y = clamp(round((boxes[i * objectSize + 1] - padTop) * invertedScaleY), 0.f, floatInputImgHeight);
        obj.width =
            clamp(round((boxes[i * objectSize + 2] - padLeft) * invertedScaleX - obj.x), 0.f, floatInputImgWidth);
        obj.height =
            clamp(round((boxes[i * objectSize + 3] - padTop) * invertedScaleY - obj.y), 0.f, floatInputImgHeight);

        if (obj.height * obj.width <= 1) {
            continue;
        }

        cv::Mat raw_cls_mask{masks_size, CV_32F, masks + masks_size.area() * i};
        cv::Mat resized_mask;
        if (postprocess_semantic_masks || has_feature_vector_name) {
            resized_mask =
                segm_postprocess(obj, raw_cls_mask, infResult.inputImageSize.height, infResult.inputImageSize.width);
        } else {
            resized_mask = raw_cls_mask;
        }
        obj.mask = postprocess_semantic_masks ? resized_mask : raw_cls_mask.clone();
        if (confidence > confidence_threshold) {
            result.segmentedObjects.push_back(obj);
        }
        if (has_feature_vector_name && confidence > confidence_threshold) {
            saliency_maps[obj.labelID - 1].push_back(resized_mask);
        }
    }
    result.saliency_map = average_and_normalize(saliency_maps);
    if (has_feature_vector_name) {
        result.feature_vector = std::move(infResult.data[feature_vector_name]);
    }
    return result;
}

InstanceSegmentationResult InstanceSegmentation::postprocess_tile(InstanceSegmentationResult result,
                                                                  const cv::Rect& coord) {
    for (auto& det : result.segmentedObjects) {
        det.x += coord.x;
        det.y += coord.y;
    }

    if (result.feature_vector) {
        auto tmp_feature_vector =
            ov::Tensor(result.feature_vector.get_element_type(), result.feature_vector.get_shape());
        result.feature_vector.copy_to(tmp_feature_vector);
        result.feature_vector = tmp_feature_vector;
    }

    return result;
}

InstanceSegmentationResult InstanceSegmentation::merge_tiling_results(
    const std::vector<InstanceSegmentationResult>& tiles_results,
    const cv::Size& image_size,
    const std::vector<cv::Rect>& tile_coords,
    const utils::TilingInfo& tiling_info) {
    size_t max_pred_number = 200;  // TODO: Actually get this from config!

    InstanceSegmentationResult output;
    std::vector<AnchorLabeled> all_detections;
    std::vector<std::reference_wrapper<const SegmentedObject>> all_detections_ptrs;
    std::vector<float> all_scores;

    for (auto& result : tiles_results) {
        for (auto& det : result.segmentedObjects) {
            all_detections.emplace_back(det.x, det.y, det.x + det.width, det.y + det.height, det.labelID);
            all_scores.push_back(det.confidence);
            all_detections_ptrs.push_back(det);
        }
    }

    auto keep_idx = multiclass_nms(all_detections, all_scores, tiling_info.iou_threshold, false, max_pred_number);

    output.segmentedObjects.reserve(keep_idx.size());
    for (auto idx : keep_idx) {
        if (postprocess_semantic_masks) {
            // why does this happen again?
            // all_detections_ptrs[idx].get().mask = ;
            // SegmentedObject obj = all_detections_ptrs[idx]; //copy
            // std::cout << "Mask size before: " << obj.mask.size() << std::endl;
            // std::cout << static_cast<cv::Rect>(obj) << std::endl;
            // obj.mask = segm_postprocess(all_detections_ptrs[idx],
            //                                                        obj.mask,
            //                                                        image_size.height,
            //                                                        image_size.width);
        }

        output.segmentedObjects.push_back(all_detections_ptrs[idx]);
    }

    if (tiles_results.size()) {
        auto first = tiles_results.front();
        if (first.feature_vector) {
            output.feature_vector =
                ov::Tensor(first.feature_vector.get_element_type(), first.feature_vector.get_shape());
        }
    }

    if (output.feature_vector) {
        float* feature_ptr = output.feature_vector.data<float>();
        size_t feature_size = output.feature_vector.get_size();

        std::fill(feature_ptr, feature_ptr + feature_size, 0.f);

        for (const auto& result : tiles_results) {
            const float* current_feature_ptr = result.feature_vector.data<float>();

            for (size_t i = 0; i < feature_size; ++i) {
                feature_ptr[i] += current_feature_ptr[i];
            }
        }

        for (size_t i = 0; i < feature_size; ++i) {
            feature_ptr[i] /= tiles_results.size();
        }
    }

    output.saliency_map = merge_saliency_maps(tiles_results, image_size, tile_coords, tiling_info);

    return output;
}

std::vector<cv::Mat_<std::uint8_t>> InstanceSegmentation::merge_saliency_maps(
    const std::vector<InstanceSegmentationResult>& tiles_results,
    const cv::Size& image_size,
    const std::vector<cv::Rect>& tile_coords,
    const utils::TilingInfo& tiling_info) {
    std::vector<std::vector<cv::Mat_<std::uint8_t>>> all_saliency_maps;
    all_saliency_maps.reserve(tiles_results.size());
    for (const auto& result : tiles_results) {
        all_saliency_maps.push_back(result.saliency_map);
    }

    std::vector<cv::Mat_<std::uint8_t>> image_saliency_map;
    if (all_saliency_maps.size()) {
        image_saliency_map = all_saliency_maps[0];
    }

    if (image_saliency_map.empty()) {
        return image_saliency_map;
    }

    size_t num_classes = image_saliency_map.size();
    std::vector<cv::Mat_<std::uint8_t>> merged_map(num_classes);
    for (auto& map : merged_map) {
        map = cv::Mat_<std::uint8_t>(image_size, 0);
    }

    size_t start_idx = tiling_info.tile_with_full_image ? 1 : 0;
    for (size_t i = start_idx; i < all_saliency_maps.size(); ++i) {
        for (size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
            auto current_cls_map_mat = all_saliency_maps[i][class_idx];
            if (current_cls_map_mat.empty()) {
                continue;
            }
            const auto& tile = tile_coords[i];
            cv::Mat tile_map;
            cv::resize(current_cls_map_mat, tile_map, tile.size());
            auto tile_map_merged = cv::Mat(merged_map[class_idx], tile);
            cv::Mat(cv::max(tile_map, tile_map_merged)).copyTo(tile_map_merged);
        }
    }

    for (size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
        auto image_map_cls =
            tiling_info.tile_with_full_image ? image_saliency_map[class_idx] : cv::Mat_<std::uint8_t>();
        if (image_map_cls.empty()) {
            if (cv::sum(merged_map[class_idx]) == cv::Scalar(0.)) {
                merged_map[class_idx] = cv::Mat_<std::uint8_t>();
            }
        } else {
            cv::resize(image_map_cls, image_map_cls, image_size);
            cv::Mat(cv::max(merged_map[class_idx], image_map_cls)).copyTo(merged_map[class_idx]);
        }
    }

    return merged_map;
}

std::vector<SegmentedObjectWithRects> InstanceSegmentation::getRotatedRectangles(
    const InstanceSegmentationResult& result) {
    std::vector<SegmentedObjectWithRects> objects_with_rects;
    objects_with_rects.reserve(result.segmentedObjects.size());
    for (const SegmentedObject& segmented_object : result.segmentedObjects) {
        objects_with_rects.push_back(SegmentedObjectWithRects{segmented_object});
        cv::Mat mask;
        segmented_object.mask.convertTo(mask, CV_8UC1);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Point> contour = {};
        for (size_t i = 0; i < contours.size(); i++) {
            contour.insert(contour.end(), contours[i].begin(), contours[i].end());
        }
        if (contour.size() > 0) {
            std::vector<cv::Point> hull;
            cv::convexHull(contour, hull);
            objects_with_rects.back().rotated_rect = cv::minAreaRect(hull);
        }
    }
    return objects_with_rects;
}

std::vector<Contour> InstanceSegmentation::getContours(const std::vector<SegmentedObject>& objects) {
    std::vector<Contour> combined_contours;
    std::vector<std::vector<cv::Point>> contours;
    for (const SegmentedObject& obj : objects) {
        cv::findContours(obj.mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        // Assuming one contour output for findContours. Based on OTX this is a safe
        // assumption
        if (contours.size() != 1) {
            throw std::runtime_error("findContours() must have returned only one contour");
        }
        combined_contours.push_back({obj.label, obj.confidence, contours[0]});
    }
    return combined_contours;
}
