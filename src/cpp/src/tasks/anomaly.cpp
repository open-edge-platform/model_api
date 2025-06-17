#include "tasks/anomaly.h"

#include "adapters/openvino_adapter.h"
#include "utils/preprocessing.h"
#include "utils/tensor.h"

cv::Size Anomaly::serialize(std::shared_ptr<ov::Model>& ov_model) {
    auto input = ov_model->inputs().front();

    auto layout = ov::layout::get_layout(input);
    if (layout.empty()) {
        layout = utils::getLayoutFromShape(input.get_partial_shape());
    }

    const ov::Shape& shape = input.get_partial_shape().get_max_shape();

    auto interpolation_mode = cv::INTER_LINEAR;
    utils::RESIZE_MODE resize_mode = utils::RESIZE_FILL;
    uint8_t pad_value = 0;
    bool reverse_input_channels = false;

    std::vector<float> scale_values;
    std::vector<float> mean_values;
    if (ov_model->has_rt_info("model_info")) {
        auto config = ov_model->get_rt_info<ov::AnyMap>("model_info");
        reverse_input_channels = utils::get_from_any_maps("reverse_input_channels", config, ov::AnyMap{}, reverse_input_channels);
        scale_values = utils::get_from_any_maps("scale_values", config, ov::AnyMap{}, scale_values);
        mean_values = utils::get_from_any_maps("mean_values", config, ov::AnyMap{}, mean_values);
    }

    auto input_shape = ov::Shape{shape[ov::layout::width_idx(layout)], shape[ov::layout::height_idx(layout)]};

    ov_model = utils::embedProcessing(ov_model,
                                      input.get_any_name(),
                                      layout,
                                      resize_mode,
                                      interpolation_mode,
                                      input_shape,
                                      pad_value,
                                      reverse_input_channels,
                                      mean_values,
                                      scale_values);

    return cv::Size(input_shape[0], input_shape[1]);
}

Anomaly Anomaly::load(const std::string& model_path) {
    auto core = ov::Core();
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    if (model->has_rt_info("model_info", "model_type")) {
        std::cout << "has model type in info: " << model->get_rt_info<std::string>("model_info", "model_type")
                  << std::endl;
    } else {
        throw std::runtime_error("Incorrect or unsupported model_type");
    }

    cv::Size origin_input_shape;
    if (utils::model_has_embedded_processing(model)) {
        std::cout << "model already was serialized" << std::endl;
        origin_input_shape = utils::get_input_shape_from_model_info(model);
    } else {
        origin_input_shape = serialize(model);
    }
    auto adapter = std::make_shared<OpenVINOInferenceAdapter>();
    adapter->loadModel(model, core, "AUTO");
    return Anomaly(adapter, origin_input_shape);
}

AnomalyResult Anomaly::infer(cv::Mat image) {
    return pipeline.infer(image);
}

std::vector<AnomalyResult> Anomaly::inferBatch(std::vector<cv::Mat> images) {
    return pipeline.inferBatch(images);
}

std::map<std::string, ov::Tensor> Anomaly::preprocess(cv::Mat image) {
    std::map<std::string, ov::Tensor> input = {};
    input.emplace(adapter->getInputNames()[0], utils::wrapMat2Tensor(image));
    return input;
}

AnomalyResult Anomaly::postprocess(InferenceResult& infResult) {
    auto tensorName = adapter->getOutputNames().front();
    ov::Tensor predictions = infResult.data[tensorName];
    const auto& inputImgSize = infResult.inputImageSize;

    double pred_score;
    std::string pred_label;
    cv::Mat anomaly_map;
    cv::Mat pred_mask;
    std::vector<cv::Rect> pred_boxes;
    if (predictions.get_shape().size() == 1) {
        pred_score = predictions.data<float>()[0];
    } else {
        const ov::Layout& layout = utils::getLayoutFromShape(predictions.get_shape());
        const ov::Shape& predictionsShape = predictions.get_shape();
        anomaly_map = cv::Mat(static_cast<int>(predictionsShape[ov::layout::height_idx(layout)]),
                              static_cast<int>(predictionsShape[ov::layout::width_idx(layout)]),
                              CV_32FC1,
                              predictions.data<float>());
        // find the max predicted score
        cv::minMaxLoc(anomaly_map, NULL, &pred_score);
    }
    pred_label = labels[pred_score > image_threshold ? 1 : 0];

    pred_mask = anomaly_map >= pixel_threshold;
    pred_mask.convertTo(pred_mask, CV_8UC1, 1 / 255.);
    cv::resize(pred_mask, pred_mask, cv::Size{inputImgSize.width, inputImgSize.height});
    anomaly_map = normalize(anomaly_map, pixel_threshold);
    anomaly_map.convertTo(anomaly_map, CV_8UC1, 255);

    pred_score = normalize(pred_score, image_threshold);
    if (pred_label == labels[0]) {    // normal label
        pred_score = 1 - pred_score;  // Score of normal is 1 - score of anomaly
    }

    if (!anomaly_map.empty()) {
        cv::resize(anomaly_map, anomaly_map, cv::Size{inputImgSize.width, inputImgSize.height});
    }

    if (task == "detection") {
        pred_boxes = getBoxes(pred_mask);
    }

    AnomalyResult result;
    result.anomaly_map = std::move(anomaly_map);
    result.pred_score = pred_score;
    result.pred_label = std::move(pred_label);
    result.pred_mask = std::move(pred_mask);
    result.pred_boxes = std::move(pred_boxes);
    return result;
}

cv::Mat Anomaly::normalize(cv::Mat& tensor, float threshold) {
    cv::Mat normalized = ((tensor - threshold) / normalization_scale) + 0.5f;
    normalized = cv::min(cv::max(normalized, 0.f), 1.f);
    return normalized;
}

double Anomaly::normalize(double& value, float threshold) {
    double normalized = ((value - threshold) / normalization_scale) + 0.5f;
    return std::min(std::max(normalized, 0.), 1.);
}

std::vector<cv::Rect> Anomaly::getBoxes(cv::Mat& mask) {
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (auto& contour : contours) {
        std::vector<int> box;
        cv::Rect rect = cv::boundingRect(contour);
        boxes.push_back(rect);
    }
    return boxes;
}
