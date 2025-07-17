#include "tasks/segment_anything/mask_predictor.h"

constexpr char image_embeddings_tensor_name[]{"image_embeddings"};
constexpr char point_coords_tensor_name[]{"point_coords"};
constexpr char point_labels_tensor_name[]{"point_labels"};
constexpr char orig_image_tensor_name[]{"orig_im_size"};
constexpr char has_mask_tensor_name[]{"has_mask_input"};
constexpr char mask_input_tensor_name[]{"mask_input"};

std::map<std::string, ov::Tensor> MaskPredictor::preprocess(std::vector<float> points, std::vector<float> labels) {
    std::vector<float> orig_image_size = {(float)input_image_size.height, (float)input_image_size.width};
    std::vector<float> has_mask_input = {0};

    std::map<std::string, ov::Tensor> input;
    input[image_embeddings_tensor_name] = image_encodings;
    input[point_coords_tensor_name] = ov::Tensor(ov::element::f32, ov::Shape({1, labels.size(), 2}), points.data());
    input[point_labels_tensor_name] = ov::Tensor(ov::element::f32, ov::Shape({1, labels.size()}), labels.data());

    input[orig_image_tensor_name] = ov::Tensor(ov::element::f32, ov::Shape({2}));
    std::copy(orig_image_size.begin(), orig_image_size.end(), input[orig_image_tensor_name].data<float>());

    input[has_mask_tensor_name] = ov::Tensor(ov::element::f32, ov::Shape({1}));
    std::copy(has_mask_input.begin(), has_mask_input.end(), input[has_mask_tensor_name].data<float>());

    ov::Shape mask_input_tensor_shape = adapter->getInputShape(mask_input_tensor_name).get_shape();
    ov::Tensor mask_input_tensor(ov::element::f32, mask_input_tensor_shape);
    std::fill(mask_input_tensor.data<float>(), mask_input_tensor.data<float>() + mask_input_tensor.get_size(), 0.0f);
    input[mask_input_tensor_name] = mask_input_tensor;
    return input;
}

std::vector<cv::Mat> MaskPredictor::infer(std::vector<cv::Point> positive, std::vector<cv::Point> negative) {
    std::map<std::string, ov::Tensor> tensors;


    tensors[image_embeddings_tensor_name] = image_encodings;

    std::vector<float> point_coord_data;
    std::vector<float> point_label_data;
    point_coord_data.reserve(positive.size() * 2);
    point_label_data.reserve(positive.size());

    for (size_t i = 0; i < positive.size(); i++) {
        auto transformed = transform(positive[i]);
        point_coord_data.push_back(transformed.x);
        point_coord_data.push_back(transformed.y);
        point_label_data.push_back(1);
    }

    for (size_t i = 0; i < negative.size(); i++) {
        auto transformed = transform(negative[i]);
        point_coord_data.push_back(transformed.x);
        point_coord_data.push_back(transformed.y);
        point_label_data.push_back(0);
    }

    InferenceResult result;
    result.data = adapter->infer(preprocess(point_coord_data, point_label_data));

    return postprocess(result);
}

std::vector<cv::Mat> MaskPredictor::postprocess(InferenceResult result) {
    auto tensorName = adapter->getOutputNames().front();
    auto predictions = result.data[tensorName];

    float* data = predictions.data<float>();

    ov::Shape shape = predictions.get_shape();
    size_t num_masks = shape[1];
    size_t height = shape[2];
    size_t width = shape[3];

    std::vector<cv::Mat> mask_mats;
    for (size_t i = 0; i < num_masks; ++i) {
        size_t offset = i * height * width;
        cv::Mat mask(height, width, CV_32F, data + offset);
        mask_mats.push_back(mask.clone());
    }
    return mask_mats;
}


std::vector<cv::Mat> MaskPredictor::infer(cv::Rect box) {
    const std::string image_embeddings_tensor_name = "image_embeddings";
    const std::string point_coords_tensor_name = "point_coords";
    const std::string point_labels_tensor_name = "point_labels";
    const std::string orig_image_tensor_name = "orig_im_size";

    std::map<std::string, ov::Tensor> tensors;


    tensors[image_embeddings_tensor_name] = image_encodings;

    std::vector<float> point_coord_data;
    std::vector<float> point_label_data;
    point_coord_data.reserve(2 * 2);
    point_label_data.reserve(2);

    auto tl = transform(box.tl());
    point_coord_data.push_back(tl.x);
    point_coord_data.push_back(tl.y);
    point_label_data.push_back(2);

    auto br = transform(box.br());
    point_coord_data.push_back(br.x);
    point_coord_data.push_back(br.y);
    point_label_data.push_back(2);

    InferenceResult result;
    result.data = adapter->infer(preprocess(point_coord_data, point_label_data));
    return postprocess(result);
}

cv::Point MaskPredictor::transform(cv::Point input) {
    cv::Vec3f vector(input.x, input.y, 1);
    auto scaled = resize_transform * vector;
    return cv::Point(scaled[0], scaled[1]);
}


void MaskPredictor::build_transform() {
    float scaleX = input_image_tensor_size.width / (float)input_image_size.width;
    float scaleY = input_image_tensor_size.height / (float)input_image_size.height;
    float s = std::min(scaleX, scaleY);
    float sx = s;
    float sy = s;

    resize_transform = {
        sx, 0, 0,
        0, sy, 0,
        0, 0, 1,
    };
}
