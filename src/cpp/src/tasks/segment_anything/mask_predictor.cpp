#include "tasks/segment_anything/mask_predictor.h"

#include "tasks/results.h"

// Inputs
constexpr char image_embeddings_tensor_name[]{"image_embeddings"};
constexpr char point_coords_tensor_name[]{"point_coords"};
constexpr char point_labels_tensor_name[]{"point_labels"};
constexpr char orig_image_tensor_name[]{"orig_im_size"};
constexpr char has_mask_tensor_name[]{"has_mask_input"};
constexpr char mask_input_tensor_name[]{"mask_input"};

// Outputs
constexpr char predictions_tensor_name[]{"masks"};
constexpr char low_res_masks_tensor_name[]{"low_res_masks"};
constexpr char iou_predictions_tensor_name[]{"iou_predictions"};

std::map<std::string, ov::Tensor> MaskPredictor::preprocess(std::vector<float> points, std::vector<float> labels) {
    std::vector<float> orig_image_size = {(float)input_image_size.height, (float)input_image_size.width};
    std::vector<float> has_mask_input = {use_previous_mask_input ? 0.0f : 1.0f};

    std::map<std::string, ov::Tensor> input;
    input[image_embeddings_tensor_name] = image_encodings;
    input[point_coords_tensor_name] = ov::Tensor(ov::element::f32, ov::Shape({1, labels.size(), 2}), points.data());
    input[point_labels_tensor_name] = ov::Tensor(ov::element::f32, ov::Shape({1, labels.size()}), labels.data());

    input[orig_image_tensor_name] = ov::Tensor(ov::element::f32, ov::Shape({2}));
    std::copy(orig_image_size.begin(), orig_image_size.end(), input[orig_image_tensor_name].data<float>());

    input[mask_input_tensor_name] = mask_input_tensor;
    input[has_mask_tensor_name] = ov::Tensor(ov::element::f32, ov::Shape({1}));
    std::copy(has_mask_input.begin(), has_mask_input.end(), input[has_mask_tensor_name].data<float>());
    return input;
}

std::vector<SegmentAnythingMask> MaskPredictor::infer(std::vector<cv::Point> positive,
                                                      std::vector<cv::Point> negative) {
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

std::vector<SegmentAnythingMask> MaskPredictor::postprocess(InferenceResult result) {
    auto predictions = result.data[predictions_tensor_name];
    auto low_res_masks = result.data[low_res_masks_tensor_name];
    auto iou_predictions = result.data[iou_predictions_tensor_name];

    // Find the best mask based on IOU tensor
    auto iou_predictions_ptr = iou_predictions.data<float>();
    size_t best_index = 0;
    float highest_iou = 0;
    for (size_t i = 0; i < iou_predictions.get_size(); i++) {
        if (highest_iou < iou_predictions_ptr[i]) {
            highest_iou = iou_predictions_ptr[i];
            best_index = i;
        }
    }

    // Copy chosen low res mask into mask input tensor for next infer.
    float* src_data = low_res_masks.data<float>();
    auto mask_size = mask_input_tensor.get_size();
    std::copy(src_data + best_index * mask_size,
              src_data + (best_index + 1) * mask_size,
              mask_input_tensor.data<float>());

    // Build result
    float* data = predictions.data<float>();
    ov::Shape shape = predictions.get_shape();
    size_t num_masks = shape[1];
    size_t height = shape[2];
    size_t width = shape[3];

    std::vector<SegmentAnythingMask> mask_mats;
    for (size_t i = 0; i < num_masks; ++i) {
        size_t offset = i * height * width;
        cv::Mat mask(height, width, CV_32F, data + offset);
        mask_mats.push_back(SegmentAnythingMask{mask.clone(), iou_predictions_ptr[i]});
    }
    return mask_mats;
}

std::vector<SegmentAnythingMask> MaskPredictor::infer(cv::Rect box) {
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

cv::Matx33f MaskPredictor::build_transform() {
    float scaleX = input_image_tensor_size.width / (float)input_image_size.width;
    float scaleY = input_image_tensor_size.height / (float)input_image_size.height;
    float s = std::min(scaleX, scaleY);
    float sx = s;
    float sy = s;

    // clang-format off
    return cv::Matx33f{
        sx, 0, 0,
        0, sy, 0,
        0, 0, 1,
    };
    // clang-format on
}

void MaskPredictor::reset_mask_input() {
    ov::Shape mask_input_tensor_shape = adapter->getInputShape(mask_input_tensor_name).get_shape();
    mask_input_tensor = ov::Tensor(ov::element::f32, mask_input_tensor_shape);
    std::fill(mask_input_tensor.data<float>(), mask_input_tensor.data<float>() + mask_input_tensor.get_size(), 0.0f);
    use_previous_mask_input = false;
}
