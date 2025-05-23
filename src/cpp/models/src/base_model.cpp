/*
 * Copyright (C) 2020-2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "models/base_model.h"

#include <adapters/inference_adapter.h>
#include <utils/image_utils.h>

#include <fstream>
#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>
#include <stdexcept>
#include <utils/ocv_common.hpp>
#include <vector>

#include "adapters/openvino_adapter.h"
#include "models/input_data.h"
#include "models/internal_model_data.h"
#include "models/results.h"
#include "utils/common.hpp"

namespace {
class TmpCallbackSetter {
public:
    BaseModel* model;
    std::function<void(std::unique_ptr<ResultBase>, const ov::AnyMap&)> last_callback;
    TmpCallbackSetter(BaseModel* model_,
                      std::function<void(std::unique_ptr<ResultBase>, const ov::AnyMap&)> tmp_callback,
                      std::function<void(std::unique_ptr<ResultBase>, const ov::AnyMap&)> last_callback_)
        : model(model_),
          last_callback(last_callback_) {
        model->setCallback(tmp_callback);
    }
    ~TmpCallbackSetter() {
        if (last_callback) {
            model->setCallback(last_callback);
        } else {
            model->setCallback([](std::unique_ptr<ResultBase>, const ov::AnyMap&) {});
        }
    }
};
}  // namespace

BaseModel::BaseModel(const std::string& modelFile,
                     const std::string& resize_type,
                     bool useAutoResize,
                     const std::string& layout)
    : useAutoResize(useAutoResize),
      resizeMode(selectResizeMode(resize_type)),
      modelFile(modelFile),
      inputsLayouts(parseLayoutString(layout)) {
    auto core = ov::Core();
    model = core.read_model(modelFile);
}

void BaseModel::load(ov::Core& core, const std::string& device, size_t num_infer_requests) {
    if (!inferenceAdapter) {
        inferenceAdapter = std::make_shared<OpenVINOInferenceAdapter>();
    }

    // Update model_info erased by pre/postprocessing
    updateModelInfo();

    inferenceAdapter->loadModel(model, core, device, {}, num_infer_requests);
}

std::shared_ptr<ov::Model> BaseModel::prepare() {
    prepareInputsOutputs(model);
    logBasicModelInfo(model);
    ov::set_batch(model, 1);

    return model;
}

ov::Layout BaseModel::getInputLayout(const ov::Output<ov::Node>& input) {
    ov::Layout layout = ov::layout::get_layout(input);
    if (layout.empty()) {
        if (inputsLayouts.empty()) {
            layout = getLayoutFromShape(input.get_partial_shape());
            slog::warn << "Automatically detected layout '" << layout.to_string() << "' for input '"
                       << input.get_any_name() << "' will be used." << slog::endl;
        } else if (inputsLayouts.size() == 1) {
            layout = inputsLayouts.begin()->second;
        } else {
            layout = inputsLayouts[input.get_any_name()];
        }
    }

    return layout;
}

size_t BaseModel::getNumAsyncExecutors() const {
    return inferenceAdapter->getNumAsyncExecutors();
}

bool BaseModel::isReady() {
    return inferenceAdapter->isReady();
}
void BaseModel::awaitAll() {
    inferenceAdapter->awaitAll();
}
void BaseModel::awaitAny() {
    inferenceAdapter->awaitAny();
}

void BaseModel::setCallback(
    std::function<void(std::unique_ptr<ResultBase>, const ov::AnyMap& callback_args)> callback) {
    lastCallback = callback;
    inferenceAdapter->setCallback([this, callback](ov::InferRequest request, CallbackData args) {
        InferenceResult result;

        InferenceOutput output;
        for (const auto& item : this->getInferenceAdapter()->getOutputNames()) {
            output.emplace(item, request.get_tensor(item));
        }

        result.outputsData = output;
        auto model_data_iter = args->find("internalModelData");
        if (model_data_iter != args->end()) {
            result.internalModelData = std::move(model_data_iter->second.as<std::shared_ptr<InternalModelData>>());
        }
        auto retVal = this->postprocess(result);
        *retVal = static_cast<ResultBase&>(result);
        callback(std::move(retVal), args ? *args : ov::AnyMap());
    });
}

std::shared_ptr<ov::Model> BaseModel::getModel() {
    if (!model) {
        throw std::runtime_error(std::string("ov::Model is not accessible for the current model adapter: ") +
                                 typeid(inferenceAdapter).name());
    }

    updateModelInfo();
    return model;
}

std::shared_ptr<InferenceAdapter> BaseModel::getInferenceAdapter() {
    if (!inferenceAdapter) {
        throw std::runtime_error(std::string("Model wasn't loaded"));
    }

    return inferenceAdapter;
}

RESIZE_MODE BaseModel::selectResizeMode(const std::string& resize_type) {
    RESIZE_MODE resize = RESIZE_FILL;
    if ("crop" == resize_type) {
        resize = RESIZE_CROP;
    } else if ("standard" == resize_type) {
        resize = RESIZE_FILL;
    } else if ("fit_to_window" == resize_type) {
        resize = RESIZE_KEEP_ASPECT;
    } else if ("fit_to_window_letterbox" == resize_type) {
        resize = RESIZE_KEEP_ASPECT_LETTERBOX;
    } else {
        throw std::runtime_error("Unknown value for resize_type arg");
    }

    return resize;
}

void BaseModel::init_from_config(const ov::AnyMap& top_priority, const ov::AnyMap& mid_priority) {
    useAutoResize = get_from_any_maps("auto_resize", top_priority, mid_priority, useAutoResize);

    std::string resize_type = "standard";
    resize_type = get_from_any_maps("resize_type", top_priority, mid_priority, resize_type);
    resizeMode = selectResizeMode(resize_type);

    labels = get_from_any_maps("labels", top_priority, mid_priority, labels);
    embedded_processing = get_from_any_maps("embedded_processing", top_priority, mid_priority, embedded_processing);
    netInputWidth = get_from_any_maps("orig_width", top_priority, mid_priority, netInputWidth);
    netInputHeight = get_from_any_maps("orig_height", top_priority, mid_priority, netInputHeight);
    int pad_value_int = 0;
    pad_value_int = get_from_any_maps("pad_value", top_priority, mid_priority, pad_value_int);
    if (0 > pad_value_int || 255 < pad_value_int) {
        throw std::runtime_error("pad_value must be in range [0, 255]");
    }
    pad_value = static_cast<uint8_t>(pad_value_int);
    reverse_input_channels =
        get_from_any_maps("reverse_input_channels", top_priority, mid_priority, reverse_input_channels);
    scale_values = get_from_any_maps("scale_values", top_priority, mid_priority, scale_values);
    mean_values = get_from_any_maps("mean_values", top_priority, mid_priority, mean_values);
}

BaseModel::BaseModel(std::shared_ptr<ov::Model>& model, const ov::AnyMap& configuration) : model(model) {
    auto layout_iter = configuration.find("layout");
    std::string layout = "";

    if (layout_iter != configuration.end()) {
        layout = layout_iter->second.as<std::string>();
    } else {
        if (model->has_rt_info("model_info", "layout")) {
            layout = model->get_rt_info<std::string>("model_info", "layout");
        }
    }
    inputsLayouts = parseLayoutString(layout);
    init_from_config(configuration,
                     model->has_rt_info("model_info") ? model->get_rt_info<ov::AnyMap>("model_info") : ov::AnyMap{});
}

BaseModel::BaseModel(std::shared_ptr<InferenceAdapter>& adapter, const ov::AnyMap& configuration)
    : inferenceAdapter(adapter) {
    const ov::AnyMap& adapter_configuration = adapter->getModelConfig();

    std::string layout = "";
    layout = get_from_any_maps("layout", configuration, adapter_configuration, layout);
    inputsLayouts = parseLayoutString(layout);

    inputNames = adapter->getInputNames();
    outputNames = adapter->getOutputNames();

    init_from_config(configuration, adapter->getModelConfig());
}

std::unique_ptr<ResultBase> BaseModel::inferImage(const ImageInputData& inputData) {
    InferenceInput inputs;
    InferenceResult result;
    auto internalModelData = this->preprocess(inputData, inputs);

    result.outputsData = inferenceAdapter->infer(inputs);
    result.internalModelData = std::move(internalModelData);

    auto retVal = this->postprocess(result);
    *retVal = static_cast<ResultBase&>(result);
    return retVal;
}

std::vector<std::unique_ptr<ResultBase>> BaseModel::inferBatchImage(const std::vector<ImageInputData>& inputImgs) {
    std::vector<std::reference_wrapper<const ImageInputData>> inputData;
    inputData.reserve(inputImgs.size());
    for (const auto& img : inputImgs) {
        inputData.push_back(img);
    }
    auto results = std::vector<std::unique_ptr<ResultBase>>(inputData.size());
    auto setter = TmpCallbackSetter(
        this,
        [&](std::unique_ptr<ResultBase> result, const ov::AnyMap& callback_args) {
            size_t id = callback_args.find("id")->second.as<size_t>();
            results[id] = std::move(result);
        },
        lastCallback);
    size_t req_id = 0;
    for (const auto& data : inputData) {
        inferAsync(data, {{"id", req_id++}});
    }
    awaitAll();
    return results;
}

void BaseModel::inferAsync(const ImageInputData& inputData, const ov::AnyMap& callback_args) {
    InferenceInput inputs;
    auto internalModelData = this->preprocess(inputData, inputs);
    auto callback_args_ptr = std::make_shared<ov::AnyMap>(callback_args);
    (*callback_args_ptr)["internalModelData"] = std::move(internalModelData);
    inferenceAdapter->inferAsync(inputs, callback_args_ptr);
}

void BaseModel::updateModelInfo() {
    if (!model) {
        throw std::runtime_error("The ov::Model object is not accessible");
    }

    if (!inputsLayouts.empty()) {
        auto layouts = formatLayouts(inputsLayouts);
        model->set_rt_info(layouts, "model_info", "layout");
    }

    model->set_rt_info(useAutoResize, "model_info", "auto_resize");
    model->set_rt_info(formatResizeMode(resizeMode), "model_info", "resize_type");

    if (!labels.empty()) {
        model->set_rt_info(labels, "model_info", "labels");
    }

    model->set_rt_info(embedded_processing, "model_info", "embedded_processing");
    model->set_rt_info(netInputWidth, "model_info", "orig_width");
    model->set_rt_info(netInputHeight, "model_info", "orig_height");
}

std::shared_ptr<ov::Model> BaseModel::embedProcessing(std::shared_ptr<ov::Model>& model,
                                                      const std::string& inputName,
                                                      const ov::Layout& layout,
                                                      const RESIZE_MODE resize_mode,
                                                      const cv::InterpolationFlags interpolationMode,
                                                      const ov::Shape& targetShape,
                                                      uint8_t pad_value,
                                                      bool brg2rgb,
                                                      const std::vector<float>& mean,
                                                      const std::vector<float>& scale,
                                                      const std::type_info& dtype) {
    ov::preprocess::PrePostProcessor ppp(model);

    // Change the input type to the 8-bit image
    if (dtype == typeid(int)) {
        ppp.input(inputName).tensor().set_element_type(ov::element::u8);
    }

    ppp.input(inputName).tensor().set_layout(ov::Layout("NHWC")).set_color_format(ov::preprocess::ColorFormat::BGR);

    if (resize_mode != NO_RESIZE) {
        ppp.input(inputName).tensor().set_spatial_dynamic_shape();
        // Doing resize in u8 is more efficient than FP32 but can lead to slightly different results
        ppp.input(inputName).preprocess().custom(
            createResizeGraph(resize_mode, targetShape, interpolationMode, pad_value));
    }

    ppp.input(inputName).model().set_layout(ov::Layout(layout));

    // Handle color format
    if (brg2rgb) {
        ppp.input(inputName).preprocess().convert_color(ov::preprocess::ColorFormat::RGB);
    }

    ppp.input(inputName).preprocess().convert_element_type(ov::element::f32);

    if (!mean.empty()) {
        ppp.input(inputName).preprocess().mean(mean);
    }
    if (!scale.empty()) {
        ppp.input(inputName).preprocess().scale(scale);
    }

    return ppp.build();
}

std::shared_ptr<InternalModelData> BaseModel::preprocess(const InputData& inputData, InferenceInput& input) {
    const auto& origImg = inputData.asRef<ImageInputData>().inputImage;
    auto img = inputTransform(origImg);

    if (!useAutoResize && !embedded_processing) {
        // Resize and copy data from the image to the input tensor
        auto tensorShape =
            inferenceAdapter->getInputShape(inputNames[0]).get_max_shape();  // first input should be image
        const ov::Layout layout("NHWC");
        const size_t width = tensorShape[ov::layout::width_idx(layout)];
        const size_t height = tensorShape[ov::layout::height_idx(layout)];
        const size_t channels = tensorShape[ov::layout::channels_idx(layout)];
        if (static_cast<size_t>(img.channels()) != channels) {
            throw std::runtime_error("The number of channels for model input: " + std::to_string(channels) +
                                     " and image: " + std::to_string(img.channels()) + " - must match");
        }
        if (channels != 1 && channels != 3) {
            throw std::runtime_error("Unsupported number of channels");
        }
        img = resizeImageExt(img, width, height, resizeMode, interpolationMode);
    }
    input.emplace(inputNames[0], wrapMat2Tensor(img));
    return std::make_shared<InternalImageModelData>(origImg.cols, origImg.rows);
}

std::vector<std::string> BaseModel::loadLabels(const std::string& labelFilename) {
    std::vector<std::string> labelsList;

    /* Read labels (if any) */
    if (!labelFilename.empty()) {
        std::ifstream inputFile(labelFilename);
        if (!inputFile.is_open())
            throw std::runtime_error("Can't open the labels file: " + labelFilename);
        std::string label;
        while (std::getline(inputFile, label)) {
            labelsList.push_back(label);
        }
        if (labelsList.empty())
            throw std::logic_error("File is empty: " + labelFilename);
    }

    return labelsList;
}
