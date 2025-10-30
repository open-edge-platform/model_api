# OpenVINO Model API

Model API is a set of wrapper classes for particular tasks and model architectures, simplifying data preprocess and postprocess as well as routine procedures (model loading, asynchronous execution, etc.). It is aimed at simplifying end-to-end model inference for different deployment scenarious, including local execution and serving. The Model API is based on the OpenVINO inference API.

## How it works

Model API searches for additional information required for model inference, data, pre/postprocessing, label names, etc. directly in OpenVINO Intermediate Representation. This information is used to prepare the inference data, process and output the inference results in a human-readable format.

## Features

- Python and C++ API
- Automatic prefetch of public models from [OpenVINO Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) (Python only)
- Synchronous and asynchronous inference
- Local inference and serving through the rest API (Python only)
- Model preprocessing embedding for faster inference

## Installation

### Python

- Clone this repository
- Navigate to `model_api/python` folder
- Run `pip install .`

### C++

- Install dependencies. For installation on Ubuntu, you can use the following script:

  ```bash
  chmod +x model_api/cpp/install_dependencies.sh
  sudo model_api/cpp/install_dependencies.sh
  ```

- Build library:
  - Create `build` folder and navigate into it:

  ```bash
  mkdir build && cd build
  ```

  - Run cmake:

  ```bash
  cmake ../model_api/cpp -DOpenCV_DIR=<OpenCV cmake dir> -DOpenVINO_DIR=<OpenVINO cmake dir>
  ```

  - Build:

  ```bash
  cmake --build . -j
  ```

  - To build a `.tar.gz` package with the library, run:

  ```bash
  cmake --build . --target package -j
  ```

## Usage

### Python

```python
from model_api.models import DetectionModel

# Create a model (downloaded and cached automatically for OpenVINO Model Zoo models)
# Use URL to work with served model, e.g. "localhost:9000/models/ssdlite_mobilenet_v2"
ssd = DetectionModel.create_model("ssdlite_mobilenet_v2")

# Run synchronous inference locally
detections = ssd(image)  # image is numpy.ndarray

# Print the list of Detection objects with box coordinates, confidence and label string
print(f"Detection results: {detections}")
```

### C++

```cpp
#include <models/detection_model.h>
#include <models/results.h>

// Load the model fetched using Python API
auto model = DetectionModel::create_model("~/.cache/omz/public/ssdlite_mobilenet_v2/FP16/ssdlite_mobilenet_v2.xml");

// Run synchronous inference locally
auto result = model->infer(image); // image is cv::Mat

// Iterate over the vector of DetectedObject with box coordinates, confidence and label string
for (auto& obj : result->objects) {
    std::cout << obj.label << " | " << obj.confidence << " | " << int(obj.x) << " | " << int(obj.y) << " | "
        << int(obj.x + obj.width) << " | " << int(obj.y + obj.height) << std::endl;
}
```

Model's static method `create_model()` has two overloads. One constructs the model from a string (a path or a model name) (shown above) and the other takes an already constructed `InferenceAdapter`.

## Prepare a model for `InferenceAdapter`

There are usecases when it is not possible to modify an internal `ov::Model` and it is hidden behind `InferenceAdapter`. For example the model can be served using [OVMS](https://github.com/openvinotoolkit/model_server). `create_model()` can construct a model from a given `InferenceAdapter`. That approach assumes that the model in `InferenceAdapter` was already configured by `create_model()` called with a string (a path or a model name). It is possible to prepare such model using C++ or Python:
C++

```Cpp
auto model = DetectionModel::create_model("~/.cache/omz/public/ssdlite_mobilenet_v2/FP16/ssdlite_mobilenet_v2.xml");
const std::shared_ptr<ov::Model>& ov_model = model->getModel();
ov::serialize(ov_model, "serialized.xml");
```

Python

```python
model = DetectionModel.create_model("~/.cache/omz/public/ssdlite_mobilenet_v2/FP16/ssdlite_mobilenet_v2.xml")
model.save("serialized.xml")
```

After that the model can be constructed from `InferenceAdapter`:

```cpp
ov::Core core;
std::shared_ptr<ov::Model> ov_model = core.read_model("serialized.xml");
std::shared_ptr<InferenceAdapter> adapter = std::make_shared<OpenVINOInferenceAdapter>();
adapter->loadModel(ov_model, core);
auto model = DetectionModel::create_model(adapter);
```

For more details please refer to the [examples](https://github.com/openvinotoolkit/model_api/tree/master/examples) of this project.

## Supported models

### Python

- Image Classification:
  - [OpenVINO Model Zoo models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md#classification-models)
- Object Detection:
  - [OpenVINO Model Zoo models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md#object-detection-models):
    - SSD-based models (e.g. "ssdlite_mobilenet_v2", etc.)
    - YOLO-based models (e.g. "yolov3", "yolov4", etc.)
    - CTPN: "ctpn"
    - DETR: "detr-resnet50"
    - CenterNet: "ctdet_coco_dlav0_512"
    - FaceBoxes: "faceboxes-pytorch"
    - RetinaFace: "retinaface-resnet50-pytorch"
    - Ultra Lightweight Face Detection: "ultra-lightweight-face-detection-rfb-320" and "ultra-lightweight-face-detection-slim-320"
    - NanoDet with ShuffleNetV2: "nanodet-m-1.5x-416"
    - NanoDet Plus with ShuffleNetV2: "nanodet-plus-m-1.5x-416"
- Semantic Segmentation:
  - [OpenVINO Model Zoo models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md#semantic-segmentation-models)
- Instance Segmentation:
  - [OpenVINO Model Zoo models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md#instance-segmentation-models)

### C++

- Image Classification:
  - [OpenVINO Model Zoo models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md#classification-models)
- Object Detection:
  - SSD-based models (e.g. "ssdlite_mobilenet_v2", etc.)
    - YOLO-based models (e.g. "yolov3", "yolov4", etc.)
    - CenterNet: "ctdet_coco_dlav0_512"
    - FaceBoxes: "faceboxes-pytorch"
    - RetinaFace: "retinaface-resnet50-pytorch"
- Semantic Segmentation:
  - [OpenVINO Model Zoo models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md#semantic-segmentation-models)

[Model configuration](https://github.com/openvinotoolkit/model_api/blob/master/docs/model-configuration.md) discusses possible configurations.
