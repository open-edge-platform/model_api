---
license: agpl-3.0
license_link: https://github.com/ultralytics/ultralytics/blob/main/LICENSE
base_model:
  - Ultralytics/YOLO11
base_model_relation: quantized

---
# YOLOv11-Detection-n-int8-ov

* Model creator: [Ultralytics](https://huggingface.co/Ultralytics)
* Original model: [Ultralytics/YOLO11](https://huggingface.co/Ultralytics/YOLO11)

## Description

This is the **Ultralytics YOLO11n** object detection model exported to **OpenVINO™ IR** (Intermediate Representation) format with **INT8** quantization.

The model files in this directory:

* `yolo11n.xml` – network topology
* `yolo11n.bin` – weights
* `metadata.yaml` – dataset/task metadata (COCO labels, input size, export arguments)


> Note: this export was generated **without NMS** (`nms: false`), so post-processing (NMS) must be applied by the application or an inference wrapper.

## Quantization Parameters

The model was exported with the Ultralytics OpenVINO exporter using INT8 post-training quantization.

Export arguments (from `metadata.yaml` / Ultralytics defaults):

* `format`: **openvino**
* `int8`: **true**
* `imgsz`: **640x640**
* `batch`: **1**
* `dynamic`: **false**
* `half`: **false**
* `nms`: **false**
* `fraction`: **1.0** (calibration subset fraction)

For reference, Ultralytics documents OpenVINO export parameters here:

* https://docs.ultralytics.com/integrations/openvino/

## Compatibility

The provided OpenVINO™ IR model is compatible with:

* OpenVINO Runtime **2025.1.0** and higher
* [OpenVINO Model API](https://github.com/open-edge-platform/model_api)

## Running Model Inference with Model API

1. Install dependencies:

```
pip install openvino-model-api
```

2. Run inference:

```
import cv2
from model_api.models import Model
from model_api.visualizer import Visualizer

# 1. Load model to NPU
model = Model.create_model("yolo11n.xml", device="NPU")

# 2. Load image
image = cv2.imread("image.jpg")

# 3. Run inference
result = model(image)

# 4. Visualize and save results
vis = Visualizer().render(image, result)
cv2.imwrite("output.jpg", vis)
```

## Limitations

* The OpenVINO export in this directory is **INT8** and may have small accuracy differences vs FP32.
* Post-processing (NMS) is not embedded in the model graph (`nms: false`).

## Legal information

The original YOLO11 model and related assets are provided by Ultralytics under the **GNU Affero General Public License v3.0 (AGPL-3.0)**, with an alternative commercial/enterprise licensing option offered by Ultralytics.

* License: https://github.com/ultralytics/ultralytics/blob/main/LICENSE
* Licensing options: https://ultralytics.com/license

This OpenVINO IR is a converted form of the original model; use of this model is subject to the terms of the original license.

## Disclaimer

Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. See [Intel’s Global Human Rights Principles](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf). Intel’s products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.
