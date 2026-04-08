---
license: agpl-3.0
tags:
  - object-detection
  - vision
base_model:
  - ultralytics/yolo11
base_model_relation: quantized
---

# YOLO11<<yolo_size>>-fp16-ov

- Model creator: [Ultralytics](https://huggingface.co/Ultralytics)
- Original model: [Ultralytics/YOLO11](https://huggingface.co/Ultralytics/YOLO11)

## Description

This is [https://huggingface.co/Ultralytics/YOLO11](https://huggingface.co/Ultralytics/YOLO11) model converted to the [OpenVINO™ IR](https://docs.openvino.ai/2026/documentation/openvino-ir-format.html) (Intermediate Representation) format. with weights compressed to FP16.

## Compatibility

The provided OpenVINO™ IR model is compatible with:

- OpenVINO version 2026.1.0 and higher
- Model API 0.4.0 and higher

## Running Model Inference with [Model API](https://github.com/open-edge-platform/model_api)

1. Install required packages:

```python
pip install openvino-model-api[huggingface]
```

<!-- markdownlint-disable MD029 -->

2. Run model inference:

```python
import cv2
from model_api.models import Model
from model_api.visualizer import Visualizer

# 1. Load model
model = Model.from_pretrained("OpenVINO/YOLO11<<yolo_size>>-fp16-ov")

# 2. Load image
image = cv2.imread("image.jpg")

# 3. Run inference
result = model(image)

# 4. Visualize and save results
vis = Visualizer().render(image, result)
cv2.imwrite("output.jpg", vis)
```

For more examples and possible optimizations, refer to the [Model API Documentation](https://open-edge-platform.github.io/model_api/latest/).

## Limitations

Check the original [model card](https://huggingface.co/Ultralytics/YOLO11) for limitations.

## Legal information

The original model is distributed under [GNU Affero General Public License v3.0](https://choosealicense.com/licenses/agpl-3.0/) license. More details can be found in [Ultralytics/YOLO11](https://huggingface.co/Ultralytics/YOLO11).

## Disclaimer

Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. See [Intel’s Global Human Rights Principles](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf). Intel’s products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.
