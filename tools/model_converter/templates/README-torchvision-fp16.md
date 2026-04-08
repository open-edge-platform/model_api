---
license: <<license>>
tags:
  - image-classification
  - vision
---

# <<model_name>>

- Model creator: [torchvision](https://github.com/pytorch/vision)
- Original model: [<<model_name>>](<<docs>>)

## Description

This is a torchvision version of [<<model_name>>](<<docs>>) model converted to the [OpenVINO™ IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html) (Intermediate Representation) format with weights compressed to FP16.

## Compatibility

The provided OpenVINO™ IR model is compatible with:

- OpenVINO version 2025.4.0 and higher
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
model = Model.from_pretrained("OpenVINO/<<model_name>>")

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

Check the original [model implementation](https://github.com/pytorch/vision) for limitations.

## Legal information

The original model is distributed under the [<<license>>](<<license_link>>) license. More details can be found in [https://github.com/pytorch/vision](https://github.com/pytorch/vision).

## Disclaimer

Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. See [Intel’s Global Human Rights Principles](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf). Intel’s products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.
