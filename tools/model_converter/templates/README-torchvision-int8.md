---
license: bsd-3-clause
tags:
- image-classification
- vision
---

# {model_name}

* Model creator: [torchvision](https://github.com/pytorch/vision)
* Original model: [{model_name}](https://github.com/pytorch/vision)

## Description

This is a torchvision version of [{model_name}](https://github.com/pytorch/vision) model converted to the [OpenVINO™ IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html) (Intermediate Representation) format with weights compressed to FP16.

## Quantization Parameters

Weight compression was performed using nncf.quantize with the following parameters:

* **Quantization method**: Post-Training Quantization (PTQ)
* **Precision**: INT8 for both weights and activations
* **Calibration dataset**: ImageNet validation subset

For more information on quantization, check the [OpenVINO model optimization guide](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training.html).

## Compatibility

The provided OpenVINO™ IR model is compatible with:

* OpenVINO version 2025.4.0 and higher
* Model API 0.4.0 and higher

## Running Model Inference with [Model API](https://github.com/open-edge-platform/model_api)

1. Install required packages:

```python
pip install openvino-model-api
```

2. Run model inference:

```python
import cv2
from model_api.models import Model
from model_api.visualizer import Visualizer

# 1. Load model
model = Model.create_model("{model_name}.xml", device="AUTO")

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

The original model is distributed under [BSD 3-Clause "New" or "Revised" License](https://choosealicense.com/licenses/bsd-3-clause/) license. More details can be found in [https://github.com/pytorch/vision](https://github.com/pytorch/vision).

## Disclaimer

Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. See [Intel’s Global Human Rights Principles](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf). Intel’s products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.
