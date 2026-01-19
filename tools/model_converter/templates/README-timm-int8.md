---
license: apache-2.0
tags:
- image-classification
- vision
base_model:
- timm/{model_name}
base_model_relation: quantized
---

# {model_name}

* Model creator: [timm](https://huggingface.co/timm)
* Original model: [timm/{model_name}](https://huggingface.co/timm/{model_name})

## Description

This is [https://huggingface.co/timm/{model_name}](https://huggingface.co/timm/{model_name}) model converted to the [OpenVINO™ IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html) (Intermediate Representation) format with weights compressed to INT8 by [NNCF](https://github.com/openvinotoolkit/nncf).

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

Check the original [model card](https://huggingface.co/timm/{model_name}) for limitations.

## Legal information

The original model is distributed under [Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/) license. More details can be found in [timm/{model_name}](https://huggingface.co/timm/{model_name}).

## Disclaimer

Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. See [Intel’s Global Human Rights Principles](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf). Intel’s products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.
