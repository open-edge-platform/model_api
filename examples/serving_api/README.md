# Serving API example

This example demonstrates how to use a Python API of OpenVINO Model API for a remote inference of models hosted with [OpenVINO Model Server](https://docs.openvino.ai/latest/ovms_what_is_openvino_model_server.html). This tutorial assumes that you are familiar with Docker subsystem and includes the following steps:

- Run Docker image with
- Instantiate a model
- Run inference
- Process results

## Prerequisites

- Install Model API from source. Please refer to the main [README](../../../README.md) for details.
- Install Docker. Please refer to the [official documentation](https://docs.docker.com/get-docker/) for details.
- Install OVMS client into the Python environment:

  ```bash
  pip install ovmsclient
  ```

- Download a model by running a Python code with Model API, see Python [exaple](../../synchronous_api/README.md) and resave a configured model at OVMS friendly folder layout:

  ```python
  from model_api.models import DetectionModel

  DetectionModel.create_model("ssd_mobilenet_v1_fpn_coco").save("/home/user/models/ssd_mobilenet_v1_fpn_coco/1/ssd_mobilenet_v1_fpn_coco.xml")
  ```

- Run docker with OVMS server:

  ```bash
  docker run -d -v /home/user/models:/models -p 8000:8000 openvino/model_server:latest --model_path /models/ssd_mobilenet_v1_fpn_coco --model_name ssd_mobilenet_v1_fpn_coco --rest_port 8000 --nireq 4 --target_device CPU
  ```

## Run example

To run the example, please execute the following command:

```bash
python run.py <path_to_image>
```
