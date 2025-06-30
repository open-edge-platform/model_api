# Synchronous API example

This example demonstrates how to use a C++ API of OpenVINO Model API for synchronous inference and its basic steps:

- Instantiate a model
- Model inference
- Process results

## Prerequisites

- Install third party dependencies by running the following script:

  ```bash
  chmod +x ../../../src/cpp/install_dependencies.sh
  sudo ../../../src/cpp/install_dependencies.sh
  ```

- Build example:

  - Create `build` folder and navigate into it:
  <!-- prettier-ignore-start -->

  ```bash
  mkdir build && cd build
  ```

<!-- prettier-ignore-end -->

- Run cmake:
<!-- prettier-ignore-start -->

```bash
cmake ../
```

  <!-- prettier-ignore-end -->

- Build:

```bash
make -j
```

- Download a model by running a Python code with Model API, see Python [example](../../python/synchronous_api/README.md):

  ```python
  from model_api.models import DetectionModel

  model = DetectionModel.create_model("ssd_mobilenet_v1_fpn_coco",
                                  download_dir="tmp")
  ```

## Run example

To run the example, please execute the following command:

```bash
./synchronous_api ./tmp/public/ssd_mobilenet_v1_fpn_coco/FP16/ssd_mobilenet_v1_fpn_coco.xml <path_to_image>
```
