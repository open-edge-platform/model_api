# Synchronous API example

This example demonstrates how to use a Python API of OpenVINO Model API for synchronous inference as well as basic features such as:

- Automatic download and convertion of public models
- Preprocessing embedding
- Creating model from local source
- Image Classification, Object Detection and Semantic Segmentation use cases

## Prerequisites

Install Model API from source. Please refer to the main [README](../../../README.md) for details.

## Run example

To run the example, please execute the following command:

```bash
python run.py <path_to_image>
```

> _NOTE_: results of Semantic Segmentation models are saved to `mask.png` file.
