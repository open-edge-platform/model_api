# Model Converter Tool

A command-line utility to download PyTorch models and convert them to OpenVINO format.

## Overview

This tool reads a JSON configuration file containing model specifications, downloads PyTorch weights from URLs, loads the models, and exports them to OpenVINO Intermediate Representation (IR) format.

## Features

- **Automatic Download**: Downloads model weights from HTTP/HTTPS URLs with caching support
- **Dynamic Model Loading**: Dynamically imports and instantiates model classes from Python paths
- **Metadata Embedding**: Embeds custom metadata into OpenVINO models
- **Input/Output Naming**: Configurable input and output tensor names
- **Batch Processing**: Process multiple models from a single configuration file
- **Selective Conversion**: Convert specific models using the `--model` flag
- **Summary Report**: Generate a console + Markdown conversion report with the `--report` flag

## Installation

### Prerequisites

```bash
# Required packages
uv sync

```

## Usage

### Basic Usage

```bash
uv run model-converter examples/config.json -o ./output_models
```

### Command-Line Options

```text
positional arguments:
  config                Path to JSON configuration file

options:
  -h, --help            Show help message and exit
  -o OUTPUT, --output OUTPUT
                        Output directory for converted models (default: ./converted_models)
  -c CACHE, --cache CACHE
                        Cache directory for downloaded weights (default: ~/.cache/torch/hub/checkpoints)
  --model MODEL         Process only the specified model (by model_short_name)
  --library LIBRARY     Comma-separated list of model libraries to process (e.g., getitune,timm)
  --report [PATH]       Generate a conversion summary report. Pass the flag alone to write to
                        <output>/conversion_report.md, or provide a PATH to override the location.
                        The report is printed to the console and saved as Markdown.
  --list                List all models in the configuration file and exit
  -v, --verbose         Enable verbose logging
```

### Examples

**List all models in configuration:**

```bash
uv run model-converter examples/config.json --list
```

**Convert all models:**

```bash
uv run model-converter examples/config.json -o ./converted_models
```

**Convert a specific model:**

```bash
uv run model-converter examples/config.json -o ./converted_models --model resnet50
```

**Use custom cache directory:**

```bash
uv run model-converter examples/config.json -o ./output -c ./my_cache
```

**Enable verbose logging:**

```bash
uv run model-converter examples/config.json -o ./output -v
```

**Generate a conversion summary report:**

```bash
# Write the report to <output>/conversion_report.md
uv run model-converter examples/config.json -o ./output --report

# Write the report to a custom path
uv run model-converter examples/config.json -o ./output --report ./reports/summary.md
```

## Conversion Summary Report

Passing `--report` produces a summary of every processed model. The report is
printed to the console **and** saved as a Markdown file (default
`<output>/conversion_report.md`, or the path given to `--report`).

The report contains one row per model with the following columns:

| Column            | Description                                                                                       |
| ----------------- | ------------------------------------------------------------------------------------------------- |
| Model Full Name   | Human-readable model name (`model_full_name`).                                                    |
| Model Type        | Task/architecture type (e.g. `Classification`, `SSD`).                                            |
| Model Library     | Source library (`torchvision`, `timm`, `yolo`, `getitune`).                                       |
| Original URL      | Exact download URL — `weights_url` or the Hugging Face repository URL; `N/A` when not applicable. |
| Original Accuracy | Top-1 accuracy of the original PyTorch model (before OpenVINO conversion), or `N/A`.              |
| FP32 Accuracy     | Top-1 accuracy of the FP32 model, or `N/A` when not measured.                                     |
| FP16 Accuracy     | Top-1 accuracy of the FP16 model, or `N/A` when not measured.                                     |
| INT8 Accuracy     | Top-1 accuracy of the quantized INT8 model, or `N/A` when not measured.                           |
| Status            | Outcome of the conversion (see below).                                                            |

Accuracy is measured only for classification models that define `labels`, over the
calibration subset. The original accuracy is computed by running the source PyTorch
model on the same preprocessed validation images, so it is directly comparable to the
FP32/FP16/INT8 numbers. Models without measurable accuracy report `N/A` and a status of
`OK (no accuracy data)`.

### Status values

| Status                  | Meaning                                                                            |
| ----------------------- | ---------------------------------------------------------------------------------- |
| `OK`                    | Converted, accuracy measured, all drops within 5%.                                 |
| `OK (no accuracy data)` | Converted (FP16 + INT8) but no accuracy could be measured.                         |
| `ACCURACY DROP >5%`     | Original→FP32, FP16, or INT8 top-1 accuracy dropped more than 5 percentage points. |
| `FAILED: conversion`    | Model load or export failed (no FP16 produced).                                    |
| `FAILED: quantization`  | FP16 produced but the INT8 model was not produced.                                 |
| `SKIPPED`               | FP16 and INT8 models already existed, so processing was skipped.                   |

### Sample report

```markdown
# Conversion Summary Report

| Model Full Name | Model Type     | Model Library | Original URL                                     | Original Accuracy | FP32 Accuracy | FP16 Accuracy | INT8 Accuracy | Status                |
| --------------- | -------------- | ------------- | ------------------------------------------------ | ----------------- | ------------- | ------------- | ------------- | --------------------- |
| ResNet-50       | Classification | torchvision   | https://download.pytorch.org/models/resnet50.pth | 80.20%            | 80.12%        | 80.10%        | 79.85%        | OK                    |
| EfficientNet-B0 | Classification | timm          | https://huggingface.co/timm/efficientnet_b0      | 77.72%            | 77.70%        | 77.65%        | 70.10%        | ACCURACY DROP >5%     |
| YOLO11n         | YOLO11         | yolo          | N/A                                              | N/A               | N/A           | N/A           | N/A           | OK (no accuracy data) |
```

## Configuration File Format

The configuration file is a JSON file with the following structure:

```json
{
  "models": [
    {
      "model_short_name": "resnet50",
      "license": "bsd-3-clause",
      "license_link": "https://spdx.org/licenses/BSD-3-Clause.html",
      "model_class_name": "torchvision.models.resnet.resnet50",
      "model_full_name": "ResNet-50",
      "description": "ResNet-50 image classification model",
      "weights_url": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
      "input_shape": [1, 3, 224, 224],
      "input_names": ["images"],
      "output_names": ["output"],
      "model_params": null,
      "model_type": "Classification"
    }
  ]
}
```

**Important**: The `model_type` field enables automatic model detection when using [Intel's model_api](https://github.com/openvinotoolkit/model_api). When specified, this metadata is embedded in the OpenVINO IR, allowing `Model.create_model()` to automatically select the correct model wrapper class.

Common `model_type` values:

- `"Classification"` - Image classification models
- `"DetectionModel"` - Object detection models
- `"YOLOX"` - YOLOX detection models
- `"SegmentationModel"` - Segmentation models

### Configuration Fields

#### Required Fields

- **`model_short_name`** (string): Short identifier for the model (used for output filename)
- **`license`** (string): SPDX license identifier for the upstream model (for example `bsd-3-clause` or `apache-2.0`)
- **`license_link`** (string): URL to the upstream license text used in generated README files
- **`model_class_name`** (string): Full Python path to the model class (e.g., `torchvision.models.resnet.resnet50`)
- **`weights_url`** (string): URL to download the PyTorch weights (.pth file)

For Hugging Face-backed models, use these required fields instead of `model_class_name` / `weights_url`:

- **`huggingface_repo`** (string): Hugging Face repository ID (for example `timm/mobilenetv2_100.ra_in1k`)
- **`huggingface_revision`** (string): Immutable commit SHA to pin the download and model load to a specific repository state

#### Optional Fields

- **`model_full_name`** (string): Full descriptive name of the model
- **`description`** (string): Description of the model
- **`docs`** (string): Documentation URL for the model
- **`input_shape`** (array of integers): Input tensor shape (default: `[1, 3, 224, 224]`)
- **`input_names`** (array of strings): Names for input tensors (default: `["input"]`)
- **`output_names`** (array of strings): Names for output tensors (default: auto-generated)
- **`model_params`** (object): Parameters to pass to model constructor (default: `null`)
- **`model_type`** (string): Model type for model_api auto-detection (e.g., `"Classification"`, `"DetectionModel"`, `"YOLOX"`, etc.)
