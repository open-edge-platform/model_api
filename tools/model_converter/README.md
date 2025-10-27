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

## Installation

### Prerequisites

```bash
# Required packages
uv pip install torch torchvision openvino

```

## Usage

### Basic Usage

```bash
python model_converter.py config.json -o ./output_models
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
  --list                List all models in the configuration file and exit
  -v, --verbose         Enable verbose logging
```

### Examples

**List all models in configuration:**

```bash
python model_converter.py example_config.json --list
```

**Convert all models:**

```bash
python model_converter.py example_config.json -o ./converted_models
```

**Convert a specific model:**

```bash
python model_converter.py example_config.json -o ./converted_models --model resnet50
```

**Use custom cache directory:**

```bash
python model_converter.py example_config.json -o ./output -c ./my_cache
```

**Enable verbose logging:**

```bash
python model_converter.py example_config.json -o ./output -v
```

## Configuration File Format

The configuration file is a JSON file with the following structure:

```json
{
  "models": [
    {
      "model_short_name": "resnet50",
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
- **`model_class_name`** (string): Full Python path to the model class (e.g., `torchvision.models.resnet.resnet50`)
- **`weights_url`** (string): URL to download the PyTorch weights (.pth file)

#### Optional Fields

- **`model_full_name`** (string): Full descriptive name of the model
- **`description`** (string): Description of the model
- **`input_shape`** (array of integers): Input tensor shape (default: `[1, 3, 224, 224]`)
- **`input_names`** (array of strings): Names for input tensors (default: `["input"]`)
- **`output_names`** (array of strings): Names for output tensors (default: auto-generated)
- **`model_params`** (object): Parameters to pass to model constructor (default: `null`)
- **`model_type`** (string): Model type for model_api auto-detection (e.g., `"Classification"`, `"DetectionModel"`, `"YOLOX"`, etc.)
