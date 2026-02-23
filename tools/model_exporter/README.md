# timm Model Exporter for OpenVINO

A Python tool to download timm models from HuggingFace and export them to OpenVINO IR format with support for multiple precision formats (FP32, FP16, INT8 weight-only quantization).

## Features

- **Minimal Configuration**: Only requires `huggingface_repo` in config - all model metadata is auto-detected from timm's `pretrained_cfg`
- **Multiple Precision Formats**: Exports to FP32, FP16, and INT8 (weight-only quantization) by default
- **Automatic Metadata Extraction**: Input shapes, preprocessing parameters (mean, std), and model configuration automatically detected
- **Duplicate Detection**: Automatically skips duplicate repository entries
- **Batch Processing**: Export multiple models in one run with comprehensive error handling
- **Detailed Logging**: Progress tracking and summary reports for all exports

## Installation

1. **Clone or navigate to the repository**:
   ```bash
   cd /home/mgumowsk/model_exporter
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - `torch>=2.0.0` - PyTorch framework
   - `timm>=0.9.0` - PyTorch Image Models library
   - `optimum[openvino]>=1.16.0` - Optimum Intel for OpenVINO export
   - `nncf>=2.7.0` - Neural Network Compression Framework for quantization
   - `huggingface-hub>=0.20.0` - HuggingFace Hub client

## Configuration

The tool uses a simple JSON configuration file (`config.json`) that only requires the HuggingFace repository path for each model:

```json
[
  {
    "huggingface_repo": "timm/mobilenetv2_050.lamb_in1k"
  },
  {
    "huggingface_repo": "timm/efficientnet_b0.ra_in1k"
  }
]
```

All other model properties (input size, preprocessing parameters, etc.) are automatically detected from the timm model's `pretrained_cfg`.

## Usage

### Basic Usage

Export all models with all formats (FP32, FP16, INT8-wo):

```bash
python export_models.py
```

This will:
- Read models from `config.json`
- Export each model in 3 formats: `fp32`, `fp16`, `int8-wo`
- Save to `./output/{model_name}-{format}/`
- Generate metadata JSON for each export

### Export Specific Format

Export only FP16 models:

```bash
python export_models.py --weight-format fp16
```

Export only INT8 weight-only quantized models:

```bash
python export_models.py --weight-format int8-wo
```

### Custom Batch Size

Export with batch size 4 instead of default 1:

```bash
python export_models.py --batch-size 4
```

### Custom Config and Output Directory

```bash
python export_models.py --config models.json --output-dir ./exported_models
```

### All Options Combined

```bash
python export_models.py \
  --config my_models.json \
  --output-dir /path/to/output \
  --batch-size 8 \
  --weight-format fp16
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | string | `config.json` | Path to configuration JSON file |
| `--output-dir` | string | `./output` | Output directory for exported models |
| `--batch-size` | int | `1` | Batch size for model export (static shape) |
| `--weight-format` | choice | _all formats_ | Export only this format: `fp32`, `fp16`, or `int8-wo` |

## Output Structure

Each exported model is saved in a separate directory with the naming convention: `{model_name}-{format}/`

Example output structure:

```
output/
├── mobilenetv2_050.lamb_in1k-fp32/
│   ├── openvino_model.xml
│   ├── openvino_model.bin
│   └── metadata.json
├── mobilenetv2_050.lamb_in1k-fp16/
│   ├── openvino_model.xml
│   ├── openvino_model.bin
│   └── metadata.json
├── mobilenetv2_050.lamb_in1k-int8-wo/
│   ├── openvino_model.xml
│   ├── openvino_model.bin
│   └── metadata.json
└── export_results.json
```

### Metadata File

Each export includes a `metadata.json` file with auto-detected model configuration:

```json
{
  "input_size": [3, 224, 224],
  "input_shape": [1, 3, 224, 224],
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "num_classes": 1000,
  "interpolation": "bicubic",
  "crop_pct": 0.875,
  "batch_size": 1
}
```

### Export Results

The tool generates an `export_results.json` file with a complete summary:

```json
{
  "timestamp": "2026-01-15T10:30:45.123456",
  "duration_seconds": 1234.56,
  "config": {
    "batch_size": 1,
    "formats": ["fp32", "fp16", "int8-wo"],
    "total_models": 10
  },
  "results": {
    "successful": [...],
    "failed": [...]
  }
}
```

## How It Works

1. **Load Configuration**: Reads `config.json` and deduplicates repository entries
2. **Extract Model Name**: Extracts short name from repository path (e.g., `timm/mobilenetv2_050.lamb_in1k` → `mobilenetv2_050.lamb_in1k`)
3. **Load timm Model**: Downloads and loads the model using `timm.create_model()`
4. **Auto-detect Metadata**: Extracts all configuration from `model.pretrained_cfg`:
   - Input size (C, H, W)
   - Preprocessing mean and std values
   - Number of classes
   - Interpolation method
   - Crop percentage
5. **Export to OpenVINO**: Converts to OpenVINO IR format using `optimum-intel`
   - FP32: Standard conversion
   - FP16: Half-precision conversion
   - INT8-wo: Weight-only quantization
6. **Save Outputs**: Saves model files and metadata to `{model_name}-{format}/` directory

## Precision Formats

- **FP32**: Full 32-bit floating point precision (largest file size, highest accuracy)
- **FP16**: Half precision (smaller file size, minimal accuracy loss on most hardware)
- **INT8-wo**: 8-bit weight-only quantization (smallest file size, good accuracy with faster inference)

## Error Handling

The tool includes comprehensive error handling:

- Skips duplicate repository entries with warnings
- Continues processing remaining models if one fails
- Logs detailed error messages for debugging
- Generates summary report showing successful and failed exports

## Example Session

```bash
$ python export_models.py --batch-size 1

2026-01-15 10:30:00 - INFO - Loading configuration from config.json
2026-01-15 10:30:00 - INFO - Loaded 10 unique models from config
2026-01-15 10:30:00 - INFO - Starting export process
2026-01-15 10:30:00 - INFO - Output directory: output
2026-01-15 10:30:00 - INFO - Batch size: 1
2026-01-15 10:30:00 - INFO - Formats to export: fp32, fp16, int8-wo
2026-01-15 10:30:00 - INFO - Total models: 10
--------------------------------------------------------------------------------

[1/10] Processing: mobilenetv2_050.lamb_in1k
2026-01-15 10:30:00 - INFO - Repository: timm/mobilenetv2_050.lamb_in1k
2026-01-15 10:30:01 - INFO - Exporting mobilenetv2_050.lamb_in1k to fp32...
2026-01-15 10:30:15 - INFO - ✓ Successfully exported mobilenetv2_050.lamb_in1k-fp32
...

================================================================================
EXPORT SUMMARY
================================================================================
Total time: 0:20:45
Successful exports: 30/30
Failed exports: 0/30

✓ SUCCESSFUL EXPORTS:
  - mobilenetv2_050.lamb_in1k-fp32: output/mobilenetv2_050.lamb_in1k-fp32
  - mobilenetv2_050.lamb_in1k-fp16: output/mobilenetv2_050.lamb_in1k-fp16
  ...

Detailed results saved to: output/export_results.json
================================================================================
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors, try:
- Reducing batch size: `--batch-size 1`
- Exporting one format at a time
- Clearing GPU cache between runs

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Model Download Issues

If HuggingFace downloads fail:
- Check internet connection
- Verify repository names in config.json
- Ensure you have access to the models (all timm models in the default config are public)

## License

This tool is provided as-is for exporting timm models to OpenVINO format.

## References

- [timm Documentation](https://huggingface.co/docs/timm)
- [OpenVINO Optimum Intel](https://huggingface.co/docs/optimum-intel/en/openvino/export)
- [HuggingFace Hub](https://huggingface.co/timm)
