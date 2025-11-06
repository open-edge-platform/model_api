# Segment Anything example

This example demonstrates how to use a Python API implementation of Segment Anything pipeline inference:

- Create encoder and decoder models
- Create a visual prompter pipeline
- Use points as prompts
- Visualized result is saved to `sam_result.jpg`

## Prerequisites

Install Model API from source. Please refer to the main [README](../../../README.md) for details.

## Run example

To run the example, please execute the following command:

```bash
python run.py <path_to_image> <encoder_path> <decoder_path> <prompts>
```

where prompts are in X Y format.

To run the pipeline out-of-the box you can download the test data by running the following command from the repo root:

```bash
pip install httpx
python tests/accuracy/download_models.py -d data -j tests/accuracy/public_scope.json -l
```

and then run

```bash
python run.py ../../../data/coco128/images/train2017/000000000127.jpg \
     ../../../data/otx_models/sam_vit_b_zsl_encoder.xml ../../../data/otx_models/sam_vit_b_zsl_decoder.xml \
     274 306 482 295
```

from the sample folder. Here two prompt poinst are passed via CLI: `(274, 306)` and `(482, 295)`

> _NOTE_: results of segmentation models are saved to `sam_result.jpg` file.
