# Accuracy Test Comparator

## Overview

The comparator framework provides a compact way to verify model inference results against reference data. It replaces large JSON dumps with a combination of compact statistical fingerprints and binary-mask artifacts.

The framework supports 4 comparison policies:

- **EXACT**: Used for discrete values like labels and class names.
- **NUMERIC_CLOSE**: Used for scalar scores and bounding box coordinates.
- **MASK_IOU**: Used for spatial masks and class maps.
- **STAT_FINGERPRINT**: Used for large floating-point arrays such as saliency or anomaly maps.

Use **EXACT** for any field where even a single bit change is a failure. Use **NUMERIC_CLOSE** for floating point values where small device-specific variations are expected. Use **MASK_IOU** for any spatial output. Use **STAT_FINGERPRINT** when you need to verify the distribution of values in a large array without storing the full array.

## Policies

| Policy           | Used For                                         | Key Parameters                | Failure Condition                                        |
| ---------------- | ------------------------------------------------ | ----------------------------- | -------------------------------------------------------- |
| EXACT            | string fields, labels, class names               | None                          | `actual != reference`                                    |
| NUMERIC_CLOSE    | scalar scores, bbox coords                       | `atol`, `rtol`                | `abs(actual - reference) > atol + rtol * abs(reference)` |
| MASK_IOU         | binary masks, class maps, instance masks         | `iou_threshold` (default 0.9) | `IoU < threshold`                                        |
| STAT_FINGERPRINT | large float arrays (saliency maps, anomaly maps) | `atol`, `rtol`, `sample_atol` | any statistical sub-check fails                          |

## Reference Directory Layout

Reference artifacts are stored in the following structure:

```text
tests/accuracy/references/<reference_dir>/
  result.json          # scalar fields + mask file paths
  metadata.json        # test_id, policy_overrides
  generated_by.json    # platform metadata (OS, OpenVINO version, ModelAPI version, Python version)
  masks/
    <field_name>.png   # binary mask (L mode) or class map (I;16 PNG)
    <field_name>.npz   # instance masks (stacked bool array)
```

## Regenerating References

To update the reference artifacts, run pytest with the `--update-references` flag. This requires the test data to be present.

```bash
# Generate references for all configs (requires --data)
uv --directory model_api run pytest --data=/path/to/data tests/accuracy/test_accuracy.py --update-references

# CI guard: this will FAIL when CI=true
CI=true uv --directory model_api run pytest --data=/path/to/data tests/accuracy/test_accuracy.py --update-references

# Override (use with caution):
CI=true uv --directory model_api run pytest --data=/path/to/data tests/accuracy/test_accuracy.py --update-references --force-update-in-ci
```

## Failure Messages

Failure messages indicate which field failed, the policy used, and the magnitude of the deviation.

- For numeric fields: `pred_score: delta=0.023 > atol=0.01 (130% over budget)`
- For masks: `pred_mask IoU=0.82 < threshold=0.90`

Each failed field will show both the actual and reference summaries to help diagnose the cause.

## Policy Overrides

You can override default policies and tolerances for specific tests by editing the `metadata.json` file in the reference directory.

Example `metadata.json` snippet:

```json
{
  "test_id": "classification_model::image.jpg",
  "policy_overrides": {
    "scores": {
      "policy": "numeric_close",
      "atol": 0.05
    }
  }
}
```

## Noise Floor

The noise floor represents the inherent variance in inference results when running the same model on the same input multiple times. This variance can be caused by non-deterministic operations in certain layers or hardware-specific behavior.

To measure the noise floor for a model:

```bash
uv --directory model_api run python -m tests.accuracy.comparator.noise_floor --help
```

Fields with a coefficient of variation (cv) greater than 0.01 are considered "noisy" and may require wider tolerances in their comparison policies.


## Testing comparator

Run the comparator tests with:

```bash
uv --directory model_api run pytest tests/accuracy/comparator
```