# Model Metadata Specification

A language-agnostic, declarative contract that describes how a model must be
**preprocessed**, **run**, and **postprocessed**, and what its **outputs** mean.

The goal of this specification is to remove `model_type` as the single
discriminator that selects a Python `Model` subclass. Instead, each concern
(preprocessing, inference I/O, postprocessing, output interpretation) is
described by explicit metadata so that **one generic pipeline per `task`**
(classification, detection, instance segmentation, keypoint detection, …) — or
any client that does **not** share the `model_api` codebase (C++, Rust, JS, …) —
can reproduce the exact pipeline from the metadata alone.

**Design goal — one generic pipeline per task.** The aim is *not* a single
universal engine that covers every task with one code path. It is a **separate
generic pipeline per `task`**, where:

- the pipeline MAY implement the task's own algorithms (e.g. the SimCC decode, the
  hierarchical label resolver, the SAM prompt encoding) **once**, inside the
  pipeline;
- the pipeline MUST NOT branch on model identity (`if model_type == …`) — **all
  per-model variation is expressed as metadata values** (`decoder`,
  `mask_postprocess`, `activation`, `decoder_constants`, …) that the pipeline reads;
- multi-model tasks (**top-down keypoints** = detect→crop→pose, **visual prompting**
  = encoder→decoder) are themselves single task pipelines whose fixed topology is
  read from [`stages`](#stages).

Under this goal every current task is metadata-drivable. All models are re-exported
to carry this metadata; the decoders that need **extra care** during that export
are the YOLO family (`YOLO`/`YoloV4`, `YOLOF`, `YOLOX`, `YOLOv3-ONNX`, `YOLO-seg`),
whose wrappers currently derive parameters at load time by **inspecting the
OpenVINO graph or tensor shapes**, or hardcode them (see
[`decoder_constants`](#decoder_constants)). Those values simply have to be
captured explicitly as metadata during the same re-export, since static RT-info
cannot reproduce the introspection.

All properties live under the `model_info` runtime-info block of the model and
are stored as strings (JSON-encoded where a structured value is needed), since
OpenVINO RT-info only persists string values.

---

## Table of Contents

1. [Pipeline stages](#pipeline-stages)
2. [Compatibility properties](#compatibility-properties)
   - [`metadata_schema_version`](#metadata_schema_version)
   - [`task`](#task)
   - [`stages`](#stages)
3. [Output description](#output-description)
   - [Result types (not metadata)](#result-types-not-metadata)
   - [Result schema definitions](#result-schema-definitions)
   - [`labels` / `path_to_labels`](#labels--path_to_labels)
   - [`output_coordinate_space`](#output_coordinate_space)
   - [`output_value_range`](#output_value_range)
   - [`derived_outputs`](#derived_outputs)
4. [Input / output contract](#input--output-contract)
   - [`input_roles`](#input_roles)
   - [`output_roles`](#output_roles)
   - [`output_layout`](#output_layout)
5. [Preprocessing properties](#preprocessing-properties)
   - [`input_modality`](#input_modality)
   - [`input_layout`](#input_layout)
   - [`embedded_processing`](#embedded_processing)
   - [`resize_type`](#resize_type)
   - [`input_dtype` / `interpolation_mode` / `pad_value`](#input_dtype--interpolation_mode--pad_value)
   - [`orig_height` / `orig_width` / `input_frame_height` / `input_frame_width`](#orig_height--orig_width--input_frame_height--input_frame_width)
   - [`intensity_mode`](#intensity_mode)
   - [`mean_values` / `scale_values` / `reverse_input_channels`](#mean_values--scale_values--reverse_input_channels)
   - [`temporal.clip_size`](#temporalclip_size)
6. [Postprocessing properties](#postprocessing-properties)
   - [`activation`](#activation)
   - [`decoder`](#decoder)
   - [`box_format`](#box_format)
   - [`label_shift`](#label_shift)
   - [`mask_postprocess`](#mask_postprocess)
   - [`nms` and NMS parameters](#nms-and-nms-parameters)
   - [Threshold parameters](#threshold-parameters)
   - [Task-specific output options](#task-specific-output-options)
   - [`decoder_constants`](#decoder_constants)
7. [Pipeline wrappers and stateful pipelines](#pipeline-wrappers-and-stateful-pipelines)
   - [`tiling`](#tiling)
   - [`state` (learn / infer pipelines)](#state-learn--infer-pipelines)
8. [Per-model mapping table](#per-model-mapping-table)
   - [Additional wrappers](#additional-wrappers)
9. [Governance notes for external clients](#governance-notes-for-external-clients)
10. [Migration plan for existing Python modules](#migration-plan-for-existing-python-modules)
    - [Classification](#classification-migration)
    - [Segmentation](#segmentation-migration)
    - [Detection](#detection-migration)
    - [Instance Segmentation](#instance-segmentation-migration)
    - [Anomaly](#anomaly-migration)
    - [Keypoint detection](#keypoint-detection-migration)
    - [Action classification](#action-classification-migration)
    - [Visual prompting](#visual-prompting-migration)
    - [Tiling](#tiling-migration)

---

## Pipeline stages

Each property in this document is tagged with the stage(s) it affects:

| Stage | Meaning |
|-------|---------|
| **Compat** | Negotiated before anything runs; decides whether the client can execute the model at all. |
| **Pre** | Preprocessing — transforming the raw input into the model's input tensor(s). |
| **Infer** | Inference — mapping named roles onto the real input/output tensors. |
| **Post** | Postprocessing — decoding raw output tensors into a structured result. |
| **Output** | Interpretation of the final result values (units, ranges, coordinate frames). |

---

## Compatibility properties

Compatibility is negotiated on the pair **(`metadata_schema_version`, `task`)**
plus the detailed per-stage fields.

### `metadata_schema_version`

- **Stage:** Compat
- **Type:** string (semver, e.g. `"1.0"`)
- **Possible values:** any valid semantic version.
- **Description:** Version of this specification that the metadata conforms to.
- **What it does:** A client compares the **major** version against the versions
  it implements. If the major is unknown, the client MUST refuse to run the model
  rather than guess. Minor/patch increments are backward compatible (new optional
  fields only).
- **Used by:** **All models.** Every model carries it so any client can decide,
  before touching tensors, whether it understands the metadata vocabulary at all.
- **Required:** Yes.

### `task`

- **Stage:** Compat / Output
- **Type:** enum string
- **Possible values:** `classification`, `segmentation`, `detection`,
  `instance_segmentation`, `anomaly`, `keypoints`, `action_classification`,
  `visual_prompting`.
- **Description:** The semantic family of the model. Determines which task
  pipeline runs and therefore which result type is returned (see
  [Result types](#result-types-not-metadata)).
- **What it does:** Selects the output data structure and the high-level contract,
  and is the primary token a client negotiates on. It does **not** by itself fix
  the decoder or topology — those come from `decoder`, `stages`, and the other
  explicit fields. Replaces the heuristic in `Model.detect_model_type` (e.g.
  detecting anomaly from output names).
- **Used by:** **All models**, with a distinct value each: Classification →
  `classification`, Action Cls → `action_classification`, Segmentation →
  `segmentation`, SSD/YOLO family/YOLO-DETR → `detection`,
  MaskRCNN/DETRInstSeg/YOLO-seg → `instance_segmentation`, Anomaly → `anomaly`,
  Keypoint → `keypoints`, SAM (incl. ZSL) → `visual_prompting`. Needed because it
  is the primary capability-negotiation token and picks the result family.
- **Required:** Yes.

### `stages`

- **Stage:** Compat / Infer
- **Type:** JSON array of stage objects
- **Possible values:** an ordered list, each element describing one inference
  stage. A single-stage model has exactly one element. Each stage object has:
  - `name` (string) — e.g. `encoder`, `decoder`, `detector`, `pose`.
  - `role` (enum) — `main`, `encoder`, `decoder`, `proposal`, `refine`.
  - `input_roles` / `output_roles` (objects) — role→tensor-name maps scoped to the
    stage (see [Input / output contract](#input--output-contract)).
  - `consumes` (array, optional) — names of earlier stages whose outputs feed this
    stage (e.g. the SAM decoder `consumes: ["encoder"]`).
- **Description:** Explicit description of pipeline topology. Replaces the
  structural information that a single profile name used to imply.
- **What it does:** Tells the client how many models to run and how to wire them.
  Single-stage tasks (classification, detection, segmentation, anomaly, keypoints,
  action) declare one stage. Multi-stage pipelines (SAM visual prompting =
  `encoder`→`decoder`; top-down keypoints = `detector`→`pose`) declare the full
  chain so orchestration is data-driven rather than hardcoded.
- **Used by:** **All models** (single-element array for most). Genuinely needed by
  **SAM visual prompting** (`encoder`→`decoder`) and **top-down Keypoint**
  (`detector`→`pose`), which are the only multi-stage pipelines; single-stage
  models declare one `main` stage so clients use a uniform execution loop.
  `stages` describes **stateless** topology only; pipelines that keep state
  between calls (SAM ZSL `learn`→`infer`) additionally declare
  [`state`](#state-learn--infer-pipelines), and wrappers that run a pipeline many
  times per input (tiling) are declared via [`tiling`](#tiling).
- **Required:** Yes (may be a single-element array).

---

## Output description

### Result types (not metadata)

The result type is **not** a metadata field. Each task pipeline returns a
**fixed, task-defined result type**, so there is nothing for the model to declare:

- **Model API** hardcodes one result class per pipeline.
- **Other clients** implement their own result types from the field definitions
  in [Result schema definitions](#result-schema-definitions) below; those
  definitions are the normative contract.

The result type follows from metadata that is already present:

| Selected by | Result type |
|---|---|
| `task = classification` / `action_classification` | `ClassificationResult` |
| `task = segmentation` | `SegmentationResult` |
| `task = detection` | `DetectionResult` |
| `task = instance_segmentation` | `InstanceSegmentationResult`; `RotatedSegmentationResult` when `derived_outputs` contains `rotated_rects` |
| `task = anomaly` | `AnomalyResult` |
| `task = keypoints` | `KeypointResult`; `KeypointResultList` when `stages` is `detector`→`pose` (top-down) |
| `task = visual_prompting` | `VisualPromptingResult`; `ZSLVisualPromptingResult` when `decoder = sam_zsl` |

Optional fields inside a result type are controlled by other metadata
(e.g. `return_soft_prediction`, `output_raw_scores`, XAI `output_roles`) and are
left empty when not produced; the result type itself never changes.

### Result schema definitions

> **Mapping to current Python classes.** The schema names below are documentation
> names for the normative, language-agnostic contract; they are not written into
> model metadata. They map to the existing `model_api` result classes
> as follows: `SegmentationResult` → `ImageResultWithSoftPrediction`,
> `KeypointResult` → `DetectedKeypoints`, `KeypointResultList` →
> `list[DetectedKeypoints]` (top-down pipeline); the others keep the same name
> (`ClassificationResult`, `DetectionResult`, `InstanceSegmentationResult`,
> `RotatedSegmentationResult`, `AnomalyResult`, `VisualPromptingResult`,
> `ZSLVisualPromptingResult`). External clients MUST rely on the field definitions
> here, not on the Python class names.

Each schema below lists its fields as `name: type [shape] — meaning`. `N` = number
of instances/detections, `K` = number of keypoints, `H`/`W` = original image
height/width, `C` = number of classes. Optional fields MAY be empty arrays when the
model does not produce them (e.g. XAI maps).

#### `ClassificationResult`
- `top_labels: array<Label>` — ranked predictions; each `Label` =
  `{ id: int, name: string, score: f32[0,1] }`.
- `saliency_map: u8[N_classes,h,w]` (optional) — per-class XAI map, empty if absent.
- `feature_vector: f32[D]` (optional) — penultimate features, empty if absent.
- `raw_scores: f32[C]` (optional) — full probability/logit vector, empty if absent.

#### `SegmentationResult`
- `result_image: i32[H,W]` — per-pixel hard class index (argmax).
- `soft_prediction: f32[H,W,C] | u8[H,W]` (optional) — per-class probabilities or
  single-channel confidence; present iff the model returns soft predictions.
- `saliency_map: u8[H,W]` (optional) — activation map, empty if absent.
- `feature_vector: f32[D]` (optional) — empty if absent.
- `contours: array<Contour>` (optional, [derived](#derived_outputs)) — each
  `Contour = { label: string, probability: f32, shape: i32[M,2],
  excluded_shapes: array<i32[M,2]> }`; background class is skipped.

> **Normative change vs. current code.** Today `SegmentationModel.postprocess`
> returns a bare `i32[H,W]` array instead of a result object when
> `return_soft_prediction = false`. Under this spec the pipeline MUST always return
> `SegmentationResult`; `return_soft_prediction = false` only leaves
> `soft_prediction` empty.

#### `DetectionResult`
- `bboxes: i32[N,4]` — boxes in `output_coordinate_space` (default
  `original_image_pixels`), always `xyxy` after decoding (see `box_format` for the
  raw encoding).
- `labels: int[N]` — final class indices (after `label_shift`).
- `scores: f32[N][0,1]` — confidence per box.
- `label_names: array<string>[N]` (optional) — resolved names, empty if no labels.
- `saliency_map: u8[C,h,w]` (optional) — empty if absent.
- `feature_vector: f32[D]` (optional) — empty if absent.

#### `InstanceSegmentationResult`
- `bboxes: i32[N,4]` — as in `DetectionResult`.
- `labels: int[N]` — final class indices (after `label_shift`).
- `scores: f32[N][0,1]`.
- `masks: u8[N,H,W] | u8[N,h,w]` — per-instance binary masks; full-image
  (`{0,1}` over `H×W`) when `mask_postprocess` resizes to the image, otherwise the
  raw per-box mask.
- `label_names: array<string>[N]` (optional).
- `saliency_map: array<u8[h,w]>[C]` (optional) — per-class aggregated map.
- `feature_vector: f32[D]` (optional).

#### `RotatedSegmentationResult`
- All fields of `InstanceSegmentationResult`, plus:
- `rotated_rects: array<RotatedRect>[N]` — minimum-area rotated rectangle per
  instance, `RotatedRect = { center: (f32 cx, f32 cy), size: (f32 w, f32 h),
  angle: f32 (degrees) }`, computed from each instance mask
  ([derived](#derived_outputs) `rotated_rects`).

#### `AnomalyResult`
- `anomaly_map: u8[H,W][0,255]` — per-pixel anomaly heat, resized to original.
- `pred_mask: u8[H,W]{0,1}` — thresholded anomaly mask.
- `pred_score: f32[0,1]` — image-level anomaly score.
- `pred_label: string` — one of the two [`labels`](#labels--path_to_labels)
  (`labels[0]` = normal, `labels[1]` = anomalous), chosen via `image_threshold`.
- `pred_boxes: i32[M,4] | null` — bounding boxes, present only when
  `anomaly_mode = detection`, else `null`.

#### `KeypointResult`
- `keypoints: f32[N,K,2]` — `(x, y)` coordinates in `output_coordinate_space`.
- `scores: f32[N,K][0,1]` — per-keypoint confidence.

#### `KeypointResultList`
- `array<KeypointResult>[D]` — one `KeypointResult` (with `N = 1`) per input
  detection, in the order of the detector's `bboxes`; keypoints already offset into
  original-image coordinates. Produced by the top-down (`detector`→`pose`) pipeline.

#### `VisualPromptingResult`
One entry per **prompt label** (`L` = number of distinct prompt labels); each list
element below is indexed by that label.
- `upscaled_masks: array<f32[P,H,W]>[L]` — raw decoder logits upsampled to the
  original image.
- `processed_mask: array<u8[P,H,W]>[L]` (optional) — post-processed binary masks.
- `low_res_masks: array<f32[P,h,w]>[L]` (optional) — decoder masks at the decoder
  resolution (e.g. 256×256).
- `iou_predictions: array<f32[P]>[L]` (optional) — decoder-predicted IoU per mask.
- `scores: array<f32[P]>[L]` — clipped `[0,1]` mask quality scores.
- `labels: array<int>[L]` — prompt label per entry.
- `hard_predictions: array<u8[H,W]{0,1}>[L]` — `upscaled_masks > mask_threshold`.
- `soft_predictions: array<f32[H,W][0,1]>[L]` — `hard_predictions × score`.
- `best_iou: array<f32>[L]` (optional) — best IoU of the selected mask.

#### `ZSLVisualPromptingResult`
- `data: map<int label, PredictedMask>` — one entry per learned label.
- `PredictedMask = { mask: array<u8[H,W]{0,1}>, points: f32[M,2|3],
  scores: f32[M][0,1] }` — masks found for that label, the matched points that
  produced them (`x, y[, score]`) and their scores.

### `labels` / `path_to_labels`

- **Stage:** Post / Output
- **Type:** list[string] / string
- **Possible values:** ordered list of class names; `path_to_labels` is a path to
  a text file with one label per line and overrides `labels`.
- **Description:** The class vocabulary used to turn predicted indices into names.
- **What it does:** Every decoder maps a final class index (after
  [`label_shift`](#label_shift)) to `labels[index]` to fill `label_names` /
  `Label.name` / `pred_label`. Normative rules:
  - **Index base.** `labels[i]` is the name of final index `i`. For decoders with
    `label_shift = 1`, `labels[0]` is the background / ignored class.
  - **"other" class.** For single-output classifiers whose logit count is
    `len(labels) + 1`, the pipeline MUST prepend `"other"` as `labels[0]`
    (current `ClassificationModel._verify_single_output` behavior). Exporters
    SHOULD instead write the complete list so this rule never fires.
  - **Anomaly.** Exactly two entries, `[normal_label, anomalous_label]`.
  - **Hierarchical classification.** `labels` gives the output order; the label
    tree itself comes from `hierarchical_config`
    ([Task-specific output options](#task-specific-output-options)).
  - **Missing labels.** Names are empty strings; indices are still returned.
- **Used by:** **All models that output classes** — Classification, Action Cls,
  Segmentation (contour labels), Detection, Instance Segmentation, Anomaly,
  SAM ZSL (prompt labels). Not used by Keypoint.
- **Required:** Required whenever names are expected in the result; strongly
  recommended for all class-producing models.

### `output_coordinate_space`

- **Stage:** Output
- **Type:** enum string
- **Possible values:** `original_image_pixels`, `normalized`, `heatmap`, `n/a`.
- **Description:** The coordinate frame of spatial outputs (boxes, keypoints, masks).
- **What it does:** Tells the client how to interpret returned coordinates. For
  example, detection boxes are already rescaled to original image pixels, so the
  client must not rescale them again.
- **Used by:** **Detection, Instance Segmentation, Anomaly (detection mode),
  Keypoint, Visual Prompting** — every model with spatial outputs. Not needed by
  plain Classification / Action Cls (no coordinates).
- **Required:** Required when the result contains spatial data.

### `output_value_range`

- **Stage:** Output
- **Type:** JSON object `{ role: range_descriptor }`
- **Possible values (examples):**
  - `{ "anomaly_map": "u8[0,255]" }`
  - `{ "scores": "f32[0,1]" }`
  - `{ "masks": "u8{0,1}" }`
- **Description:** The dtype and value range of each output role.
- **What it does:** Enables correct interpretation/rendering of raw output tensors
  without reading reference code (e.g. anomaly map is uint8 scaled to 0–255).
- **Used by:** Most notably **Anomaly** (map is `u8[0,255]`, not a probability),
  **Action Cls** (score is a raw logit, not a probability) and **Instance
  Segmentation / Visual Prompting** (masks are `{0,1}`); useful for all models so
  clients render/threshold outputs correctly.
- **Required:** Recommended.

### `derived_outputs`

- **Stage:** Post / Output
- **Type:** JSON array of enum strings
- **Possible values:** `contours` (segmentation), `rotated_rects` (instance
  segmentation → `RotatedSegmentationResult`). Empty array (default) = none.
- **Description:** Optional, deterministic computations applied **after** decoding
  that enrich the result without touching model outputs.
- **What it does:** `contours` runs contour extraction per class on `result_image`
  (holes included) and averages the class soft prediction inside each contour for
  `probability`. `rotated_rects` computes a minimum-area rotated rectangle from each
  instance mask's largest contour. Today these are helper functions
  (`SegmentationModel.get_contours`, `get_contours`, `add_rotated_rects`) that
  callers invoke by hand; the field lets a model declare that its consumers expect
  them (e.g. rotated-object models).
- **Used by:** **Segmentation** (`contours`) and **Instance Segmentation**
  (`rotated_rects`).
- **Required:** Optional (default `[]`).

---

## Input / output contract

### `input_roles`

- **Stage:** Infer / Pre
- **Type:** JSON object `{ role: tensor_name }`
- **Possible roles:** `image`, `image_info`, `image_embeddings`, `point_coords`,
  `point_labels`, `mask_input`, `has_mask_input`, `orig_size`, `boxes`, `points`,
  `polygons`, `labels`.
- **Description:** Maps logical input roles to the actual model input tensor names.
- **What it does:** Removes shape-based input sniffing (e.g. "4D tensor = image,
  2D tensor = image_info"). A non-Python client can bind inputs directly by name.
  Roles with non-obvious content carry a qualifier:
  - `image_info` MUST declare its content via `image_info_content` ∈
    {`resized_hw1` (`[[h, w, 1]]` of the **model input**, SSD / MaskRCNN
    segmentoly), `original_hw` (`[[h, w]]` of the **original image**,
    YOLOv3-ONNX)}.
  - `mask_input` / `has_mask_input` (SAM decoder) take constant defaults declared
    in `decoder_constants` (`mask_input_shape: [1,1,256,256]`, zeros).
  - `boxes` / `points` / `polygons` / `labels` are **user prompts** (pipeline
    inputs), not tensors; the pipeline encodes them into `point_coords` /
    `point_labels` per [`decoder_constants`](#decoder_constants).
- **Used by:** **All models.** Critical for **SSD/MaskRCNN segmentoly** and
  **YOLOv3-ONNX** (`image_info` with different content) and **SAM** (decoder
  tensors + user prompts); single-input models still declare the `image` role for
  uniform binding.
- **Required:** Yes.

### `output_roles`

- **Stage:** Infer / Post
- **Type:** JSON object `{ role: tensor_name }`
- **Possible roles:** `logits`, `boxes`, `labels`, `scores`, `masks`, `seg`,
  `simcc_x`, `simcc_y`, `anomaly_map`, `pred_score`, `pred_label`, `pred_mask`,
  `saliency_map`, `feature_vector`, `raw_scores`, `iou`, `predictions` (fused
  box/score/class tensor — Ultralytics, YOLOX, YOLO-DETR, YOLO-seg), `region_<i>`
  (i-th YOLO region/grid output, in the order the decoder consumes them),
  `nms_indices` (in-model NMS result, YOLOv3-ONNX), `protos` (YOLO-seg prototype
  masks), `image_embeddings` (SAM encoder), `upscaled_masks` / `low_res_masks`
  (SAM decoder).
- **Description:** Maps logical output roles to actual model output tensor names.
- **What it does:** Removes fragile heuristics such as shape-length branching,
  `startswith("TopK")` filtering, `find_layer_by_name`, and
  `next(iter(outputs.values()))`. XAI outputs (`saliency_map`, `feature_vector`)
  become declared rather than string-matched.
- **Used by:** **All models.** Essential for multi-output models —
  **Instance Segmentation** (boxes/labels/masks; YOLO-seg `predictions`+`protos`),
  **Anomaly** (4 named outputs), **Keypoint** (simcc_x/simcc_y), **YOLO region
  models** (`region_<i>`), **YOLOv3-ONNX** (boxes/scores/`nms_indices`), **SAM**,
  and **Detection/Classification/Segmentation** with optional XAI
  (`saliency_map`, `feature_vector`).
- **Required:** Yes.

### `output_layout`

- **Stage:** Post
- **Type:** JSON object `{ role: "<dtype>:<shape_grammar>" }`
- **Possible values (examples):** `"f32:1x1xNx7"` (SSD single output),
  `"f32:Nx5"` (boxes+score), `"f32:1x(4+C)xN"` (Ultralytics, transposed),
  `"f32:1x(4+C+M)xN"` (YOLO-seg detections), `"f32:1xMxPhxPw"` (YOLO-seg protos),
  `"f32:1xNx6"` (YOLO-DETR), `"f32:1xA(5+C)xHxW"` / `"f32:1xHxWxA(5+C)"` (YOLO
  region NCHW / NHWC), `"i32:Kx3"` (YOLOv3-ONNX NMS indices), `"f32:NxKxW"`
  (SimCC).
- **Description:** Documented dtype and shape grammar for each output role.
- **What it does:** Lets a client validate and index tensors deterministically and
  replaces the shape/precision checks currently scattered in constructors (e.g.
  YOLOv5 requires `f32` and rank 3; YOLO-DETR requires `[1,N,6]`; YOLO-seg detects
  outputs by rank). Symbols `C` (classes), `M` (mask dim), `A` (anchors per cell)
  MUST be resolvable from `labels` / `decoder_constants`, not inferred from shapes.
- **Used by:** **Detection** (every decoder), **Instance Segmentation** (mask
  stacks, prototypes), **Keypoint** (`NxKxW` SimCC). Informational for simple
  single-vector classifiers.
- **Required:** Required for detection / instance segmentation decoders;
  recommended otherwise.

---

## Preprocessing properties

### `input_modality`

- **Stage:** Pre
- **Type:** enum string
- **Possible values:** `image`, `video`, `prompt`.
- **Description:** The kind of raw input the model consumes.
- **What it does:** Chooses the preprocessing family. `video` enables temporal
  handling (per-frame resize + 6D layout); `prompt` enables the prompt-encoding
  path (SAM decoder). Replaces separate `base_preprocess` overrides.
- **Used by:** **All models.** `image` for most; **Action Cls** needs `video`
  (per-frame handling + temporal axis); **SAM** needs `prompt` for the decoder
  stage. Without it a client cannot pick the right preprocessing family.
- **Required:** Yes.

### `input_layout`

- **Stage:** Pre
- **Type:** enum string
- **Possible values:** `NCHW`, `NHWC`, `NSCTHW`, `NSTHWC`.
- **Description:** The axis order of the model input tensor.
- **What it does:** Determines the layout transform applied after resize/normalize.
  The `S`/`T` axes (clips/time) are only present for video models.
- **Used by:** **All models.** Image models are `NCHW`/`NHWC`; **Action Cls** is
  the only one using the 6D `NSCTHW`/`NSTHWC` forms. Needed so the client transposes
  the preprocessed tensor to match the model input.
- **Required:** Yes.

### `embedded_processing`

- **Stage:** Pre
- **Type:** boolean
- **Possible values:** `true`, `false`.
- **Description:** Whether resize + normalization were **baked into the model graph
  at export time**.
- **What it does:** Three cases must be distinguished:
  1. `embedded_processing = true` — the graph already contains resize/normalize;
     the client feeds the raw image and MUST NOT repeat it.
  2. `embedded_processing = false` and the client **can** embed preprocessing at
     load time (e.g. OpenVINO PrePostProcessor on a static-shape model) — the
     client builds graph-side preprocessing from the resize/intensity/mean/scale
     fields. This is what `model_api` does today for most static image models.
  3. `embedded_processing = false` and the client **cannot** embed (dynamic shapes,
     6D video inputs, runtimes without PPP, prompt inputs) — the client performs
     the full chain in host code.

  Cases 2 and 3 MUST produce identical tensors; the choice is a client
  implementation detail, not model metadata.
- **Used by:** **All models.** Models exported with PPP use `true`. Models whose
  preprocessing cannot be expressed as graph PPP — **Action Cls** (6D video),
  **SAM decoder** (prompt tensors) — always fall into case 3 on the client.
- **Required:** Yes.

### `resize_type`

- **Stage:** Pre
- **Type:** enum string
- **Possible values:**
  - `standard` — stretch to model size, no aspect preservation.
  - `fit_to_window` — keep aspect, pad bottom/right.
  - `fit_to_window_letterbox` — keep aspect, pad symmetrically (centered).
  - `fit_to_window_top_left` — keep aspect, image at the top-left, pad
    bottom/right with `pad_value`, invert with a single scale `min(W/w, H/h)`
    (YOLOX).
  - `longest_side` — scale the longest side to `image_size`, pad the rest (SAM).
  - `crop` — center crop after aspect-preserving resize.

  `RESIZE_TYPES` in `adapters/utils.py` is the authoritative list of modes
  implemented today; `fit_to_window_top_left` and `longest_side` formalize logic
  currently hardcoded in `YOLOX._resize_image` and the SAM wrappers.
- **Description:** Strategy used to resize the input to the model's spatial size.
- **What it does:** Defines padding/scaling behavior. Each mode's exact padding
  rule must be specified so coordinate inversion in postprocessing matches
  (`ResizeMetadata` today).
- **Used by:** **All image/video models**, and it must agree with box/keypoint/mask
  rescaling. YOLO family / YOLO-seg / YOLO-DETR / Ultralytics use
  `fit_to_window_letterbox`, YOLOX `fit_to_window_top_left`, YOLOF `standard`, SAM
  `longest_side`, Detection/Instance Seg/Keypoint usually `fit_to_window`.
- **Required:** Yes (needed for coordinate inversion even when preprocessing is
  embedded).

### `input_dtype` / `interpolation_mode` / `pad_value`

- **Stage:** Pre
- **Type:** enum string / enum string / integer
- **Possible values:**
  - `input_dtype`: `u8` (default), `f32`, `u16`, `i16`.
  - `interpolation_mode`: `LINEAR` (default), `CUBIC`, `NEAREST`.
  - `pad_value`: `0..65535` (default `0`; Ultralytics, YOLO-seg and YOLOX use
    `114`).
- **Description:** Element type of the image fed to the pipeline, the resize
  interpolation, and the fill value for padded regions.
- **What it does:** Ensures bit-exact preprocessing across clients. Today some of
  these are hardcoded in wrappers (YOLOX pads with `114` in `_resize_image`;
  YOLOv3-ONNX forces `CUBIC`), which must become metadata.
- **Used by:** **All image models**; `pad_value` matters for every padding
  `resize_type`; `interpolation_mode = CUBIC` for YOLOv3-ONNX; `input_dtype` for
  16-bit / medical inputs.
- **Required:** Optional (defaults as above).

### `orig_height` / `orig_width` / `input_frame_height` / `input_frame_width`

- **Stage:** Pre / Post
- **Type:** integer
- **Possible values:** positive integers or unset.
- **Description:** `orig_height`/`orig_width` = the model's spatial input size
  **before** preprocessing was embedded (after embedding, the graph input becomes
  the raw frame size, so this is the only record of the network resolution).
  `input_frame_height`/`input_frame_width` = fixed raw-frame size used to build a
  static-shape embedded preprocessing graph.
- **What it does:** `orig_*` is required to invert coordinates
  (`ResizeMetadata.compute(model_width=orig_width, …)`) for boxes, masks and
  keypoints when preprocessing is embedded. `input_frame_*` lets a client compile a
  static graph (GPU/NPU) instead of a dynamic one.
- **Used by:** **Detection, Instance Segmentation (incl. YOLO-seg), Keypoint** —
  any model that maps spatial outputs back; `input_frame_*` for any model deployed
  on static-shape devices.
- **Required:** `orig_*` required when `embedded_processing = true` and outputs
  are spatial; `input_frame_*` optional.

### `intensity_mode`

- **Stage:** Pre
- **Type:** enum string
- **Possible values:** `none`, `scale_to_unit`, `window`, `percentile`, `range_scale`.
- **Description:** Intensity scaling applied before mean/scale normalization.
- **What it does:** Normalizes pixel intensities. For example, anomaly models use
  `scale_to_unit` (divide by max value, e.g. 255) instead of a custom `/255`
  override. Related numeric parameters: `intensity_max_value`,
  `intensity_window_center`, `intensity_window_width`, `intensity_percentile_low`,
  `intensity_percentile_high`, `intensity_scale_factor`, `intensity_min_value`,
  `intensity_repeat_channels`.
- **Used by:** **Anomaly** (`scale_to_unit`, replacing its custom `/255`) and any
  model trained on non-8-bit or medical imagery (window/percentile/range_scale).
  Standard RGB models use `none`. `intensity_repeat_channels` is for single-channel
  inputs fed to 3-channel backbones.
- **Required:** Optional (defaults to `none`).

### `mean_values` / `scale_values` / `reverse_input_channels`

- **Stage:** Pre
- **Type:** list[float] / list[float] / boolean
- **Possible values:** per-channel floats; booleans `true`/`false`.
- **Description:** Standard normalization: `(pixel - mean) / scale`, with optional
  BGR↔RGB channel reversal.
- **What it does:** Brings input into the distribution the model was trained on.
  Ultralytics and YOLO-seg use `scale_values = [255]` (unit scaling) and
  YOLOv5/8/11 use `reverse_input_channels = true`; these are currently wrapper
  defaults and MUST be written explicitly at export. YOLOv3-ONNX applies no
  normalization (empty lists).
- **Used by:** **All image/video models.** Applied in host code for case-3 clients
  (see [`embedded_processing`](#embedded_processing)) — notably **Action Cls** —
  and in the graph otherwise.
- **Required:** Optional (defaults: empty lists, `false`).

### `temporal.clip_size`

- **Stage:** Pre
- **Type:** integer
- **Possible values:** positive integer (e.g. `16`), or `-1` for dynamic.
- **Description:** Number of frames (T axis) a video model expects per clip.
- **What it does:** Validates/assembles the temporal dimension for
  `input_modality: video`. Ignored for image/prompt models.
- **Used by:** **Action Cls only** — it is the one video model and needs `T` to
  validate/assemble the clip. All other models omit it.
- **Required:** Required for `input_modality: video`.

---

## Postprocessing properties

### `activation`

- **Stage:** Post
- **Type:** enum string
- **Possible values:** `softmax`, `sigmoid`, `argmax`, `none`.
- **Description:** Activation applied to raw logits before decoding.
- **What it does:** Converts logits into probabilities (softmax/sigmoid), a class
  index (argmax), or leaves them untouched (none). The spec must fix the axis and
  numerical conventions. The value MUST be explicit: today `ClassificationModel`
  guesses at runtime with `is_softmaxed(...)` whether the model already applied
  softmax — a generic pipeline MUST NOT guess; exporters write `none` when the
  graph already outputs probabilities.
- **Used by:** **Classification** (softmax multiclass / sigmoid multilabel / none),
  **Segmentation** & **Action Cls** (argmax), **Anomaly** (threshold-normalize),
  **Keypoint** (optional softmax on SimCC), **YOLO family** (sigmoid on
  objectness/class for YoloV4/YOLOF; `none` for Ultralytics/YOLO-DETR/YOLOX),
  instance-seg and YOLO-seg masks (sigmoid/threshold).
- **Required:** Yes for score-producing tasks.

### `decoder`

- **Stage:** Post
- **Type:** enum string
- **Possible values:**
  - Classification: `topk`, `multilabel`, `hierarchical`, `action_logits`.
  - Segmentation: `seg_argmax`.
  - Detection: `ssd_boxes` (SSD single / multi / boxes-labels outputs),
    `yolo_region` (YOLOv3/v4/tiny anchor regions), `yolof` (YOLOF single-level
    anchor decode, no objectness), `yolox` (anchor-free grid × stride decode),
    `ultralytics` (YOLOv5/v8/11, `[1,4+C,N]`, no objectness), `yolo_detr`
    (decoded `[1,N,6]` queries), `in_model_nms` (model already ran NMS and outputs
    boxes / scores / `nms_indices`, YOLOv3-ONNX).
  - Instance segmentation: `maskrcnn_crop`, `detr_fullmask`, `yolo_proto`
    (YOLO-seg prototype masks).
  - Other: `anomaly_map`, `simcc`, `sam_decode`, `sam_zsl`, `none`.
- **Description:** The named algorithm that turns activated outputs into the result
  structure.
- **What it does:** Selects the decoding routine previously encoded by the Python
  class name (e.g. `SSD` vs `YOLO`, `MaskRCNN` vs `DETRInstSeg`). Each value MUST
  be defined to test-vector precision in the spec. These values are **variation
  points inside a single task pipeline**, not separate model implementations: the
  task pipeline implements each named decoder once and selects it from this field;
  it never branches on `model_type`.
- **Used by:** **All models**, and it is the field that disambiguates same-`task`
  variants: SSD vs the six YOLO-family detection decoders; MaskRCNN vs DETRInstSeg
  vs YOLO-seg; standard vs ZSL visual prompting. Classification variants use
  `topk`/`multilabel`/`hierarchical`.
- **Required:** Yes.

### `box_format`

- **Stage:** Post
- **Type:** enum string
- **Possible values:** `xyxy_norm`, `xyxy_pixel`, `yxyx_pixel`, `cxcywh_norm`,
  `cxcywh_pixel`, `xywh_norm`, `xywh_pixel`. `_norm` = normalized to model input
  size, `_pixel` = model-input pixels.
- **Description:** Encoding of bounding boxes in the raw output.
- **What it does:** Tells the decoder how to interpret and convert box coordinates
  to `xyxy` before rescaling into `output_coordinate_space`.
- **Used by:** **Detection** and **Instance Segmentation** — any model emitting
  boxes: SSD (`xyxy_norm`), Ultralytics/YOLOX/YOLO-seg (`cxcywh_pixel`), YOLO-DETR
  (`cxcywh_norm`), YOLOv3-ONNX (`yxyx_pixel`), MaskRCNN/DETRInstSeg
  (`xyxy_pixel`). Anchor region decoders (`yolo_region`, `yolof`) produce boxes
  from anchors and do not use it.
- **Required:** Required for detection / instance segmentation (except anchor
  region decoders).

### `label_shift`

- **Stage:** Post
- **Type:** integer
- **Possible values:** `0`, `1` (typically).
- **Description:** Constant added to raw class indices to obtain final label ids.
- **What it does:** Aligns model output indices with the label list. Currently the
  hidden `_labels_shift` (1 for MaskRCNN / DETRInstSeg) and a literal `+1` in
  YOLO-seg. The shifted index is used for **both** `labels` and `label_names`
  (current YOLO-seg looks up names with the unshifted index — an inconsistency the
  generic pipeline resolves).
- **Used by:** **MaskRCNN**, **DETRInstSeg**, **YOLO-seg** (`1`, background at
  index 0); all other models use `0`.
- **Required:** Optional (defaults to `0`).

### `mask_postprocess`

- **Stage:** Post
- **Type:** enum string
- **Possible values:** `none`, `box_crop`, `full_image`, `prototype`.
- **Description:** How a raw instance mask is expanded into image space.
- **What it does:** `box_crop` resizes a small per-box mask (e.g. 28×28) into the
  bounding box (MaskRCNN); `full_image` resizes a whole-image mask to the original
  dimensions and thresholds it (DETR-Seg); `prototype` computes
  `sigmoid(coeffs @ protos)`, zeroes pixels outside each box (scaled to prototype
  resolution), upsamples to model input, removes letterbox padding, resizes to the
  original image and thresholds at `mask_threshold` (YOLO-seg).
- **Used by:** **MaskRCNN** (`box_crop`), **DETRInstSeg** (`full_image`),
  **YOLO-seg** (`prototype`). Not applicable to other tasks.
- **Required:** Required for instance segmentation.

### `nms` and NMS parameters

- **Stage:** Post
- **Type:** boolean + numeric group
- **Properties:**
  - `nms_execute` (bool) — whether to run NMS.
  - `iou_threshold` (float, default `0.5`) — IoU cutoff.
  - `agnostic_nms` (bool) — treat all classes as one.
  - `nms_max_predictions` (int, default `200`, `0` = unlimited).
  - `nms_include_boundaries` (bool, default `false`) — compute box area/IoU with
    the `+1` pixel convention (YOLOX).
- **Description:** Non-maximum suppression configuration.
- **What it does:** Removes overlapping duplicate detections/instances. Exactly one
  NMS pass is applied by the pipeline; the extra same-class `_filter` pass that
  `YOLO._parse_outputs` currently runs after `_calculate_nms` is folded into this
  single pass. Not applied when `decoder = in_model_nms`.
- **Used by:** **SSD / YOLO family Detection** and **Instance Segmentation**
  (MaskRCNN/DETRInstSeg/YOLO-seg); also used by [`tiling`](#tiling) merging.
  Irrelevant to classification, segmentation, anomaly, keypoints, action.
- **Required:** Optional (detection/instance seg only).

### Threshold parameters

- **Stage:** Post
- **Type:** float
- **Properties:**
  - `confidence_threshold` (default `0.5`) — filter low-confidence predictions
    (detection, instance seg, multilabel/hierarchical classification).
  - `soft_threshold` (default `-inf`) — per-pixel confidence for segmentation.
  - `image_threshold` (default `0.5`) — image-level anomaly cutoff.
  - `pixel_threshold` (default `0.5`) — pixel-level anomaly cutoff.
  - `mask_threshold` — binarization cutoff for decoded masks (instance seg `0.5`;
    SAM decoder `0.0` on logits).
  - `zsl_match_threshold` (default `0.65`) — similarity cutoff when matching
    target features against learned reference features (SAM ZSL).
- **Description:** Confidence cutoffs applied during decoding.
- **What it does:** Controls which predictions/pixels survive into the result.
- **Used by:** `confidence_threshold` → **Detection, Instance Seg,
  multilabel/hierarchical Classification**; `soft_threshold` → **Segmentation**;
  `image_threshold` + `pixel_threshold` → **Anomaly**; `mask_threshold` →
  **Instance Seg / Visual Prompting**; `zsl_match_threshold` → **SAM ZSL**. Each
  task uses only the thresholds relevant to its decoder.
- **Required:** Optional; depends on `task`.

### Task-specific output options

- **Stage:** Post / Output
- **Type:** per-task parameter group (already present in `ParameterRegistry`;
  listed here so external clients implement them).
- **Properties:**
  - **Classification:** `topk` (int ≥ 1, default `1`) — number of labels returned
    by `topk`; `output_raw_scores` (bool) — fill `raw_scores` with the full
    activated vector; `hierarchical_config` (JSON string: `cls_heads_info`,
    `label_tree_edges`, per-head logit ranges) — required for
    `decoder = hierarchical`; `hierarchical_postproc` ∈ {`greedy`,
    `probabilistic`} — label-tree resolution algorithm. For hierarchical models,
    saliency maps are reordered from model-head order into `labels` order.
  - **Segmentation:** `return_soft_prediction` (bool, default `true`) — fill
    `soft_prediction`.
  - **Instance segmentation:** `postprocess_semantic_masks` (bool, default
    `true`) — `true`: masks expanded to full image per `mask_postprocess`;
    `false`: raw per-instance masks are returned.
  - **Keypoint:** `apply_softmax` (bool, default `true`) — compute SimCC scores
    from softmaxed distributions.
  - **Anomaly:** `anomaly_mode` ∈ {`classification`, `segmentation`,
    `detection`} — which result fields are meaningful; `detection` fills
    `pred_boxes`. (Named `task` in today's `ParameterRegistry.ANOMALY`; renamed to
    avoid clashing with the top-level [`task`](#task).)
- **Description:** Switches that change *which* result fields are produced or how
  they are resolved, without changing the decoder algorithm.
- **What it does:** Lets one task pipeline serve all output variants of a task.
- **Used by:** as listed per task.
- **Required:** Optional (defaults as listed), except `hierarchical_config` for
  hierarchical classification.

### `decoder_constants`

- **Stage:** Post
- **Type:** JSON object (free-form, decoder-specific)
- **Possible values (examples):**
  - SimCC (keypoints): `{ "simcc_split_ratio": 2.0, "decode_beta": 150.0, "sigma": 6.0 }`
  - MaskRCNN: `{ "expand_box_scale": "auto", "mask_threshold": 0.5 }`
  - DETR-Seg: `{ "mask_threshold": 0.5 }`
  - Anomaly: `{ "normalization_scale": 1.0, "map_scale": 255 }`
  - Segmentation: `{ "blur_strength": -1 }`
  - YOLO region (per output): `{ "anchors": [...], "masks": [...], "num": 3,
    "coords": 4, "classes": 80, "sides": [h, w], "layout": "NCHW",
    "use_input_size": true }`
  - YOLOF: `{ "anchors": [...], "num": 6, "classes": 80 }`
  - YOLOX: `{ "strides": [8, 16, 32] }`
  - YOLOv3-ONNX: `{ "classes": 80 }`
  - YOLO-seg: `{ "mask_dim": 32, "proto_size": [160, 160], "mask_threshold": 0.5 }`
  - SAM decoder: `{ "image_size": 1024, "mask_threshold": 0.0, "embed_dim": 256,
    "mask_input_shape": [1, 1, 256, 256], "box_point_labels": [2, 3],
    "point_label": 1 }`
  - SAM ZSL: `{ "num_bg_points": 1, "downsizing": 64,
    "default_threshold_reference": 0.3, "default_threshold_target": 0.0,
    "is_cascade": false }`
- **Description:** Externalizes constants that are currently hardcoded in Python and
  therefore invisible to non-shared-codebase clients.
- **What it does:** Provides every numeric constant a client needs to reproduce
  keypoint, mask, anomaly, YOLO-family or SAM decoding exactly.
- **Used by:** **Keypoint** (SimCC), **MaskRCNN/DETRInstSeg/YOLO-seg** (mask
  constants), **Anomaly** (normalization), **Segmentation** (`blur_strength`),
  **YOLO family** (anchors / grids / strides / class count), **SAM / SAM ZSL**
  (prompt conventions and matching constants). Not needed by plain Classification
  / Action Cls.
- **Required:** Required whenever the chosen `decoder` relies on such constants.

> **YOLO family — values to capture at export.** All models are re-exported to
> carry the new metadata, so re-export itself is not YOLO-specific. What *is*
> specific is where these wrappers get their parameters today:
> - `YOLO` / `YoloV4`: `operations_by_type("RegionYolo")` reads op `.meta`
>   (num/coords/classes/anchors/mask); grid `cx,cy` come from `//32` arithmetic;
>   the variant (`is_tiny`) from `len(outputs) == 2`; per-output layout from
>   `shape[2] == shape[3]`; `classes` from `channels // num - 5`; anchor/mask
>   defaults from `is_tiny`; outputs are ordered by spatial size.
> - `YOLOF`: anchors from the `ANCHORS["YOLOF"]` table, `num = 6`, `classes`
>   from `shape[1] // num - 4`.
> - `YOLOX`: strides `[8, 16, 32]` and grids built from the input size.
> - `YoloV3ONNX`: `classes = 80` hardcoded; outputs identified by shape.
> - `YOLO-seg`: `mask_dim`, prototype size and class count derived from output
>   ranks/shapes.
>
> A metadata-only detection pipeline **cannot** perform this introspection, so
> these fields MUST be captured explicitly in `decoder_constants` (and output
> order in `output_roles`). `YOLO-DETR` and Ultralytics (v5/8/11) need only
> `labels` and `output_layout`.

---

## Pipeline wrappers and stateful pipelines

Two current use cases are neither a decoder nor a stateless stage graph: tiling,
which runs a task pipeline many times per input, and ZSL visual prompting, which
keeps learned state between calls. Both are described declaratively here so the
generic task pipelines can support them without model-specific code.

### `tiling`

- **Stage:** Pre / Post (wraps the whole task pipeline)
- **Type:** JSON object (optional)
- **Possible values:**
  `{ "enabled": true, "tile_size": 400, "tiles_overlap": 0.5,
  "tile_with_full_img": true, "max_pred_number": 100, "iou_threshold": 0.45,
  "merge": "nms" }`
  - `tile_size` (int ≥ 1) — tile edge in original-image pixels.
  - `tiles_overlap` (float `[0,1]`) — fractional overlap; stride =
    `tile_size × (1 − tiles_overlap)`.
  - `tile_with_full_img` (bool) — also run the pipeline on the full image.
  - `max_pred_number` (int) — cap on merged predictions.
  - `iou_threshold` (float) — NMS IoU used during merging (independent of the
    model's own NMS).
  - `merge` ∈ {`nms` (detection / instance seg: offset boxes and masks by tile
    origin, then NMS), `average` (segmentation: accumulate soft predictions per
    pixel and average overlaps)}.
- **Description:** Splits large images into overlapping tiles, runs the task
  pipeline on each, and merges the results into one task result.
- **What it does:** Replaces the `DetectionTiler`, `InstanceSegmentationTiler` and
  `SemanticSegmentationTiler` classes with one tiling wrapper configured by
  metadata. Saliency maps are merged per tile into a full-image map.
- **Used by:** **Detection**, **Instance Segmentation**, **Segmentation** models
  trained for tiled inference.
- **Required:** Optional (absent = no tiling).

### `state` (learn / infer pipelines)

- **Stage:** Compat / Infer
- **Type:** JSON object (optional)
- **Possible values:**
  `{ "kind": "reference_features", "phases": ["learn", "infer"],
  "learn_inputs": ["image", "boxes", "points", "polygons"],
  "infer_inputs": ["image"], "persisted": ["feature_vectors", "used_indices"] }`
- **Description:** Declares that the pipeline has a `learn` phase producing state
  that later `infer` calls consume.
- **What it does:** For SAM ZSL: `learn` encodes the image, decodes masks from the
  prompts (polygons are rasterized directly without the decoder), and stores masked
  mean features per label (`VisualPromptingFeatures = { feature_vectors:
  f32[L,1,D], used_indices: int[L] }`). `infer` encodes the target image, matches
  features against the references with `zsl_match_threshold`, selects foreground /
  background points (`num_bg_points`, `downsizing`), decodes masks and returns
  `ZSLVisualPromptingResult`. The persisted state is serializable so a client can
  skip `learn` by loading previously generated features.
- **Used by:** **SAM ZSL** (`SAMLearnableVisualPrompter`). All other pipelines are
  stateless and omit this field.
- **Required:** Required for `decoder = sam_zsl`; absent otherwise.

---

## Per-model mapping table

| Property | Classification | Segmentation | SSD / Detection | YOLO region (v3/v4/tiny) | MaskRCNN | DETRInstSeg | Anomaly | Keypoint | Action Cls | Visual Prompting (SAM) |
|---|---|---|---|---|---|---|---|---|---|---|
| `task` | classification | segmentation | detection | detection | instance_segmentation | instance_segmentation | anomaly | keypoints | action_classification | visual_prompting |
| `stages` | 1 (`main`) | 1 (`main`) | 1 (`main`) | 1 (`main`) | 1 (`main`) | 1 (`main`) | 1 (`main`) | 1 (`main`); 2 if top-down (`detector`→`pose`) | 1 (`main`) | 2 (`encoder`→`decoder`) |
| result type (implied, not metadata) | `ClassificationResult` | `SegmentationResult` | `DetectionResult` | `DetectionResult` | `InstanceSegmentationResult` (`Rotated…` with `rotated_rects`) | `InstanceSegmentationResult` | `AnomalyResult` | `KeypointResult` (`KeypointResultList` top-down) | `ClassificationResult` | `VisualPromptingResult` |
| `labels` | class list (+`"other"` rule) | class list | class list | class list | background + classes | background + classes | `[normal, anomalous]` | — | class list | prompt labels |
| `input_modality` | image | image | image | image | image | image | image | image | video | prompt (+image for encoder) |
| `input_layout` | NCHW/NHWC | NCHW/NHWC | NCHW/NHWC | NCHW/NHWC | NCHW/NHWC | NCHW/NHWC | NCHW/NHWC | NCHW/NHWC | NSCTHW / NSTHWC | NCHW |
| `input_roles` | image | image | image (+`image_info`=`resized_hw1` opt.) | image | image (+`image_info`=`resized_hw1` if segmentoly) | image | image | image | image (video) | encoder: image; decoder: image_embeddings, point_coords, point_labels, mask_input, has_mask_input, orig_size |
| `output_roles` | logits (+saliency, feature, raw_scores) | seg (+feature_vector) | boxes/labels/scores (+saliency, feature) | `region_<i>` | boxes/labels/masks (+scores, saliency, feature) | boxes/labels/masks | pred_score/pred_label/pred_mask/anomaly_map **or** single map | simcc_x/simcc_y | logits | encoder: image_embeddings; decoder: upscaled_masks, low_res_masks, iou/scores |
| `embedded_processing` | per export | per export | per export | per export | per export | per export | per export (`scale_to_unit` when host-side) | per export | false (6D video, host-side) | encoder per export; decoder host-side prompts |
| `resize_type` | standard | standard | standard / fit_to_window | fit_to_window_letterbox | fit_to_window | fit_to_window | standard | fit_to_window | standard | longest_side |
| `intensity_mode` | per-model | per-model | per-model | per-model | per-model | per-model | `scale_to_unit` | per-model | mean/scale | SAM normalize |
| `temporal.clip_size` | — | — | — | — | — | — | — | — | `t` (e.g. 16) | — |
| `activation` | softmax / sigmoid / none (explicit) | argmax (+optional blur/thresh) | none (decoded in graph) | sigmoid (YoloV4) / none (v3) | — | — | threshold-normalize | argmax (+optional softmax) | argmax | mask threshold |
| `decoder` | topk / multilabel / hierarchical | seg_argmax | ssd_boxes | yolo_region | maskrcnn_crop | detr_fullmask | anomaly_map | simcc | action_logits | sam_decode |
| `box_format` | — | — | xyxy_norm | — (anchor decode) | xyxy_pixel | xyxy_pixel | xyxy (from mask) | — | — | — |
| `label_shift` | 0 | 0 | 0 | 0 | **1** | **1** | 0 | 0 | 0 | 0 |
| `mask_postprocess` | — | — | — | — | box_crop | full_image | — | — | — | — (decoder upscales) |
| `nms` | — | — | configurable | configurable | configurable | configurable | — | — | — | — |
| thresholds | `confidence_threshold` (multilabel/hier) | `soft_threshold` | `confidence_threshold` | `confidence_threshold` | `confidence_threshold`, `mask_threshold` | `confidence_threshold`, `mask_threshold` | `image_threshold`, `pixel_threshold` | — | — | `mask_threshold` |
| task options | `topk`, `output_raw_scores`, `hierarchical_config`, `hierarchical_postproc` | `return_soft_prediction` | — | — | `postprocess_semantic_masks` | `postprocess_semantic_masks` | `anomaly_mode` | `apply_softmax` | — | — |
| `decoder_constants` | — | `blur_strength` | — | anchors/masks/num/coords/classes/sides/layout **per output** | expand_box scale, mask>0.5 | mask>0.5 | `normalization_scale`, ×255 | `simcc_split_ratio=2.0`, `decode_beta=150.0`, `sigma=6.0` | — | image_size, mask_threshold, embed_dim, mask_input_shape, point-label conventions |
| `derived_outputs` | — | `contours` (opt.) | — | — | `rotated_rects` (opt.) | `rotated_rects` (opt.) | — | — | — | — |
| `tiling` | — | optional (`average`) | optional (`nms`) | optional (`nms`) | optional (`nms`) | optional (`nms`) | — | — | — | — |
| `output_coordinate_space` | n/a | original_image_pixels | original_image_pixels | original_image_pixels | original_image_pixels | original_image_pixels | original_image_pixels | original_image_pixels | n/a | original_image_pixels |
| `output_value_range` | scores [0,1] | labels int / soft u8 | scores [0,1] | scores [0,1] | masks {0,1}, scores [0,1] | masks {0,1} | anomaly_map u8[0,255], score [0,1] | scores [0,1], coords px | score = raw logit | masks {0,1}, scores [0,1] |

### Additional wrappers

Wrappers not shown above, listed by the fields where they differ from their
task's column.

| Property | YOLOF | YOLOX | YOLOv3-ONNX | YOLOv5 / v8 / 11 (Ultralytics) | YOLO-DETR | YOLO-seg | SAM ZSL |
|---|---|---|---|---|---|---|---|
| `task` | detection | detection | detection | detection | detection | instance_segmentation | visual_prompting |
| `decoder` | `yolof` | `yolox` | `in_model_nms` | `ultralytics` | `yolo_detr` | `yolo_proto` | `sam_zsl` |
| `input_roles` | image | image | image, `image_info`=`original_hw` | image | image | image | encoder + decoder (as SAM) |
| `output_roles` | `region_0` | `predictions` | boxes, scores, `nms_indices` | `predictions` | `predictions` | `predictions`, `protos` | as SAM |
| `output_layout` | `f32:1xA(4+C)xHxW` | `f32:1xNx(5+C)` | boxes `f32:1xNx4`, scores `f32:1xCxN`, `i32:Kx3` | `f32:1x(4+C)xN` | `f32:1xNx6` | `f32:1x(4+C+M)xN`, `f32:1xMxPhxPw` | as SAM |
| `resize_type` | standard | fit_to_window_top_left | fit_to_window_letterbox | fit_to_window_letterbox | fit_to_window_letterbox | fit_to_window_letterbox | longest_side |
| `pad_value` / `interpolation_mode` | 0 / LINEAR | 114 / LINEAR | 0 / CUBIC | 114 / LINEAR | 0 / LINEAR | 114 / LINEAR | 0 / LINEAR |
| `scale_values` / `reverse_input_channels` | per-model | per-model | none (no normalization) | `[255]` / true | per-model | `[255]` / false | SAM normalize |
| `activation` | sigmoid | none | none | none | none | sigmoid (masks) | — |
| `box_format` | — (anchor decode) | `cxcywh_pixel` | `yxyx_pixel` | `cxcywh_pixel` | `cxcywh_norm` | `cxcywh_pixel` | — |
| `label_shift` | 0 | 0 | 0 | 0 | 0 | **1** | 0 |
| `mask_postprocess` | — | — | — | — | — | `prototype` | — |
| `nms` | per-class | agnostic, `nms_include_boundaries`, IoU 0.65 | **not applied** (in model) | per-class, IoU 0.7, max 30000 | off by default | per-class, IoU 0.5 | — |
| `confidence_threshold` | 0.5 | 0.5 | 0.5 | 0.25 | 0.5 | 0.25 | `zsl_match_threshold` 0.65 |
| `decoder_constants` | anchors, num=6, classes | strides `[8,16,32]` | classes=80 | — | — | mask_dim, proto_size, mask_threshold | SAM + num_bg_points, downsizing, reference/target thresholds |
| `state` | — | — | — | — | — | — | `reference_features` (learn → infer) |
| result type (implied, not metadata) | `DetectionResult` | `DetectionResult` | `DetectionResult` | `DetectionResult` | `DetectionResult` | `InstanceSegmentationResult` | `ZSLVisualPromptingResult` |

### Key observations

- **`label_shift`** is `1` only for instance-segmentation decoders that reserve a
  background index (`MaskRCNN`, `DETRInstSeg`, `YOLO-seg`) — today encoded as
  `_labels_shift` or a literal `+1`.
- **`mask_postprocess`** is the main difference between the three
  instance-segmentation decoders (`box_crop` / `full_image` / `prototype`).
- **Preprocessing location is a client choice**, not model metadata: only models
  with `embedded_processing = true` have it baked in; for the rest a client either
  builds graph PPP at load time or runs host-side code, with identical results.
  Action Cls (6D video) and the SAM decoder (prompt tensors) always need host-side
  preprocessing.
- **`decoder_constants`** is where the currently-invisible Python constants live;
  without externalizing these, a non-shared-codebase client cannot reproduce
  keypoint, mask, anomaly, YOLO-family or SAM outputs.
- **Visual Prompting** and **top-down Keypoint** remain multi-stage pipelines —
  the `stages` field describes each stage and how they are wired. **SAM ZSL**
  additionally needs `state`, and **tiling** wraps detection / instance
  segmentation / segmentation without new decoders.
- **One generic pipeline per task.** Every task pipeline selects behavior from
  metadata only; the task's own algorithms (SimCC, hierarchical resolver, SAM
  prompt encoding, ZSL matching, tile merging) live inside the pipeline, never
  behind a `model_type` branch.
- **YOLO family needs introspected values captured.** All models are re-exported
  with the new metadata regardless; `YOLO`/`YoloV4`, `YOLOF`, `YOLOX`,
  `YOLOv3-ONNX` and `YOLO-seg` additionally need their currently introspected or
  hardcoded parameters captured as explicit `decoder_constants` (see the callout
  under [`decoder_constants`](#decoder_constants)).

---

## Governance notes for external clients

- **Publish the schema as a standalone artifact** (JSON Schema + prose spec),
  versioned independently of the Python package, so a C++/Rust/JS team can validate
  `model_info` without any Python dependency. The artifact also includes the
  [Result schema definitions](#result-schema-definitions), which clients use to
  implement their own result types (result types are not part of `model_info`).
- **Ship conformance test vectors** — `(input, metadata, expected result)` tuples
  per `task` and per `decoder` value — so any implementation can prove parity with
  the Python reference. This replaces "read the Python to understand the pipeline".
- **Capability negotiation without a profile.** A client advertises the set of
  `task`s it supports and, within each, the `decoder` / `mask_postprocess` /
  `input_modality` / `stages` / `tiling` / `state` values it implements. If a model
  requires an unsupported combination, the client fails cleanly. `task` + explicit
  fields are authoritative; there is no `pipeline_profile`.
- **Keep `model_type` as a backward-compat alias.** When the new fields are absent,
  derive them from `model_type` via a documented expansion table; when present, the
  new fields are authoritative.
- **Unknown-field policy:** clients MUST ignore unknown keys within a major version
  (forward compatibility) but MUST reject unknown enum *values* in known fields
  (safety).
- **Feasibility.** All models are re-exported to carry this metadata. With these
  fields a generic per-task pipeline reproduces every model with no per-model
  branches. The decoders needing extra attention during export are the YOLO family
  (`YOLO`/`YoloV4`, `YOLOF`, `YOLOX`, `YOLOv3-ONNX`, `YOLO-seg`), whose currently
  introspected or hardcoded parameters must be captured as explicit
  `decoder_constants`; all other decoders (Ultralytics, YOLO-DETR, SSD, MaskRCNN,
  DETR-Seg, Anomaly, Keypoint, Action, Classification, Segmentation, SAM) carry
  their config directly. Tiling and SAM ZSL are covered by the
  [`tiling`](#tiling) and [`state`](#state-learn--infer-pipelines) fields.
- **ONNX and other runtimes.** Which decoders a runtime supports is part of the
  client's capability declaration, not an allow-list in the model loader (today
  `Model.__init__` restricts `ONNXRuntimeAdapter` to a fixed set of wrappers).

---

## Migration plan for existing Python modules

This section maps each `task` to the concrete modules in
`src/model_api/models/` (and `src/model_api/tilers/`) and describes how to
collapse the current per-model subclasses into **one generic pipeline per task**
driven by the metadata above.

Each plan follows the same shape:

- **Modules** — current files/classes involved.
- **Target** — the single generic pipeline class to introduce.
- **Metadata to emit** — fields the exporter must write into `model_info`.
- **Steps** — ordered refactor actions.
- **Removable logic** — hardcoded branches/heuristics deleted once metadata drives it.

A shared prerequisite for all tasks: implement the metadata reader on `Model`
(parse `task`, `stages`, `*_roles`, `output_layout`, `labels`, `decoder`,
`decoder_constants`, `tiling`, `state`, …), a shared preprocessing builder driven
by `resize_type` / `pad_value` / `interpolation_mode` / `input_dtype` / mean /
scale / `orig_*`, and a `model_type → metadata` back-compat expansion table so
un-migrated models keep loading. The ONNX adapter allow-list in `Model.__init__`
is replaced by per-decoder capability checks.

### Classification migration

- **Modules:** `classification.py` (`ClassificationModel`, `GreedyLabelsResolver`,
  `ProbabilisticLabelsResolver`, `SimpleLabelsGraph`); `action_classification.py`
  shares the `ClassificationResult`.
- **Target:** `ClassificationPipeline`.
- **Metadata to emit:** `decoder` ∈ {`topk`,`multilabel`,`hierarchical`};
  explicit `activation` ∈ {`softmax`,`sigmoid`,`none`}; `labels` (complete list,
  including `"other"` if the model has it); `topk`; `output_raw_scores`;
  `hierarchical_config` + `hierarchical_postproc`; `output_roles` for `logits`/
  `saliency_map`/`feature_vector`/`raw_scores`.
- **Steps:**
  1. Move multiclass/multilabel/hierarchical decoding into the pipeline keyed by
     `decoder`, not by `self.params.hierarchical`/`multilabel` flags.
  2. Keep the label-tree resolvers as pipeline-internal algorithms selected when
     `decoder == hierarchical`; feed them `hierarchical_config` from metadata;
     keep the saliency-map reordering for hierarchical heads.
  3. Replace `_get_non_xai_names` / `startswith` output sniffing with `output_roles`.
  4. Replace the runtime `is_softmaxed(...)` guess with the explicit `activation`.
- **Removable logic:** `_verify_single_output` shape checks and implicit
  `"other"` label insertion, `is_softmaxed` heuristic, `_setup_*` flag branching,
  XAI name string-matching.

### Segmentation migration

- **Modules:** `segmentation.py` (`SegmentationModel`,
  `create_hard_prediction_from_soft_prediction`, `_get_activation_map`,
  `get_contours`), `tilers/semantic_segmentation.py`.
- **Target:** `SegmentationPipeline`.
- **Metadata to emit:** `decoder = seg_argmax`; `decoder_constants.blur_strength`;
  `soft_threshold`; `return_soft_prediction`; `labels`; `derived_outputs`
  (`contours` if consumers need them); optional `tiling` (`merge = average`);
  `output_roles` for `seg`/`feature_vector`.
- **Steps:**
  1. Drive the 3D-argmaxed vs 4D-logits branch from `output_layout`/`output_roles`
     instead of inspecting `out_channels`.
  2. Move blur/threshold/resize into the pipeline, parameterized by
     `decoder_constants`.
  3. Always return `SegmentationResult` (leave `soft_prediction` empty when
     `return_soft_prediction = false`) instead of a bare array.
  4. Run `get_contours` as the `contours` derived output.
- **Removable logic:** `_get_outputs` shape-length branching; `out_channels`
  heuristic; the bare-array return path.

### Detection migration

- **Modules:** `detection_model.py` (`DetectionModel` base, `_resize_detections`,
  `_filter_detections`, NMS helpers), `ssd.py` (3 parsers), `yolo.py`
  (`YOLO`/`YoloV4`/tiny, `YOLOF`, `YOLOX`, `YoloV3ONNX`, `YOLOv5`/`YOLOv8`/`YOLO11`,
  `Params`, `ANCHORS`), `yolo_detr.py` (`YOLODETR`), `tilers/detection.py`.
- **Target:** `DetectionPipeline` with decoders selected by `decoder` ∈
  {`ssd_boxes`,`yolo_region`,`yolof`,`yolox`,`ultralytics`,`yolo_detr`,
  `in_model_nms`}.
- **Metadata to emit:** `decoder`; `box_format`; NMS group (incl.
  `nms_include_boundaries`); `confidence_threshold`; `resize_type`, `pad_value`,
  `interpolation_mode`, `scale_values`, `reverse_input_channels`, `orig_*`;
  `labels`; `output_roles` + `output_layout`; `input_roles.image_info` with
  `image_info_content`; YOLO-family `decoder_constants` (anchors / masks / grid /
  layout per output, YOLOF anchors, YOLOX strides, class counts); optional
  `tiling`.
- **Steps:**
  1. Keep `_resize_detections`/NMS/label-naming as shared pipeline utilities.
  2. Replace SSD's try/except parser selection with explicit `output_roles` +
     `box_format`.
  3. Port YOLO region decode into `yolo_region` (no `operations_by_type`, no
     `//32`, no `is_tiny`), YOLOF into `yolof`, and the YOLOX grid/stride decode
     into `yolox` (strides from metadata, top-left padding via `resize_type`).
  4. Collapse `YOLOv5`/`YOLOv8`/`YOLO11` into the `ultralytics` decoder (they are
     already identical).
  5. Port `YoloV3ONNX` into `in_model_nms` (gather by `nms_indices`, convert
     `yxyx`, fill `image_info` with the original size, no extra NMS).
  6. `YOLODETR` becomes the `yolo_detr` decoder as-is.
  7. Apply exactly one NMS pass (drop YOLO's extra `_filter` pass).
- **Removable logic:** `SingleOutputParser`/`MultipleOutputParser`/
  `BoxesLabelsParser` auto-detection; `YOLO._get_output_info` graph/shape
  introspection; `is_tiny`; anchor/mask default tables; YOLOX's custom
  `_resize_image`; YoloV3ONNX's forced CUBIC/no-normalization overrides and
  hardcoded `classes = 80`; per-wrapper `parameters()` default overrides (become
  exported values); precision/rank checks in constructors (become
  `output_layout`).

### Instance Segmentation migration

- **Modules:** `instance_segmentation.py` (`InstanceSegmentationModel`,
  `MaskRCNNModel`, `DETRInstanceSegmentation`, `_segm_postprocess`,
  `_full_image_mask_postprocess`, `_average_and_normalize`), `yolo_seg.py`
  (`YOLOSeg`), `utils.py` (`add_rotated_rects`, `get_contours`),
  `tilers/instance_segmentation.py`.
- **Target:** `InstanceSegmentationPipeline`.
- **Metadata to emit:** `mask_postprocess` ∈ {`box_crop`,`full_image`,`prototype`};
  `label_shift`; `labels`; `postprocess_semantic_masks`; `mask_threshold`; NMS
  group; `confidence_threshold`; `box_format`; `output_roles` + `output_layout`
  (incl. `predictions`/`protos` for YOLO-seg); `input_roles.image_info` for the
  segmentoly variant; YOLO-seg `decoder_constants`; optional `derived_outputs`
  (`rotated_rects`) and `tiling`.
- **Steps:**
  1. Replace the subclasses with one pipeline that calls `_segm_postprocess`
     (`box_crop`), `_full_image_mask_postprocess` (`full_image`) or the YOLO-seg
     prototype decode + `crop_mask` (`prototype`) per metadata.
  2. Drive `labels += _labels_shift` from `label_shift`, applied consistently to
     both `labels` and `label_names`.
  3. Replace `_get_outputs`/`_get_segmentoly_outputs`/YOLO-seg rank-based
     output detection with `output_roles`; detect segmentoly via `input_roles`
     not `len(inputs) == 2`.
  4. Produce `RotatedSegmentationResult` via the `rotated_rects` derived output.
- **Removable logic:** `MaskRCNNModel`/`DETRInstanceSegmentation`/`YOLOSeg`
  subclasses; `_labels_shift` class var and literal `+1`; shape-length output
  detection; `is_segmentoly` by input count; class count derived from shapes.

### Anomaly migration

- **Modules:** `anomaly.py` (`AnomalyDetection`, `_normalize`, `_get_boxes`).
- **Target:** `AnomalyPipeline`.
- **Metadata to emit:** `decoder = anomaly_map`; `intensity_mode = scale_to_unit`
  (replacing the custom `/255`); `image_threshold`/`pixel_threshold`;
  `decoder_constants.normalization_scale`/`map_scale`; `anomaly_mode` (renamed
  from the `task` parameter); `labels = [normal, anomalous]`; `output_roles` for
  the 4-output or single-map case.
- **Steps:**
  1. Select the "4 named outputs" vs "single map" branch from `output_roles`
     instead of the `all(key in outputs)` probe.
  2. Replace the `_input_transform` `/255` override with `scale_to_unit`.
  3. Keep `_normalize`/`_get_boxes` as pipeline internals parameterized by
     `decoder_constants` and `anomaly_mode`.
- **Removable logic:** custom `_input_transform`; output-key probing; the
  `task`-named parameter. The NPU dynamic-reshape workaround stays an adapter
  concern.

### Keypoint detection migration

- **Modules:** `keypoint_detection.py` (`KeypointDetectionModel`,
  `TopDownKeypointDetectionPipeline`, `_decode_simcc`, `_get_simcc_maximum`).
- **Target:** `KeypointPipeline` (single stage) and a `detect_then_crop`
  orchestration for the top-down case.
- **Metadata to emit:** `decoder = simcc`; `apply_softmax`;
  `decoder_constants` (`simcc_split_ratio`, `decode_beta`, `sigma`); `output_roles`
  for `simcc_x`/`simcc_y`; `stages` = `detector`→`pose` for top-down.
- **Steps:**
  1. Move SimCC decode into the pipeline as the `simcc` decoder, reading constants
     from metadata (no hardcoded `2.0`/`150.0`/`6.0`).
  2. Express the top-down flow via `stages`; the crop-and-remap glue stays as the
     orchestration's own logic and returns `KeypointResultList`.
- **Removable logic:** hardcoded SimCC constants; manual `resize_info` naming (use
  shared `ResizeMetadata`).

### Action classification migration

- **Modules:** `action_classification.py` (`ActionClassificationModel`).
- **Target:** folded into `ClassificationPipeline` via `input_modality = video`.
- **Metadata to emit:** `input_modality = video`; `input_layout` ∈
  {`NSCTHW`,`NSTHWC`}; `temporal.clip_size`; `mean_values`/`scale_values`;
  `labels`; `decoder = action_logits` (argmax); `embedded_processing = false`.
- **Steps:**
  1. Generalize preprocessing to read `input_modality`/`input_layout` so the 6D
     video path is selected by metadata, not a dedicated class.
  2. Reuse the classification `argmax` decode.
- **Removable logic:** `ActionClassificationModel` class, `_get_inputs` 6D check,
  bespoke `_change_layout`.

### Visual prompting migration

- **Modules:** `visual_prompting.py` (`SAMVisualPrompter`,
  `SAMLearnableVisualPrompter`, `Prompt`, `VisualPromptingFeatures`,
  `_polygon_to_mask`, `_generate_masked_features`), `sam_models.py`
  (`SAMImageEncoder`, `SAMDecoder`).
- **Target:** `VisualPromptingPipeline` with a fixed `encoder`→`decoder`
  topology from `stages`, plus an optional `learn`/`infer` mode driven by `state`.
- **Metadata to emit:** `stages` (encoder + decoder, `consumes`); `decoder` ∈
  {`sam_decode`, `sam_zsl`}; `decoder_constants` (`image_size`, `mask_threshold`,
  `embed_dim`, `mask_input_shape`, point-label conventions, ZSL constants);
  `input_roles` for decoder tensors (`image_embeddings`, `point_coords`,
  `point_labels`, `mask_input`, `has_mask_input`, `orig_size`) and user prompts
  (`boxes`/`points`/`polygons`/`labels`); `resize_type = longest_side`;
  `zsl_match_threshold` and `state` for ZSL.
- **Steps:**
  1. Keep encoder and decoder as two declared stages wired via `stages`.
  2. Move the prompt-tensor construction (point-label `[2,3]`/`[1]` conventions,
     `mask_input`/`has_mask_input`, `apply_coords`) into the pipeline as its own
     algorithm, parameterized by `decoder_constants`.
  3. Return `VisualPromptingResult` with the full field set defined above.
  4. Implement ZSL as the `sam_zsl` decoder with the `learn`/`infer` phases from
     `state`; polygon prompts rasterize directly; reference features are
     serializable `VisualPromptingFeatures`.
- **Removable logic:** per-class `base_preprocess`/`postprocess` overrides on the
  SAM models; `_check_io_number` no-ops; hardcoded prompt conventions and ZSL
  constants (`0.65`, `num_bg_points`, `downsizing = 64`, reference threshold
  `0.3`) move into documented metadata.

### Tiling migration

- **Modules:** `tilers/tiler.py` (`Tiler`), `tilers/detection.py`
  (`DetectionTiler`), `tilers/instance_segmentation.py`
  (`InstanceSegmentationTiler`), `tilers/semantic_segmentation.py`
  (`SemanticSegmentationTiler`).
- **Target:** one `TilingWrapper` that wraps any Detection, Instance
  Segmentation or Segmentation pipeline.
- **Metadata to emit:** `tiling` object (`tile_size`, `tiles_overlap`,
  `tile_with_full_img`, `max_pred_number`, `iou_threshold`, `merge`).
- **Steps:**
  1. Keep tile generation in the wrapper, driven by `tile_size`/`tiles_overlap`.
  2. Select the merge strategy from `tiling.merge` (`nms` for boxes/instances,
     `average` for segmentation soft predictions) instead of from the tiler
     subclass.
  3. Read tiler parameters from the model's `model_info` instead of a separate
     tiler configuration.
- **Removable logic:** the three task-specific tiler subclasses; task detection
  by result type.

