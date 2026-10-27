# Model Metadata Specification

A language-agnostic, declarative contract that describes how a model must be
**preprocessed**, **run**, and **postprocessed**, and what its **outputs** mean.

The goal of this specification is to remove `model_type` as the single
discriminator that selects a Python `Model` subclass. Instead, each concern
(preprocessing, inference I/O, postprocessing, output interpretation) is
described by explicit metadata so that **one generic pipeline per `task`**
(classification, detection, instance segmentation, segmentation) — or
any client that does **not** share the `model_api` codebase (C++, Rust, JS, …) —
can reproduce the exact pipeline from the metadata alone.

**Design goal — one generic pipeline per task.** The aim is *not* a single
universal engine that covers every task with one code path. It is a **separate
generic pipeline per `task`**, where:

- the pipeline MAY implement the task's own algorithms (e.g. the YOLO decode, the
  SSD box decode) **once**, inside the pipeline;
- the pipeline MUST NOT branch on model identity (`if model_type == …`) — **all
  per-model variation is expressed as metadata values** (`decoder`,
  `mask_postprocess`, `activation`, `decoder_constants`, …) that the pipeline reads.

Under this goal every current task is metadata-drivable. All models are re-exported
to carry this metadata; the decoder that needs **extra care** during that export
is the YOLO family (`YOLOX`, Ultralytics `YOLOv5`/`v8`/`11`), whose wrappers
currently derive parameters at load time by **inspecting the OpenVINO graph or
tensor shapes**, or hardcode them (see [`decoder_constants`](#decoder_constants)).
Those values simply have to be captured explicitly as metadata during the same
re-export, since static RT-info cannot reproduce the introspection.

All properties live under the `model_info` runtime-info block of the model and
are stored as strings (JSON-encoded where a structured value is needed), since
OpenVINO RT-info only persists string values.

---

## Table of Contents

1. [Pipeline stages](#pipeline-stages)
2. [Compatibility properties](#compatibility-properties)
   - [`metadata_schema_version`](#metadata_schema_version)
   - [`task`](#task)
3. [Preprocessing properties](#preprocessing-properties)
   - [`input_layout`](#input_layout)
   - [`embedded_processing`](#embedded_processing)
   - [`resize_type`](#resize_type)
   - [`input_dtype` / `interpolation_mode` / `pad_value`](#input_dtype--interpolation_mode--pad_value)
   - [`orig_height` / `orig_width` / `input_frame_height` / `input_frame_width`](#orig_height--orig_width--input_frame_height--input_frame_width)
   - [`mean_values` / `scale_values` / `reverse_input_channels`](#mean_values--scale_values--reverse_input_channels)
4. [Input / output contract](#input--output-contract)
   - [`input_roles`](#input_roles)
   - [`output_roles`](#output_roles)
   - [`output_layout`](#output_layout)
5. [Postprocessing properties](#postprocessing-properties)
   - [`activation`](#activation)
   - [`decoder`](#decoder)
   - [`box_format`](#box_format)
   - [`label_shift`](#label_shift)
   - [`mask_postprocess`](#mask_postprocess)
   - [`nms` and NMS parameters](#nms-and-nms-parameters)
   - [Threshold parameters](#threshold-parameters)
   - [Task-specific output options](#task-specific-output-options)
   - [`decoder_constants`](#decoder_constants)
6. [Output description](#output-description)
   - [Result types (not metadata)](#result-types-not-metadata)
   - [Result schema definitions](#result-schema-definitions)
   - [`labels`](#labels)
   - [`output_coordinate_space`](#output_coordinate_space)
   - [`output_value_range`](#output_value_range)
   - [`derived_outputs`](#derived_outputs)
7. [Per-model mapping table](#per-model-mapping-table)
   - [Additional wrappers](#additional-wrappers)
8. [Governance notes for external clients](#governance-notes-for-external-clients)
9. [Migration plan for existing Python modules](#migration-plan-for-existing-python-modules)
    - [Classification](#classification-migration)
    - [Segmentation](#segmentation-migration)
    - [Detection](#detection-migration)
    - [Instance Segmentation](#instance-segmentation-migration)

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
  `instance_segmentation`.
- **Description:** The semantic family of the model. Determines which task
  pipeline runs and therefore which result type is returned (see
  [Result types](#result-types-not-metadata)).
- **What it does:** Selects the output data structure and the high-level contract,
  and is the primary token a client negotiates on. It does **not** by itself fix
  the decoder — that comes from `decoder` and the other explicit
  fields. Replaces the heuristic in `Model.detect_model_type`.
- **Used by:** **All models**, with a distinct value each: Classification →
  `classification`, Segmentation → `segmentation`, SSD/YOLO family → `detection`,
  MaskRCNN/DETRInstSeg → `instance_segmentation`. Needed because it is the primary
  capability-negotiation token and picks the result family.
- **Required:** Yes.

---


## Preprocessing properties

### `input_layout`

- **Stage:** Pre
- **Type:** enum string
- **Possible values:** `NCHW`, `NHWC`.
- **Description:** The axis order of the model input tensor.
- **What it does:** Determines the layout transform applied after resize/normalize.
- **Used by:** **All models.** Needed so the client transposes the preprocessed
  tensor to match the model input.
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
     client builds graph-side preprocessing from the resize/mean/scale
     fields. This is what `model_api` does today for most static image models.
  3. `embedded_processing = false` and the client **cannot** embed (dynamic shapes,
     runtimes without PPP) — the client performs the full chain in host code.

  Cases 2 and 3 MUST produce identical tensors; the choice is a client
  implementation detail, not model metadata.
- **Used by:** **All models.** Models exported with PPP use `true`.
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
  - `crop` — center crop after aspect-preserving resize.

  `RESIZE_TYPES` in `adapters/utils.py` is the authoritative list of modes
  implemented today; `fit_to_window_top_left` formalizes logic currently hardcoded
  in `YOLOX._resize_image`.
- **Description:** Strategy used to resize the input to the model's spatial size.
- **What it does:** Defines padding/scaling behavior. Each mode's exact padding
  rule must be specified so coordinate inversion in postprocessing matches
  (`ResizeMetadata` today).
- **Used by:** **All image models**, and it must agree with box/mask rescaling.
  YOLO family / Ultralytics use `fit_to_window_letterbox`, YOLOX
  `fit_to_window_top_left`, Detection/Instance Seg usually `fit_to_window`.
- **Required:** Yes (needed for coordinate inversion even when preprocessing is
  embedded).

### `input_dtype` / `interpolation_mode` / `pad_value`

- **Stage:** Pre
- **Type:** enum string / enum string / integer
- **Possible values:**
  - `input_dtype`: `u8` (default), `f32`.
  - `interpolation_mode`: `LINEAR` (default), `CUBIC`, `NEAREST`.
  - `pad_value`: `0..65535` (default `0`; Ultralytics and YOLOX use `114`).
- **Description:** Element type of the image fed to the pipeline, the resize
  interpolation, and the fill value for padded regions.
- **What it does:** Ensures bit-exact preprocessing across clients. Today some of
  these are hardcoded in wrappers (YOLOX pads with `114` in `_resize_image`), which
  must become metadata.
- **Used by:** **All image models**; `pad_value` matters for every padding
  `resize_type`; `interpolation_mode = CUBIC` for transformer classifiers that
  require it.
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
  (`ResizeMetadata.compute(model_width=orig_width, …)`) for boxes and masks when
  preprocessing is embedded. `input_frame_*` lets a client compile a
  static graph (GPU/NPU) instead of a dynamic one.
- **Used by:** **Detection, Instance Segmentation** — any model that maps spatial
  outputs back; `input_frame_*` for any model deployed on static-shape devices.
- **Required:** `orig_*` required when `embedded_processing = true` and outputs
  are spatial; `input_frame_*` optional.


### `mean_values` / `scale_values` / `reverse_input_channels`

- **Stage:** Pre
- **Type:** list[float] / list[float] / boolean
- **Possible values:** per-channel floats; booleans `true`/`false`.
- **Description:** Standard normalization: `(pixel - mean) / scale`, with optional
  BGR↔RGB channel reversal.
- **What it does:** Brings input into the distribution the model was trained on.
  Ultralytics use `scale_values = [255]` (unit scaling) and YOLOv5/8/11 use
  `reverse_input_channels = true`; these are currently wrapper defaults and MUST be
  written explicitly at export.
- **Used by:** **All image models.** Applied in host code for case-3 clients
  (see [`embedded_processing`](#embedded_processing)) and in the graph otherwise.
- **Required:** Optional (defaults: empty lists, `false`).


---

## Input / output contract

### `input_roles`

- **Stage:** Infer / Pre
- **Type:** JSON object `{ role: tensor_name }`
- **Possible roles:** `image`, `image_info`.
- **Description:** Maps logical input roles to the actual model input tensor names.
- **What it does:** Removes shape-based input sniffing (e.g. "4D tensor = image,
  2D tensor = image_info"). A non-Python client can bind inputs directly by name.
  Roles with non-obvious content carry a qualifier:
  - `image_info` MUST declare its content via `image_info_content` ∈
    {`resized_hw1` (`[[h, w, 1]]` of the **model input**, SSD / MaskRCNN
    segmentoly)}.
- **Used by:** **All models.** Critical for **SSD/MaskRCNN segmentoly**
  (`image_info` content); single-input models still declare the `image` role for
  uniform binding.
- **Required:** Yes.

### `output_roles`

- **Stage:** Infer / Post
- **Type:** JSON object `{ role: tensor_name }`
- **Possible roles:** `logits`, `boxes`, `labels`, `scores`, `masks`, `seg`,
  `saliency_map`, `feature_vector`, `raw_scores`, `predictions` (fused
  box/score/class tensor — Ultralytics, YOLOX).
- **Description:** Maps logical output roles to actual model output tensor names.
- **What it does:** Removes fragile heuristics such as shape-length branching,
  `startswith("TopK")` filtering, `find_layer_by_name`, and
  `next(iter(outputs.values()))`. XAI outputs (`saliency_map`, `feature_vector`)
  become declared rather than string-matched.
- **Used by:** **All models.** Essential for multi-output models —
  **Instance Segmentation** (boxes/labels/masks), and
  **Detection/Classification/Segmentation** with optional XAI
  (`saliency_map`, `feature_vector`).
- **Required:** Yes.

### `output_layout`

- **Stage:** Post
- **Type:** JSON object `{ role: "<dtype>:<shape_grammar>" }`
- **Possible values (examples):** `"f32:1x1xNx7"` (SSD single output),
  `"f32:Nx5"` (boxes+score), `"f32:1x(4+C)xN"` (Ultralytics, transposed),
  `"f32:1xNx(5+C)"` (YOLOX).
- **Description:** Documented dtype and shape grammar for each output role.
- **What it does:** Lets a client validate and index tensors deterministically and
  replaces the shape/precision checks currently scattered in constructors (e.g.
  Ultralytics requires `f32` and rank 3). Symbols `C` (classes) MUST be resolvable
  from `labels` / `decoder_constants`, not inferred from shapes.
- **Used by:** **Detection** (every decoder), **Instance Segmentation** (mask
  stacks). Informational for simple single-vector classifiers.
- **Required:** Required for detection / instance segmentation decoders;
  recommended otherwise.

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
  **Segmentation** (argmax), **Instance Segmentation** and YOLO masks
  (sigmoid/threshold); **YOLO family** uses `none` for Ultralytics/YOLOX.
- **Required:** Yes for score-producing tasks.

### `decoder`

- **Stage:** Post
- **Type:** enum string
- **Possible values:**
  - Classification: `topk`, `multilabel`.
  - Segmentation: `seg_argmax`.
  - Detection: `ssd_boxes` (SSD single / multi / boxes-labels outputs, also used by
    the DETR-based detectors exported as SSD), `yolox` (anchor-free grid × stride
    decode), `ultralytics` (YOLOv5/v8/11, `[1,4+C,N]`, no objectness).
  - Instance segmentation: `maskrcnn_crop`, `detr_fullmask`.
  - Other: `none`.
- **Description:** The named algorithm that turns activated outputs into the result
  structure.
- **What it does:** Selects the decoding routine previously encoded by the Python
  class name (e.g. `SSD` vs `YOLO`, `MaskRCNN` vs `DETRInstSeg`). Each value MUST
  be defined to test-vector precision in the spec. These values are **variation
  points inside a single task pipeline**, not separate model implementations: the
  task pipeline implements each named decoder once and selects it from this field;
  it never branches on `model_type`.
- **Used by:** **All models**, and it is the field that disambiguates same-`task`
  variants: SSD vs the YOLO-family detection decoders; MaskRCNN vs DETRInstSeg.
  Classification variants use `topk`/`multilabel`.
- **Required:** Yes.

### `box_format`

- **Stage:** Post
- **Type:** enum string
- **Possible values:** `xyxy_norm`, `xyxy_pixel`, `cxcywh_pixel`. `_norm` =
  normalized to model input size, `_pixel` = model-input pixels.
- **Description:** Encoding of bounding boxes in the raw output.
- **What it does:** Tells the decoder how to interpret and convert box coordinates
  to `xyxy` before rescaling into `output_coordinate_space`.
- **Used by:** **Detection** and **Instance Segmentation** — any model emitting
  boxes: SSD (`xyxy_norm`), Ultralytics/YOLOX (`cxcywh_pixel`),
  MaskRCNN/DETRInstSeg (`xyxy_pixel`).
- **Required:** Required for detection / instance segmentation.

### `label_shift`

- **Stage:** Post
- **Type:** integer
- **Possible values:** `0`, `1` (typically).
- **Description:** Constant added to raw class indices to obtain final label ids.
- **What it does:** Aligns model output indices with the label list. Currently the
  hidden `_labels_shift` (1 for MaskRCNN / DETRInstSeg). The shifted index is used
  for **both** `labels` and `label_names`.
- **Used by:** **MaskRCNN**, **DETRInstSeg** (`1`, background at index 0); all
  other models use `0`.
- **Required:** Optional (defaults to `0`).

### `mask_postprocess`

- **Stage:** Post
- **Type:** enum string
- **Possible values:** `none`, `box_crop`, `full_image`.
- **Description:** How a raw instance mask is expanded into image space.
- **What it does:** `box_crop` resizes a small per-box mask (e.g. 28×28) into the
  bounding box (MaskRCNN); `full_image` resizes a whole-image mask to the original
  dimensions and thresholds it (DETR-Seg).
- **Used by:** **MaskRCNN** (`box_crop`), **DETRInstSeg** (`full_image`). Not
  applicable to other tasks.
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
  single pass.
- **Used by:** **SSD / YOLO family Detection** and **Instance Segmentation**
  (MaskRCNN/DETRInstSeg). Irrelevant to classification and segmentation.
- **Required:** Optional (detection/instance seg only).

### Threshold parameters

- **Stage:** Post
- **Type:** float
- **Properties:**
  - `confidence_threshold` (default `0.5`) — filter low-confidence predictions
    (detection, instance seg, multilabel classification).
  - `soft_threshold` (default `-inf`) — per-pixel confidence for segmentation.
  - `mask_threshold` — binarization cutoff for decoded masks (instance seg `0.5`).
- **Description:** Confidence cutoffs applied during decoding.
- **What it does:** Controls which predictions/pixels survive into the result.
- **Used by:** `confidence_threshold` → **Detection, Instance Seg,
  multilabel Classification**; `soft_threshold` → **Segmentation**;
  `mask_threshold` → **Instance Seg**. Each task uses only the thresholds relevant
  to its decoder.
- **Required:** Optional; depends on `task`.

### Task-specific output options

- **Stage:** Post / Output
- **Type:** per-task parameter group (already present in `ParameterRegistry`;
  listed here so external clients implement them).
- **Properties:**
  - **Classification:** `topk` (int ≥ 1, default `1`) — number of labels returned
    by `topk`; `output_raw_scores` (bool) — fill `raw_scores` with the full
    activated vector.
  - **Segmentation:** `return_soft_prediction` (bool, default `true`) — fill
    `soft_prediction`.
  - **Instance segmentation:** `postprocess_semantic_masks` (bool, default
    `true`) — `true`: masks expanded to full image per `mask_postprocess`;
    `false`: raw per-instance masks are returned.
- **Description:** Switches that change *which* result fields are produced or how
  they are resolved, without changing the decoder algorithm.
- **What it does:** Lets one task pipeline serve all output variants of a task.
- **Used by:** as listed per task.
- **Required:** Optional (defaults as listed).

### `decoder_constants`

- **Stage:** Post
- **Type:** JSON object (free-form, decoder-specific)
- **Possible values (examples):**
  - MaskRCNN: `{ "expand_box_scale": "auto", "mask_threshold": 0.5 }`
  - DETR-Seg: `{ "mask_threshold": 0.5 }`
  - Segmentation: `{ "blur_strength": -1 }`
  - YOLOX: `{ "strides": [8, 16, 32] }`
- **Description:** Externalizes constants that are currently hardcoded in Python and
  therefore invisible to non-shared-codebase clients.
- **What it does:** Provides every numeric constant a client needs to reproduce
  mask or YOLO-family decoding exactly.
- **Used by:** **MaskRCNN/DETRInstSeg** (mask constants), **Segmentation**
  (`blur_strength`), **YOLO family** (grids / strides / class count). Not needed by
  plain Classification.
- **Required:** Required whenever the chosen `decoder` relies on such constants.

> **YOLO family — values to capture at export.** All models are re-exported to
> carry the new metadata, so re-export itself is not YOLO-specific. What *is*
> specific is where these wrappers get their parameters today:
> - `YOLOX`: strides `[8, 16, 32]` and grids built from the input size.
>
> A metadata-only detection pipeline **cannot** perform this introspection, so
> these fields MUST be captured explicitly in `decoder_constants` (and output
> order in `output_roles`). Ultralytics (v5/8/11) need only `labels` and
> `output_layout`.

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
| `task = classification` | `ClassificationResult` |
| `task = segmentation` | `SegmentationResult` |
| `task = detection` | `DetectionResult` |
| `task = instance_segmentation` | `InstanceSegmentationResult`; `RotatedSegmentationResult` when `derived_outputs` contains `rotated_rects` |

Optional fields inside a result type are controlled by other metadata
(e.g. `return_soft_prediction`, `output_raw_scores`, XAI `output_roles`) and are
left empty when not produced; the result type itself never changes.

### Result schema definitions

> **Mapping to current Python classes.** The schema names below are documentation
> names for the normative, language-agnostic contract; they are not written into
> model metadata. They map to the existing `model_api` result classes
> as follows: `SegmentationResult` → `ImageResultWithSoftPrediction`; the others
> keep the same name (`ClassificationResult`, `DetectionResult`,
> `InstanceSegmentationResult`, `RotatedSegmentationResult`). External clients MUST
> rely on the field definitions here, not on the Python class names.

Each schema below lists its fields as `name: type [shape] — meaning`. `N` = number
of instances/detections, `H`/`W` = original image height/width, `C` = number of
classes. Optional fields MAY be empty arrays when the model does not produce them
(e.g. XAI maps).

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


### `labels`

- **Stage:** Post / Output
- **Type:** list[string]
- **Possible values:** ordered list of class names.
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
  - **Missing labels.** Names are empty strings; indices are still returned.
- **Used by:** **All models that output classes** — Classification,
  Segmentation (contour labels), Detection, Instance Segmentation.
- **Required:** Required whenever names are expected in the result; strongly
  recommended for all class-producing models.

### `output_coordinate_space`

- **Stage:** Output
- **Type:** enum string
- **Possible values:** `original_image_pixels`, `normalized`, `n/a`.
- **Description:** The coordinate frame of spatial outputs (boxes, masks).
- **What it does:** Tells the client how to interpret returned coordinates. For
  example, detection boxes are already rescaled to original image pixels, so the
  client must not rescale them again.
- **Used by:** **Detection, Instance Segmentation** — every model with spatial
  outputs. Not needed by plain Classification (no coordinates).
- **Required:** Required when the result contains spatial data.

### `output_value_range`

- **Stage:** Output
- **Type:** JSON object `{ role: range_descriptor }`
- **Possible values (examples):**
  - `{ "scores": "f32[0,1]" }`
  - `{ "masks": "u8{0,1}" }`
- **Description:** The dtype and value range of each output role.
- **What it does:** Enables correct interpretation/rendering of raw output tensors
  without reading reference code.
- **Used by:** Most notably **Instance Segmentation** (masks are `{0,1}`); useful
  for all models so clients render/threshold outputs correctly.
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


## Per-model mapping table

| Property | Classification | Segmentation | SSD / Detection | MaskRCNN | DETRInstSeg |
|---|---|---|---|---|---|
| `task` | classification | segmentation | detection | instance_segmentation | instance_segmentation |
| result type (implied, not metadata) | `ClassificationResult` | `SegmentationResult` | `DetectionResult` | `InstanceSegmentationResult` (`Rotated…` with `rotated_rects`) | `InstanceSegmentationResult` |
| `labels` | class list (+`"other"` rule) | class list | class list | background + classes | background + classes |
| `input_layout` | NCHW/NHWC | NCHW/NHWC | NCHW/NHWC | NCHW/NHWC | NCHW/NHWC |
| `input_roles` | image | image | image (+`image_info`=`resized_hw1` opt.) | image (+`image_info`=`resized_hw1` if segmentoly) | image |
| `output_roles` | logits (+saliency, feature, raw_scores) | seg (+feature_vector) | boxes/labels/scores (+saliency, feature) | boxes/labels/masks (+scores, saliency, feature) | boxes/labels/masks |
| `embedded_processing` | per export | per export | per export | per export | per export |
| `resize_type` | standard | standard | standard / fit_to_window | fit_to_window | fit_to_window |
| `activation` | softmax / sigmoid / none (explicit) | argmax (+optional blur/thresh) | none (decoded in graph) | — | — |
| `decoder` | topk / multilabel | seg_argmax | ssd_boxes | maskrcnn_crop | detr_fullmask |
| `box_format` | — | — | xyxy_norm | xyxy_pixel | xyxy_pixel |
| `label_shift` | 0 | 0 | 0 | **1** | **1** |
| `mask_postprocess` | — | — | — | box_crop | full_image |
| `nms` | — | — | configurable | configurable | configurable |
| thresholds | `confidence_threshold` (multilabel) | `soft_threshold` | `confidence_threshold` | `confidence_threshold`, `mask_threshold` | `confidence_threshold`, `mask_threshold` |
| task options | `topk`, `output_raw_scores` | `return_soft_prediction` | — | `postprocess_semantic_masks` | `postprocess_semantic_masks` |
| `decoder_constants` | — | `blur_strength` | — | expand_box scale, mask>0.5 | mask>0.5 |
| `derived_outputs` | — | `contours` (opt.) | — | `rotated_rects` (opt.) | `rotated_rects` (opt.) |
| `output_coordinate_space` | n/a | original_image_pixels | original_image_pixels | original_image_pixels | original_image_pixels |
| `output_value_range` | scores [0,1] | labels int / soft u8 | scores [0,1] | masks {0,1}, scores [0,1] | masks {0,1} |

### Additional wrappers

Wrappers not shown above, listed by the fields where they differ from their
task's column.

| Property | YOLOX | YOLOv5 / v8 / 11 (Ultralytics) |
|---|---|---|
| `task` | detection | detection |
| `decoder` | `yolox` | `ultralytics` |
| `input_roles` | image | image |
| `output_roles` | `predictions` | `predictions` |
| `output_layout` | `f32:1xNx(5+C)` | `f32:1x(4+C)xN` |
| `resize_type` | fit_to_window_top_left | fit_to_window_letterbox |
| `pad_value` / `interpolation_mode` | 114 / LINEAR | 114 / LINEAR |
| `scale_values` / `reverse_input_channels` | per-model | `[255]` / true |
| `activation` | none | none |
| `box_format` | `cxcywh_pixel` | `cxcywh_pixel` |
| `label_shift` | 0 | 0 |
| `nms` | agnostic, `nms_include_boundaries`, IoU 0.65 | per-class, IoU 0.7, max 30000 |
| `confidence_threshold` | 0.5 | 0.25 |
| `decoder_constants` | strides `[8,16,32]` | — |
| result type (implied, not metadata) | `DetectionResult` | `DetectionResult` |

### Key observations

- **`label_shift`** is `1` only for instance-segmentation decoders that reserve a
  background index (`MaskRCNN`, `DETRInstSeg`) — today encoded as `_labels_shift`.
- **`mask_postprocess`** is the main difference between the two
  instance-segmentation decoders (`box_crop` / `full_image`).
- **Preprocessing location is a client choice**, not model metadata: only models
  with `embedded_processing = true` have it baked in; for the rest a client either
  builds graph PPP at load time or runs host-side code, with identical results.
- **`decoder_constants`** is where the currently-invisible Python constants live;
  without externalizing these, a non-shared-codebase client cannot reproduce mask
  or YOLO-family outputs.
- **One generic pipeline per task.** Every task pipeline selects behavior from
  metadata only; the task's own algorithms (box decode, mask expansion) live inside
  the pipeline, never behind a `model_type` branch.
- **YOLO family needs introspected values captured.** All models are re-exported
  with the new metadata regardless; `YOLOX` additionally needs its currently
  introspected or hardcoded parameters captured as explicit `decoder_constants`
  (see the callout under [`decoder_constants`](#decoder_constants)).

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
  `task`s it supports and, within each, the `decoder` / `mask_postprocess`
  values it implements. If a model requires an unsupported combination,
  the client fails cleanly. `task` + explicit fields are authoritative; there is no
  `pipeline_profile`.
- **Keep `model_type` as a backward-compat alias.** When the new fields are absent,
  derive them from `model_type` via a documented expansion table; when present, the
  new fields are authoritative.
- **Unknown-field policy:** clients MUST ignore unknown keys within a major version
  (forward compatibility) but MUST reject unknown enum *values* in known fields
  (safety).
- **Feasibility.** All models are re-exported to carry this metadata. With these
  fields a generic per-task pipeline reproduces every model with no per-model
  branches. The decoder needing extra attention during export is the YOLO family
  (`YOLOX`), whose currently introspected or hardcoded parameters must be captured
  as explicit `decoder_constants`; all other decoders (Ultralytics, SSD, MaskRCNN,
  DETR-Seg, Classification, Segmentation) carry their config directly.
- **ONNX and other runtimes.** Which decoders a runtime supports is part of the
  client's capability declaration, not an allow-list in the model loader (today
  `Model.__init__` restricts `ONNXRuntimeAdapter` to a fixed set of wrappers).

---

## Migration plan for existing Python modules

This section maps each `task` to the concrete modules in
`src/model_api/models/` and describes how to
collapse the current per-model subclasses into **one generic pipeline per task**
driven by the metadata above.

Each plan follows the same shape:

- **Modules** — current files/classes involved.
- **Target** — the single generic pipeline class to introduce.
- **Metadata to emit** — fields the exporter must write into `model_info`.
- **Steps** — ordered refactor actions.
- **Removable logic** — hardcoded branches/heuristics deleted once metadata drives it.

A shared prerequisite for all tasks: implement the metadata reader on `Model`
(parse `task`, `*_roles`, `output_layout`, `labels`, `decoder`,
`decoder_constants`, …), a shared preprocessing builder driven
by `resize_type` / `pad_value` / `interpolation_mode` / `input_dtype` / mean /
scale / `orig_*`, and a `model_type → metadata` back-compat expansion table so
un-migrated models keep loading. The ONNX adapter allow-list in `Model.__init__`
is replaced by per-decoder capability checks.

### Classification migration

- **Modules:** `classification.py` (`ClassificationModel`).
- **Target:** `ClassificationPipeline`.
- **Metadata to emit:** `decoder` ∈ {`topk`,`multilabel`};
  explicit `activation` ∈ {`softmax`,`sigmoid`,`none`}; `labels` (complete list,
  including `"other"` if the model has it); `topk`; `output_raw_scores`;
  `output_roles` for `logits`/`saliency_map`/`feature_vector`/`raw_scores`.
- **Steps:**
  1. Move multiclass/multilabel decoding into the pipeline keyed by `decoder`, not
     by `self.params.multilabel` flags.
  2. Replace `_get_non_xai_names` / `startswith` output sniffing with `output_roles`.
  3. Replace the runtime `is_softmaxed(...)` guess with the explicit `activation`.
- **Removable logic:** `_verify_single_output` shape checks and implicit
  `"other"` label insertion, `is_softmaxed` heuristic, `_setup_*` flag branching,
  XAI name string-matching.

### Segmentation migration

- **Modules:** `segmentation.py` (`SegmentationModel`,
  `create_hard_prediction_from_soft_prediction`, `_get_activation_map`,
  `get_contours`).
- **Target:** `SegmentationPipeline`.
- **Metadata to emit:** `decoder = seg_argmax`; `decoder_constants.blur_strength`;
  `soft_threshold`; `return_soft_prediction`; `labels`; `derived_outputs`
  (`contours` if consumers need them); `output_roles` for `seg`/`feature_vector`.
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
  (`YOLOX`, `YOLOv5`/`YOLOv8`/`YOLO11`, `Params`).
- **Target:** `DetectionPipeline` with decoders selected by `decoder` ∈
  {`ssd_boxes`,`yolox`,`ultralytics`}.
- **Metadata to emit:** `decoder`; `box_format`; NMS group (incl.
  `nms_include_boundaries`); `confidence_threshold`; `resize_type`, `pad_value`,
  `interpolation_mode`, `scale_values`, `reverse_input_channels`, `orig_*`;
  `labels`; `output_roles` + `output_layout`; `input_roles.image_info` with
  `image_info_content`; YOLOX `decoder_constants` (strides, class counts).
- **Steps:**
  1. Keep `_resize_detections`/NMS/label-naming as shared pipeline utilities.
  2. Replace SSD's try/except parser selection with explicit `output_roles` +
     `box_format` (the DETR-based detectors exported as SSD use this path).
  3. Port the YOLOX grid/stride decode into `yolox` (strides from metadata,
     top-left padding via `resize_type`).
  4. Collapse `YOLOv5`/`YOLOv8`/`YOLO11` into the `ultralytics` decoder (they are
     already identical).
  5. Apply exactly one NMS pass (drop YOLO's extra `_filter` pass).
- **Removable logic:** `SingleOutputParser`/`MultipleOutputParser`/
  `BoxesLabelsParser` auto-detection; `YOLO._get_output_info` graph/shape
  introspection; YOLOX's custom `_resize_image`; per-wrapper `parameters()` default
  overrides (become exported values); precision/rank checks in constructors (become
  `output_layout`).

### Instance Segmentation migration

- **Modules:** `instance_segmentation.py` (`InstanceSegmentationModel`,
  `MaskRCNNModel`, `DETRInstanceSegmentation`, `_segm_postprocess`,
  `_full_image_mask_postprocess`, `_average_and_normalize`),
  `utils.py` (`add_rotated_rects`, `get_contours`).
- **Target:** `InstanceSegmentationPipeline`.
- **Metadata to emit:** `mask_postprocess` ∈ {`box_crop`,`full_image`};
  `label_shift`; `labels`; `postprocess_semantic_masks`; `mask_threshold`; NMS
  group; `confidence_threshold`; `box_format`; `output_roles` + `output_layout`;
  `input_roles.image_info` for the segmentoly variant; optional `derived_outputs`
  (`rotated_rects`).
- **Steps:**
  1. Replace the subclasses with one pipeline that calls `_segm_postprocess`
     (`box_crop`) or `_full_image_mask_postprocess` (`full_image`) per metadata.
  2. Drive `labels += _labels_shift` from `label_shift`, applied consistently to
     both `labels` and `label_names`.
  3. Replace `_get_outputs`/`_get_segmentoly_outputs` output detection with
     `output_roles`; detect segmentoly via `input_roles` not `len(inputs) == 2`.
  4. Produce `RotatedSegmentationResult` via the `rotated_rects` derived output.
- **Removable logic:** `MaskRCNNModel`/`DETRInstanceSegmentation` subclasses;
  `_labels_shift` class var; shape-length output detection; `is_segmentoly` by
  input count; class count derived from shapes.


