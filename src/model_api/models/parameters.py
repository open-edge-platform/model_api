#
# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Parameter registry and decorators for model configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from model_api.adapters.utils import RESIZE_TYPES
from model_api.models.types import BooleanValue, ListValue, NumericalValue, StringValue

if TYPE_CHECKING:
    from model_api.models.model import Model


class ParameterAccessor:
    """Provides attribute-style access to model parameters."""

    def __init__(self, model: "Model") -> None:
        self._model = model

    def __getattr__(self, name: str) -> Any:
        """Get parameter value by attribute name."""
        try:
            return self._model.get_param(name)
        except Exception as e:
            msg = f"Parameter '{name}' not found"
            raise AttributeError(msg) from e

    def __dir__(self) -> list[str]:
        """Return available parameter names"""
        try:
            return list(self._model.parameters().keys())
        except (AttributeError, TypeError, ValueError):
            return []


class ParameterRegistry:
    """Registry for common parameter groups used across models.

    This centralizes parameter definitions to reduce duplication and ensure
    consistency across model classes.
    """

    # Confidence threshold for filtering predictions
    CONFIDENCE_THRESHOLD: ClassVar[dict[str, Any]] = {
        "confidence_threshold": NumericalValue(
            default_value=0.5,
            description="Probability threshold value for filtering",
        ),
    }

    # Label-related parameters
    LABELS: ClassVar[dict[str, Any]] = {
        "labels": ListValue(
            description="List of class labels",
            value_type=str,
        ),
        "path_to_labels": StringValue(
            description="Path to file with labels. Overrides the labels parameter if both are provided",
        ),
    }

    # Image preprocessing parameters
    IMAGE_PREPROCESSING: ClassVar[dict[str, Any]] = {
        "embedded_processing": BooleanValue(
            description="Flag that pre/postprocessing is embedded in the model",
            default_value=False,
        ),
        "mean_values": ListValue(
            description="Normalization values to subtract from image channels during preprocessing",
            default_value=[],
        ),
        "scale_values": ListValue(
            description="Scale values to divide image channels during preprocessing",
            default_value=[],
        ),
        "reverse_input_channels": BooleanValue(
            default_value=False,
            description="Reverse the input channel order (e.g., RGB to BGR)",
        ),
    }

    # Image resizing parameters
    IMAGE_RESIZE: ClassVar[dict[str, Any]] = {
        "resize_type": StringValue(
            default_value="standard",
            choices=tuple(RESIZE_TYPES.keys()),
            description="Type of input image resizing",
        ),
        "pad_value": NumericalValue(
            int,
            min=0,
            max=255,
            description="Pad value for resize_image_letterbox embedded into a model",
            default_value=0,
        ),
        "orig_height": NumericalValue(
            int,
            description="Model input height before embedding processing",
            default_value=None,
        ),
        "orig_width": NumericalValue(
            int,
            description="Model input width before embedding processing",
            default_value=None,
        ),
    }

    # Top-k classification parameters
    TOP_K: ClassVar[dict[str, Any]] = {
        "topk": NumericalValue(
            value_type=int,
            default_value=1,
            min=1,
            description="Number of most likely labels to return",
        ),
    }

    # Multi-label classification parameters
    MULTILABEL: ClassVar[dict[str, Any]] = {
        "multilabel": BooleanValue(
            default_value=False,
            description="Predict a set of labels per image",
        ),
    }

    # Hierarchical classification parameters
    HIERARCHICAL: ClassVar[dict[str, Any]] = {
        "hierarchical": BooleanValue(
            default_value=False,
            description="Predict a hierarchy of labels per image",
        ),
        "hierarchical_config": StringValue(
            default_value="",
            description="Extra config for decoding hierarchical predictions",
        ),
        "hierarchical_postproc": StringValue(
            default_value="greedy",
            choices=("probabilistic", "greedy"),
            description="Type of hierarchical postprocessing",
        ),
    }

    # Output control parameters
    OUTPUT_RAW_SCORES: ClassVar[dict[str, Any]] = {
        "output_raw_scores": BooleanValue(
            default_value=False,
            description="Output all scores for multiclass classification",
        ),
    }

    # Segmentation parameters
    SEGMENTATION_POSTPROCESS: ClassVar[dict[str, Any]] = {
        "blur_strength": NumericalValue(
            value_type=int,
            description="Blurring kernel size. -1 value means no blurring and no soft_threshold",
            default_value=-1,
        ),
        "soft_threshold": NumericalValue(
            value_type=float,
            description="Probability threshold for pixel filtering. -inf means no thresholding",
            default_value=float("-inf"),
        ),
        "return_soft_prediction": BooleanValue(
            description="Return raw resized model prediction in addition to processed one",
            default_value=True,
        ),
    }

    # Instance segmentation parameters
    INSTANCE_SEGMENTATION: ClassVar[dict[str, Any]] = {
        "postprocess_semantic_masks": BooleanValue(
            description="Resize and apply 0.5 threshold to instance segmentation masks",
            default_value=True,
        ),
    }

    # NMS parameters
    NMS: ClassVar[dict[str, Any]] = {
        "iou_threshold": NumericalValue(
            default_value=0.5,
            description="Threshold for non-maximum suppression (NMS) intersection over union (IOU) filtering",
        ),
    }

    # Anomaly detection parameters
    ANOMALY: ClassVar[dict[str, Any]] = {
        "image_threshold": NumericalValue(
            description="Image-level anomaly threshold",
            default_value=0.5,
        ),
        "pixel_threshold": NumericalValue(
            description="Pixel-level anomaly threshold",
            default_value=0.5,
        ),
        "normalization_scale": NumericalValue(
            description="Scale factor for normalization",
            default_value=1.0,
        ),
        "task": StringValue(
            description="Task type: classification, segmentation, or detection",
            default_value="classification",
            choices=("classification", "segmentation", "detection"),
        ),
    }

    # Tiler base parameters
    TILER: ClassVar[dict[str, Any]] = {
        "tile_size": NumericalValue(
            value_type=int,
            default_value=400,
            min=1,
            description="Size of one tile",
        ),
        "tiles_overlap": NumericalValue(
            value_type=float,
            default_value=0.5,
            min=0.0,
            max=1.0,
            description="Overlap of tiles",
        ),
        "tile_with_full_img": BooleanValue(
            default_value=True,
            description="Whether to include full image as a tile",
        ),
    }

    # Detection tiler parameters
    DETECTION_TILER: ClassVar[dict[str, Any]] = {
        "max_pred_number": NumericalValue(
            value_type=int,
            default_value=100,
            min=1,
            description="Maximum numbers of prediction per image",
        ),
        "iou_threshold": NumericalValue(
            value_type=float,
            default_value=0.45,
            min=0,
            max=1.0,
            description="IoU threshold which is used to apply NMS to bounding boxes",
        ),
    }

    SOFTMAX: ClassVar[dict[str, Any]] = {
        "apply_softmax": BooleanValue(
            default_value=True,
            description="Whether to apply softmax on the heatmap.",
        ),
    }

    @staticmethod
    def merge(*param_groups: dict[str, Any]) -> dict[str, Any]:
        """Merge multiple parameter groups into a single dictionary."""
        result = {}
        for group in param_groups:
            result.update(group)
        return result
