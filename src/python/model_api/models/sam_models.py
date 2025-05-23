#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations  # TODO: remove when Python3.9 support is dropped

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from model_api.adapters.inference_adapter import InferenceAdapter
from model_api.models.types import BooleanValue, NumericalValue

from .image_model import ImageModel
from .segmentation import SegmentationModel

if TYPE_CHECKING:
    from model_api.adapters.inference_adapter import InferenceAdapter


class SAMImageEncoder(ImageModel):
    """Image Encoder for SAM: https://arxiv.org/abs/2304.02643"""

    __model__ = "sam_image_encoder"

    def __init__(
        self,
        inference_adapter: InferenceAdapter,
        configuration: dict[str, Any] = {},
        preload: bool = False,
    ):
        super().__init__(inference_adapter, configuration, preload)
        self.output_name: str = next(iter(self.outputs.keys()))
        self.resize_type: str
        self.image_size: int

    @classmethod
    def parameters(cls) -> dict[str, Any]:
        parameters = super().parameters()
        parameters.update(
            {
                "image_size": NumericalValue(
                    value_type=int,
                    default_value=1024,
                    min=0,
                    max=2048,
                ),
            },
        )
        return parameters

    def preprocess(
        self,
        inputs: np.ndarray,
    ) -> list[dict]:
        """Update meta for image encoder."""
        dict_inputs, meta = super().preprocess(inputs)
        meta["resize_type"] = self.resize_type
        return [dict_inputs, meta]

    def postprocess(
        self,
        outputs: dict[str, np.ndarray],
        meta: dict[str, Any],
    ) -> np.ndarray:
        return outputs[self.output_name]


class SAMDecoder(SegmentationModel):
    """Image Decoder for SAM: https://arxiv.org/abs/2304.02643"""

    __model__ = "sam_decoder"

    def __init__(
        self,
        model_adapter: InferenceAdapter,
        configuration: dict[str, Any] = {},
        preload: bool = False,
    ):
        super().__init__(model_adapter, configuration, preload)

        self.mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        self.has_mask_input = np.zeros((1, 1), dtype=np.float32)
        self.image_size: int
        self.mask_threshold: float
        self.embed_dim: int

    @classmethod
    def parameters(cls) -> dict[str, Any]:
        parameters = super().parameters()
        parameters.update(
            {
                "image_size": NumericalValue(
                    value_type=int,
                    default_value=1024,
                    min=0,
                    max=2048,
                ),
            },
        )
        parameters.update(
            {
                "mask_threshold": NumericalValue(
                    value_type=float,
                    default_value=0.0,
                    min=0,
                    max=1,
                ),
            },
        )
        parameters.update(
            {
                "embed_dim": NumericalValue(
                    value_type=int,
                    default_value=256,
                    min=0,
                    max=512,
                ),
            },
        )
        parameters.update({"embedded_processing": BooleanValue(default_value=True)})
        return parameters

    def _get_outputs(self) -> str:
        return "upscaled_masks"

    def preprocess(self, inputs: dict[str, Any]) -> list[dict]:
        """Preprocess prompts."""
        processed_prompts: list[dict[str, Any]] = []
        for prompt_name in ["bboxes", "points"]:
            if (prompts := inputs.get(prompt_name)) is None or (
                labels := inputs["labels"].get(prompt_name, None)
            ) is None:
                continue

            for prompt, label in zip(prompts, labels):
                if prompt_name == "bboxes":
                    point_coords = self.apply_coords(
                        prompt.reshape(-1, 2, 2),
                        inputs["orig_size"],
                    )
                    point_labels = np.array([2, 3], dtype=np.float32).reshape(-1, 2)
                else:
                    point_coords = self.apply_coords(
                        prompt.reshape(-1, 1, 2),
                        inputs["orig_size"],
                    )
                    point_labels = np.array([1], dtype=np.float32).reshape(-1, 1)

                processed_prompts.append(
                    {
                        "point_coords": point_coords,
                        "point_labels": point_labels,
                        "mask_input": self.mask_input,
                        "has_mask_input": self.has_mask_input,
                        "orig_size": np.array(
                            inputs["orig_size"],
                            dtype=np.int64,
                        ).reshape(-1, 2),
                        "label": label,
                    },
                )
        return processed_prompts

    def apply_coords(
        self,
        coords: np.ndarray,
        orig_size: np.ndarray | list[int] | tuple[int, int],
    ) -> np.ndarray:
        """Process coords according to preprocessed image size using image meta."""
        old_h, old_w = orig_size
        new_h, new_w = self._get_preprocess_shape(old_h, old_w, self.image_size)
        coords = deepcopy(coords).astype(np.float32)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def _get_preprocess_shape(
        self,
        old_h: int,
        old_w: int,
        image_size: int,
    ) -> tuple[int, int]:
        """Compute the output size given input size and target image size."""
        scale = image_size / max(old_h, old_w)
        new_h, new_w = old_h * scale, old_w * scale
        new_w = int(new_w + 0.5)
        new_h = int(new_h + 0.5)
        return (new_h, new_w)

    def _check_io_number(
        self,
        number_of_inputs: int | tuple[int, ...],
        number_of_outputs: int | tuple[int, ...],
    ) -> None:
        pass

    def _get_inputs(self) -> tuple[list[str], list[str]]:
        """Get input layer name and shape."""
        image_blob_names = list(self.inputs.keys())
        image_info_blob_names: list = []
        return image_blob_names, image_info_blob_names

    def postprocess(
        self,
        outputs: dict[str, np.ndarray],
        meta: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Postprocess to convert soft prediction to hard prediction.

        Args:
            outputs (dict[str, np.ndarray]): The output of the model.
            meta (dict[str, Any]): Contain label and original size.

        Returns:
            (dict[str, np.ndarray]): The postprocessed output of the model.
        """
        outputs = deepcopy(outputs)
        probability = np.clip(outputs["scores"], 0.0, 1.0)
        hard_prediction = outputs[self.output_blob_name].squeeze(0) > self.mask_threshold
        soft_prediction = hard_prediction * probability.reshape(-1, 1, 1)

        outputs["hard_prediction"] = hard_prediction
        outputs["soft_prediction"] = soft_prediction

        return outputs
