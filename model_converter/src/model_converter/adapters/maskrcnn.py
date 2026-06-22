#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Mask R-CNN export adapter for TorchVision models."""

from collections import OrderedDict

import torch

from model_converter.adapters.base import ExportAdapter
from model_converter.metrics.coco_detection import COCO91_TO_COCO80

# Lookup tensor mapping a torchvision COCO 91-class category ID to the value the
# exported model must emit so that, after the Model API ``MaskRCNN`` wrapper adds
# ``+1`` to every label, the reported label equals the contiguous 80-class index
# (0-79) expected by the labels metadata and the COCO evaluator. We therefore
# store ``COCO80_index - 1`` here. Category IDs that COCO dropped (and the
# background ID 0) are unreachable for real predictions; they map to ``-1`` purely
# to keep the lookup well-defined.
_COCO91_TO_COCO80_LUT = torch.tensor(
    [idx - 1 if idx is not None else -1 for idx in COCO91_TO_COCO80],
    dtype=torch.int64,
)


class TorchvisionMaskRCNNExportAdapter(ExportAdapter):
    """Adapt TorchVision Mask R-CNN to the Model API MaskRCNN output contract."""

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return boxes-with-scores, wrapper-compensated labels, and raw masks for one image."""
        image_list = [images[0]]
        transformed_images, _ = self.model.transform(image_list, None)
        features = self.model.backbone(transformed_images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, _ = self.model.rpn(transformed_images, features, None)
        predictions, _ = self.model.roi_heads(features, proposals, transformed_images.image_sizes, None)
        prediction = predictions[0]
        boxes = torch.cat((prediction["boxes"], prediction["scores"].unsqueeze(1)), dim=1)
        lut = _COCO91_TO_COCO80_LUT.to(prediction["labels"].device)
        labels = lut[prediction["labels"]]
        masks = prediction["masks"].squeeze(1)
        return boxes, labels, masks
