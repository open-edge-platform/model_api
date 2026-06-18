#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Mask R-CNN export adapter for TorchVision models."""

from collections import OrderedDict

import torch

from model_converter.adapters.base import ExportAdapter
from model_converter.metrics.coco_detection import COCO91_TO_COCO80

# Lookup tensor mapping a torchvision COCO 91-class category ID to the contiguous
# 80-class index (0-79) expected by the Model API / COCO evaluator. Category IDs
# that COCO dropped (and background) are unreachable for predictions; they map to
# 0 here purely to keep the lookup well-defined.
_COCO91_TO_COCO80_LUT = torch.tensor(
    [idx if idx is not None else 0 for idx in COCO91_TO_COCO80],
    dtype=torch.int64,
)


class TorchvisionMaskRCNNExportAdapter(ExportAdapter):
    """Adapt TorchVision Mask R-CNN to the Model API MaskRCNN output contract."""

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return boxes-with-scores, 80-class labels, and raw masks for one image."""
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
