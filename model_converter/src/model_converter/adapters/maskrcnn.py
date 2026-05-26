#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Mask R-CNN export adapter for TorchVision models."""

from collections import OrderedDict

import torch

from model_converter.adapters.base import ExportAdapter


class TorchvisionMaskRCNNExportAdapter(ExportAdapter):
    """Adapt TorchVision Mask R-CNN to the Model API MaskRCNN output contract."""

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return boxes-with-scores, shifted labels, and raw masks for one image."""
        image_list = [images[0]]
        transformed_images, _ = self.model.transform(image_list, None)
        features = self.model.backbone(transformed_images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, _ = self.model.rpn(transformed_images, features, None)
        predictions, _ = self.model.roi_heads(features, proposals, transformed_images.image_sizes, None)
        prediction = predictions[0]
        boxes = torch.cat((prediction["boxes"], prediction["scores"].unsqueeze(1)), dim=1)
        labels = prediction["labels"] - 1
        masks = prediction["masks"].squeeze(1)
        return boxes, labels, masks
