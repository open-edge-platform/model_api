#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Export adapters for different model types."""

from model_converter.adapters.base import ExportAdapter
from model_converter.adapters.maskrcnn import TorchvisionMaskRCNNExportAdapter

_ADAPTER_REGISTRY: dict[str, type[ExportAdapter]] = {
    "maskrcnn": TorchvisionMaskRCNNExportAdapter,
}


def get_adapter(model_type: str, model: "torch.nn.Module") -> "torch.nn.Module":
    """
    Get the appropriate export adapter for a model type.

    If no adapter is registered for the model type, returns the model unchanged.

    Args:
        model_type: Model type string (e.g., "MaskRCNN")
        model: The PyTorch model to adapt

    Returns:
        Adapted model (or original model if no adapter needed)
    """
    import torch.nn as nn

    adapter_class = _ADAPTER_REGISTRY.get(model_type.lower())
    if adapter_class is not None:
        return adapter_class(model)
    return model


__all__ = ["ExportAdapter", "TorchvisionMaskRCNNExportAdapter", "get_adapter"]
