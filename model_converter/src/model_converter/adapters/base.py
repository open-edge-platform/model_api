#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Base export adapter interface."""

import torch.nn as nn


class ExportAdapter(nn.Module):
    """Base class for export adapters that reshape model outputs for Model API."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
