#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Model weight and file downloaders."""

from model_converter.downloaders.huggingface import HuggingFaceDownloader
from model_converter.downloaders.url import URLDownloader

__all__ = ["HuggingFaceDownloader", "URLDownloader"]
