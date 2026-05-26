#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Base downloader interface."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseDownloader:
    """Base class for model weight downloaders."""

    def __init__(self, cache_dir: Path):
        """
        Initialize downloader.

        Args:
            cache_dir: Directory to cache downloaded files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
