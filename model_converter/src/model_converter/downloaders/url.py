#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Direct URL downloader."""

import logging
import urllib.request
from pathlib import Path

from model_converter.downloaders.base import BaseDownloader

logger = logging.getLogger(__name__)


class URLDownloader(BaseDownloader):
    """Download model weights from direct URLs."""

    def download(
        self,
        url: str,
        filename: str | None = None,
    ) -> Path:
        """
        Download model weights from URL with caching.

        Args:
            url: URL to download weights from
            filename: Optional filename to save as (default: extract from URL)

        Returns:
            Path to the downloaded/cached weights file
        """
        if filename is None:
            filename = url.split("/")[-1]

        cached_file = self.cache_dir / filename

        if cached_file.exists():
            logger.info(f"Using cached weights: {cached_file}")
            return cached_file

        logger.info(f"Downloading weights from: {url}")
        logger.info(f"Saving to: {cached_file}")

        try:
            urllib.request.urlretrieve(  # noqa: S310  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
                url,
                cached_file,
            )
            logger.info("✓ Download complete")
            return cached_file
        except Exception as e:
            logger.error(f"Failed to download weights: {e}")
            raise
