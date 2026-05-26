#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Hugging Face Hub downloader."""

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from model_converter.downloaders.base import BaseDownloader

logger = logging.getLogger(__name__)


class HuggingFaceDownloader(BaseDownloader):
    """Download models and files from Hugging Face Hub."""

    def download(
        self,
        repo_id: str,
        revision: str,
        filename: str | None = None,
    ) -> Path:
        """
        Download model from Hugging Face Hub with caching.

        Args:
            repo_id: Hugging Face repository ID (e.g., 'timm/mobilenetv2_050.lamb_in1k')
            revision: Immutable revision/commit SHA to download from
            filename: Optional specific file to download (if None, downloads the whole repo)

        Returns:
            Path to the downloaded model file or directory
        """
        logger.info(f"Downloading from Hugging Face Hub: {repo_id}")

        try:
            if filename:
                cached_file = hf_hub_download(  # nosec B615
                    repo_id=repo_id,
                    revision=revision,
                    filename=filename,
                    cache_dir=self.cache_dir,
                )
                logger.info(f"✓ Downloaded file: {cached_file}")
                return Path(cached_file)
            cached_dir = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=self.cache_dir,
            )
            logger.info(f"✓ Downloaded repository to: {cached_dir}")
            return Path(cached_dir)
        except Exception as e:
            logger.error(f"Failed to download from Hugging Face: {e}")
            raise
