#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for model_converter.downloaders module."""

from pathlib import Path
from unittest.mock import patch

import pytest
from model_converter.downloaders import HuggingFaceDownloader, URLDownloader
from model_converter.downloaders.base import BaseDownloader


class TestBaseDownloader:
    """Tests for BaseDownloader."""

    def test_creates_cache_dir(self, tmp_path):
        """BaseDownloader creates cache directory on init."""
        cache_dir = tmp_path / "new_cache" / "nested"
        downloader = BaseDownloader(cache_dir=cache_dir)
        assert downloader.cache_dir.exists()
        assert downloader.cache_dir == cache_dir

    def test_existing_cache_dir(self, tmp_path):
        """BaseDownloader handles existing cache directory."""
        cache_dir = tmp_path / "existing_cache"
        cache_dir.mkdir()
        downloader = BaseDownloader(cache_dir=cache_dir)
        assert downloader.cache_dir.exists()

    def test_cache_dir_is_path(self, tmp_path):
        """BaseDownloader converts cache_dir to Path."""
        cache_dir = tmp_path / "cache"
        downloader = BaseDownloader(cache_dir=cache_dir)
        assert isinstance(downloader.cache_dir, Path)


class TestURLDownloader:
    """Tests for URLDownloader."""

    def test_cache_hit(self, tmp_path):
        """URLDownloader returns cached file if it exists."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached_file = cache_dir / "weights.pth"
        cached_file.write_text("dummy weights")

        downloader = URLDownloader(cache_dir=cache_dir)
        result = downloader.download("https://example.com/weights.pth")

        assert result == cached_file

    def test_download_success(self, tmp_path):
        """URLDownloader downloads file and returns path."""
        cache_dir = tmp_path / "cache"
        downloader = URLDownloader(cache_dir=cache_dir)

        with patch("urllib.request.urlretrieve") as mock_urlretrieve:
            mock_urlretrieve.side_effect = lambda url, path: Path(path).write_text("data")
            result = downloader.download("https://example.com/model_weights.pth")

        assert result == cache_dir / "model_weights.pth"
        mock_urlretrieve.assert_called_once_with(
            "https://example.com/model_weights.pth",
            cache_dir / "model_weights.pth",
        )

    def test_download_with_custom_filename(self, tmp_path):
        """URLDownloader uses custom filename when provided."""
        cache_dir = tmp_path / "cache"
        downloader = URLDownloader(cache_dir=cache_dir)

        with patch("urllib.request.urlretrieve") as mock_urlretrieve:
            mock_urlretrieve.side_effect = lambda url, path: Path(path).write_text("data")
            result = downloader.download("https://example.com/v1/download", filename="custom.pth")

        assert result == cache_dir / "custom.pth"

    def test_download_failure(self, tmp_path):
        """URLDownloader raises on download failure."""
        cache_dir = tmp_path / "cache"
        downloader = URLDownloader(cache_dir=cache_dir)

        with patch("urllib.request.urlretrieve") as mock_urlretrieve:
            mock_urlretrieve.side_effect = OSError("Connection refused")
            with pytest.raises(OSError, match="Connection refused"):
                downloader.download("https://example.com/weights.pth")

    def test_filename_extracted_from_url(self, tmp_path):
        """URLDownloader extracts filename from URL when not provided."""
        cache_dir = tmp_path / "cache"
        downloader = URLDownloader(cache_dir=cache_dir)

        with patch("urllib.request.urlretrieve") as mock_urlretrieve:
            mock_urlretrieve.side_effect = lambda url, path: Path(path).write_text("data")
            result = downloader.download("https://example.com/path/to/resnet50.pth")

        assert result.name == "resnet50.pth"


class TestHuggingFaceDownloader:
    """Tests for HuggingFaceDownloader."""

    def test_download_single_file(self, tmp_path):
        """HuggingFaceDownloader downloads a single file."""
        cache_dir = tmp_path / "cache"
        downloader = HuggingFaceDownloader(cache_dir=cache_dir)

        with patch("model_converter.downloaders.huggingface.hf_hub_download") as mock_hf:
            mock_hf.return_value = str(cache_dir / "model.safetensors")
            result = downloader.download(
                repo_id="timm/resnet50",
                revision="abc123",
                filename="model.safetensors",
            )

        assert result == cache_dir / "model.safetensors"
        mock_hf.assert_called_once_with(
            repo_id="timm/resnet50",
            revision="abc123",
            filename="model.safetensors",
            cache_dir=cache_dir,
        )

    def test_download_snapshot(self, tmp_path):
        """HuggingFaceDownloader downloads full repository."""
        cache_dir = tmp_path / "cache"
        downloader = HuggingFaceDownloader(cache_dir=cache_dir)

        with patch("model_converter.downloaders.huggingface.snapshot_download") as mock_snap:
            mock_snap.return_value = str(cache_dir / "repo_snapshot")
            result = downloader.download(
                repo_id="timm/resnet50",
                revision="abc123",
            )

        assert result == cache_dir / "repo_snapshot"
        mock_snap.assert_called_once_with(
            repo_id="timm/resnet50",
            revision="abc123",
            cache_dir=cache_dir,
        )

    def test_download_failure(self, tmp_path):
        """HuggingFaceDownloader raises on download failure."""
        cache_dir = tmp_path / "cache"
        downloader = HuggingFaceDownloader(cache_dir=cache_dir)

        with patch("model_converter.downloaders.huggingface.hf_hub_download") as mock_hf:
            mock_hf.side_effect = Exception("Repository not found")
            with pytest.raises(Exception, match="Repository not found"):
                downloader.download(
                    repo_id="nonexistent/model",
                    revision="abc123",
                    filename="weights.bin",
                )

    def test_download_snapshot_failure(self, tmp_path):
        """HuggingFaceDownloader raises on snapshot download failure."""
        cache_dir = tmp_path / "cache"
        downloader = HuggingFaceDownloader(cache_dir=cache_dir)

        with patch("model_converter.downloaders.huggingface.snapshot_download") as mock_snap:
            mock_snap.side_effect = Exception("Network error")
            with pytest.raises(Exception, match="Network error"):
                downloader.download(
                    repo_id="timm/resnet50",
                    revision="abc123",
                )
