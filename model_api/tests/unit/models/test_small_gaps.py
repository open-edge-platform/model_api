#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Small gap coverage tests for utils, hf_hub_helper, and onnx_adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# add_rotated_rects: angle > 90 and angle <= 0 branches
# ---------------------------------------------------------------------------

from model_api.models.result import InstanceSegmentationResult
from model_api.models.utils import OutputTransform, add_rotated_rects


class TestAddRotatedRects:
    def _make_inst_seg_result(self, masks):
        n = len(masks)
        return InstanceSegmentationResult(
            bboxes=np.zeros((n, 4), dtype=np.float32),
            labels=np.zeros(n, dtype=np.int32),
            masks=masks,
            scores=np.ones(n, dtype=np.float32),
            label_names=["obj"] * n,
        )

    def test_rotated_rect_angle_adjustment(self):
        # Create a tilted rectangle mask that will produce angle needing adjustment
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Draw a tilted rectangle
        pts = np.array([[50, 30], [170, 50], [160, 120], [40, 100]], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)

        result = add_rotated_rects(self._make_inst_seg_result([mask]))
        assert len(result.rotated_rects) == 1
        (cx, cy), (w, h), angle = result.rotated_rects[0]
        # After adjustment, angle must be in (0, 90]
        assert 0 < angle <= 90

    def test_empty_mask_returns_zeros(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = add_rotated_rects(self._make_inst_seg_result([mask]))
        assert result.rotated_rects[0] == ((0, 0), (0, 0), 0)


# ---------------------------------------------------------------------------
# OutputTransform.resize with scale_factor == 1
# ---------------------------------------------------------------------------


class TestOutputTransformResize:
    def test_scale_factor_one_returns_image(self):
        # When output_resolution matches input_size, scale_factor == 1
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        transform = OutputTransform(input_size=(100, 200), output_resolution=(200, 100))
        # scale_factor = min(200/200, 100/100) = 1.0
        result = transform.resize(img)
        np.testing.assert_array_equal(result, img)

    def test_no_output_resolution_returns_image(self):
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        transform = OutputTransform(input_size=(100, 200), output_resolution=None)
        result = transform.resize(img)
        np.testing.assert_array_equal(result, img)


# ---------------------------------------------------------------------------
# hf_hub_helper: .bin download failure & ImportError
# ---------------------------------------------------------------------------


class TestDownloadFromHf:
    def test_bin_download_failure_logs_debug(self):
        """When .xml download succeeds but .bin fails with OSError, it should log debug and not raise."""
        from pathlib import Path
        from unittest.mock import call

        mock_hf_hub_download = MagicMock()
        mock_snapshot_download = MagicMock()

        # First call (for .xml) succeeds, second call (for .bin) raises OSError
        mock_hf_hub_download.side_effect = [
            str(Path("/fake/model.xml")),
            OSError("Not found"),
        ]

        with patch.dict(
            "sys.modules",
            {
                "huggingface_hub": MagicMock(
                    hf_hub_download=mock_hf_hub_download,
                    snapshot_download=mock_snapshot_download,
                ),
            },
        ):
            # Need to reimport to pick up the mock
            import importlib

            import model_api.utils.hf_hub_helper as hf_mod

            importlib.reload(hf_mod)

            result = hf_mod.download_from_hf(repo_id="test/repo", filename="model.xml")
            assert result == Path("/fake/model.xml")
            # .bin download was attempted
            assert mock_hf_hub_download.call_count == 2

    def test_import_error_when_huggingface_not_installed(self):
        """When huggingface_hub is not installed, should raise ImportError."""
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            import importlib

            import model_api.utils.hf_hub_helper as hf_mod

            importlib.reload(hf_mod)

            with pytest.raises(ImportError, match="huggingface_hub"):
                hf_mod.download_from_hf(repo_id="test/repo", filename="model.xml")


# ---------------------------------------------------------------------------
# ONNXRuntimeAdapter ImportError when onnx is absent
# ---------------------------------------------------------------------------


class TestONNXAdapterImportError:
    def test_raises_import_error_when_onnx_absent(self):
        import model_api.adapters.onnx_adapter as onnx_mod

        with patch.object(onnx_mod, "onnxrt_absent", True):
            with pytest.raises(ImportError, match="onnx"):
                onnx_mod.ONNXRuntimeAdapter("dummy.onnx")
