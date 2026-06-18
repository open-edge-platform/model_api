#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for model_converter.adapters module."""

from unittest.mock import MagicMock

import torch
from model_converter.adapters import TorchvisionMaskRCNNExportAdapter, get_adapter
from model_converter.adapters.base import ExportAdapter as BaseExportAdapter


class TestGetAdapter:
    """Tests for get_adapter registry function."""

    def test_known_type_maskrcnn(self):
        """get_adapter returns MaskRCNN adapter for 'maskrcnn' type."""
        mock_model = MagicMock()
        result = get_adapter("maskrcnn", mock_model)
        assert isinstance(result, TorchvisionMaskRCNNExportAdapter)

    def test_known_type_case_insensitive(self):
        """get_adapter handles case-insensitive lookup."""
        mock_model = MagicMock()
        result = get_adapter("MaskRCNN", mock_model)
        assert isinstance(result, TorchvisionMaskRCNNExportAdapter)

    def test_unknown_type_returns_model(self):
        """get_adapter returns model unchanged for unknown types."""
        mock_model = MagicMock()
        result = get_adapter("unknown_model_type", mock_model)
        assert result is mock_model

    def test_empty_type_returns_model(self):
        """get_adapter returns model unchanged for empty string."""
        mock_model = MagicMock()
        result = get_adapter("", mock_model)
        assert result is mock_model


class TestExportAdapter:
    """Tests for ExportAdapter base class."""

    def test_init_stores_model(self):
        """ExportAdapter stores the model as an attribute."""
        mock_model = MagicMock(spec=torch.nn.Module)
        adapter = BaseExportAdapter(mock_model)
        assert adapter.model is mock_model

    def test_is_nn_module(self):
        """ExportAdapter is a subclass of nn.Module."""
        assert issubclass(BaseExportAdapter, torch.nn.Module)


class TestTorchvisionMaskRCNNExportAdapter:
    """Tests for TorchvisionMaskRCNNExportAdapter."""

    def test_forward(self):
        """Test forward pass transforms MaskRCNN output correctly."""
        # Create a mock MaskRCNN model
        mock_model = MagicMock()

        # Mock transform
        mock_image_list = MagicMock()
        mock_image_list.tensors = torch.randn(1, 3, 224, 224)
        mock_image_list.image_sizes = [(224, 224)]
        mock_model.transform.return_value = (mock_image_list, None)

        # Mock backbone
        mock_features = {"0": torch.randn(1, 256, 56, 56)}
        mock_model.backbone.return_value = mock_features

        # Mock RPN
        mock_proposals = [torch.randn(100, 4)]
        mock_model.rpn.return_value = (mock_proposals, None)

        # Mock ROI heads
        mock_predictions = [
            {
                "boxes": torch.randn(10, 4),
                "scores": torch.rand(10),
                "labels": torch.randint(1, 80, (10,)),
                "masks": torch.rand(10, 1, 28, 28),
            },
        ]
        mock_model.roi_heads.return_value = (mock_predictions, None)

        # Create adapter and run forward
        adapter = TorchvisionMaskRCNNExportAdapter(mock_model)
        images = torch.randn(1, 3, 224, 224)
        boxes, labels, masks = adapter.forward(images)

        # Verify outputs
        assert boxes.shape == (10, 5)  # boxes (4) + scores (1)
        assert labels.shape == (10,)
        assert masks.shape == (10, 28, 28)  # squeezed from (10, 1, 28, 28)

        # Labels should be remapped from COCO 91-class IDs to 80-class indices
        from model_converter.metrics.coco_detection import COCO91_TO_COCO80

        expected_labels = torch.tensor(
            [
                COCO91_TO_COCO80[int(cat_id)] if COCO91_TO_COCO80[int(cat_id)] is not None else 0
                for cat_id in mock_predictions[0]["labels"]
            ],
            dtype=torch.int64,
        )
        assert torch.equal(labels, expected_labels)

    def test_forward_with_tensor_features(self):
        """Test forward when backbone returns a tensor instead of OrderedDict."""
        mock_model = MagicMock()

        mock_image_list = MagicMock()
        mock_image_list.tensors = torch.randn(1, 3, 224, 224)
        mock_image_list.image_sizes = [(224, 224)]
        mock_model.transform.return_value = (mock_image_list, None)

        # Return a tensor instead of dict
        mock_model.backbone.return_value = torch.randn(1, 256, 56, 56)

        mock_proposals = [torch.randn(50, 4)]
        mock_model.rpn.return_value = (mock_proposals, None)

        mock_predictions = [
            {
                "boxes": torch.randn(5, 4),
                "scores": torch.rand(5),
                "labels": torch.randint(1, 80, (5,)),
                "masks": torch.rand(5, 1, 28, 28),
            },
        ]
        mock_model.roi_heads.return_value = (mock_predictions, None)

        adapter = TorchvisionMaskRCNNExportAdapter(mock_model)
        images = torch.randn(1, 3, 224, 224)
        boxes, labels, masks = adapter.forward(images)

        # When backbone returns tensor, it should be wrapped in OrderedDict
        # Verify rpn was called with correct features format
        rpn_call_features = mock_model.rpn.call_args[0][1]
        assert isinstance(rpn_call_features, dict)
        assert "0" in rpn_call_features

        assert boxes.shape == (5, 5)
        assert labels.shape == (5,)
        assert masks.shape == (5, 28, 28)
