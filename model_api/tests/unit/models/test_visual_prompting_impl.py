#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for visual_prompting helpers and wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from model_api.models.visual_prompting import (
    Prompt,
    SAMLearnableVisualPrompter,
    SAMVisualPrompter,
    VisualPromptingFeatures,
    _decide_masks,
    _generate_masked_features,
    _get_prepadded_size,
    _get_prompt_candidates,
    _inspect_overlapping_areas,
    _pad_to_square,
    _point_selection,
    _polygon_to_mask,
    _resize_to_original_shape,
    _topk_numpy,
)


# ---------------------------------------------------------------------------
# _polygon_to_mask
# ---------------------------------------------------------------------------


class TestPolygonToMask:
    def test_integer_array(self):
        polygon = np.array([[10, 10], [10, 50], [50, 50], [50, 10]], dtype=np.int32)
        mask = _polygon_to_mask(polygon, 100, 100)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        assert mask.sum() > 0

    def test_list_of_points(self):
        polygon = [[5.0, 5.0], [5.0, 20.0], [20.0, 20.0], [20.0, 5.0]]
        mask = _polygon_to_mask(polygon, 50, 50)
        assert mask.shape == (50, 50)
        assert mask.sum() > 0


# ---------------------------------------------------------------------------
# _generate_masked_features
# ---------------------------------------------------------------------------


class TestGenerateMaskedFeatures:
    def test_valid_mask_returns_features(self):
        feats = np.random.rand(64, 64, 256).astype(np.float32)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        result = _generate_masked_features(feats, mask, threshold_mask=0.3, image_size=1024)
        assert result is not None
        assert result.shape == (1, 256)
        # Should be normalized
        norm = np.linalg.norm(result, axis=-1)
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    def test_empty_mask_returns_none(self):
        feats = np.random.rand(64, 64, 256).astype(np.float32)
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = _generate_masked_features(feats, mask, threshold_mask=0.5, image_size=1024)
        assert result is None


# ---------------------------------------------------------------------------
# _pad_to_square
# ---------------------------------------------------------------------------


class TestPadToSquare:
    def test_padding(self):
        x = np.ones((500, 700), dtype=np.float32)
        result = _pad_to_square(x, image_size=1024)
        assert result.shape == (1024, 1024)
        # original area preserved
        assert result[:500, :700].sum() == 500 * 700
        # padded area is zero
        assert result[500:, :].sum() == 0
        assert result[:, 700:].sum() == 0


# ---------------------------------------------------------------------------
# _decide_masks
# ---------------------------------------------------------------------------


class TestDecideMasks:
    def test_is_single_uses_idx_0(self):
        masks = np.ones((1, 4, 32, 32), dtype=np.float32)
        logits = np.random.rand(1, 4, 16, 16).astype(np.float32)
        scores = np.array([[0.2, 0.9, 0.5, 0.3]])
        log_out, mask_out, score = _decide_masks(masks, logits, scores, is_single=True)
        assert log_out is not None
        assert log_out.shape[1] == 1
        # is_single always uses best_idx=0
        np.testing.assert_array_equal(log_out, logits[:, [0]])

    def test_not_single_skips_first(self):
        masks = np.ones((1, 4, 32, 32), dtype=np.float32)
        logits = np.random.rand(1, 4, 16, 16).astype(np.float32)
        scores = np.array([[0.1, 0.9, 0.8, 0.5]])
        # After skipping index 0, remaining scores are [0.9, 0.8, 0.5], best is index 0 of remaining
        log_out, mask_out, score = _decide_masks(masks, logits, scores, is_single=False)
        assert score == pytest.approx(0.9, abs=1e-5)

    def test_all_zero_masks(self):
        masks = np.zeros((1, 4, 32, 32), dtype=np.float32)
        logits = np.random.rand(1, 4, 16, 16).astype(np.float32)
        scores = np.array([[0.1, 0.9, 0.8, 0.5]])
        log_out, mask_out, score = _decide_masks(masks, logits, scores, is_single=False)
        assert log_out is None
        assert score == 0.0
        assert mask_out.sum() == 0


# ---------------------------------------------------------------------------
# _topk_numpy
# ---------------------------------------------------------------------------


class TestTopkNumpy:
    def test_largest_true(self):
        x = np.array([1, 5, 3, 4, 2])
        vals, inds = _topk_numpy(x, k=3, largest=True)
        assert list(vals) == [5, 4, 3]

    def test_largest_false(self):
        x = np.array([1, 5, 3, 4, 2])
        vals, inds = _topk_numpy(x, k=2, largest=False)
        assert list(vals) == [1, 2]


# ---------------------------------------------------------------------------
# _get_prepadded_size
# ---------------------------------------------------------------------------


class TestGetPrepaddedSize:
    def test_landscape(self):
        original = np.array([500, 1000])
        result = _get_prepadded_size(original, 1024)
        # longer side scaled to 1024
        assert result[1] == 1024
        assert result[0] < 1024

    def test_square(self):
        original = np.array([800, 800])
        result = _get_prepadded_size(original, 1024)
        assert result[0] == result[1] == 1024


# ---------------------------------------------------------------------------
# _resize_to_original_shape
# ---------------------------------------------------------------------------


class TestResizeToOriginalShape:
    def test_output_shape(self):
        masks = np.random.rand(64, 64).astype(np.float32)
        original_shape = np.array([480, 640])
        result = _resize_to_original_shape(masks, 1024, original_shape)
        assert result.shape == (480, 640)


# ---------------------------------------------------------------------------
# _point_selection
# ---------------------------------------------------------------------------


class TestPointSelection:
    def test_with_points_above_threshold(self):
        sim = np.zeros((100, 100), dtype=np.float32)
        sim[40:60, 40:60] = 0.9
        original_shape = np.array([100, 100])
        pts, bg = _point_selection(sim, original_shape, threshold=0.5, num_bg_points=1, image_size=1024, downsizing=64)
        assert pts is not None
        assert bg is not None
        assert pts.shape[1] == 3  # x, y, score

    def test_empty_case(self):
        sim = np.zeros((100, 100), dtype=np.float32)
        original_shape = np.array([100, 100])
        pts, bg = _point_selection(sim, original_shape, threshold=0.5, num_bg_points=1, image_size=1024, downsizing=64)
        assert pts is None
        assert bg is None


# ---------------------------------------------------------------------------
# _inspect_overlapping_areas
# ---------------------------------------------------------------------------


class TestInspectOverlappingAreas:
    def test_no_overlap(self):
        mask_a = np.zeros((50, 50), dtype=np.float32)
        mask_a[:25, :] = 1
        mask_b = np.zeros((50, 50), dtype=np.float32)
        mask_b[25:, :] = 1
        predicted = {0: [mask_a], 1: [mask_b]}
        used_pts = {0: [np.array([10, 10, 0.9])], 1: [np.array([10, 30, 0.8])]}
        _inspect_overlapping_areas(predicted, used_pts)
        assert len(predicted[0]) == 1
        assert len(predicted[1]) == 1

    def test_high_overlap_removes_lower_score(self):
        mask = np.ones((50, 50), dtype=np.float32)
        predicted = {0: [mask.copy()], 1: [mask.copy()]}
        used_pts = {0: [np.array([10, 10, 0.9])], 1: [np.array([10, 10, 0.5])]}
        _inspect_overlapping_areas(predicted, used_pts, threshold_iou=0.8)
        # Lower-scored mask (label 1) should be removed
        assert len(predicted[1]) == 0
        assert len(predicted[0]) == 1

    def test_partial_overlap_zeros_intersection(self):
        mask_a = np.zeros((50, 50), dtype=np.float32)
        mask_a[:30, :] = 1
        mask_b = np.zeros((50, 50), dtype=np.float32)
        mask_b[20:, :] = 1
        predicted = {0: [mask_a.copy()], 1: [mask_b.copy()]}
        used_pts = {0: [np.array([10, 10, 0.9])], 1: [np.array([10, 30, 0.5])]}
        _inspect_overlapping_areas(predicted, used_pts, threshold_iou=0.95)
        # Partial overlap: intersection region zeroed in lower-scored mask
        assert predicted[1][0][20:30, :].sum() == 0


# ---------------------------------------------------------------------------
# _get_prompt_candidates
# ---------------------------------------------------------------------------


class TestGetPromptCandidates:
    def test_basic_flow(self):
        ref_feats = np.random.rand(2, 1, 256).astype(np.float32)
        ref_feats = ref_feats / np.linalg.norm(ref_feats, axis=-1, keepdims=True)
        image_emb = np.random.rand(1, 256, 64, 64).astype(np.float32)
        used_indices = np.array([0, 1])
        original_shape = np.array([480, 640])
        pts_scores, bg_coords = _get_prompt_candidates(
            image_embeddings=image_emb,
            reference_feats=ref_feats,
            used_indices=used_indices,
            original_shape=original_shape,
            threshold=0.0,
            num_bg_points=1,
            default_threshold_target=0.0,
            image_size=1024,
            downsizing=64,
        )
        assert isinstance(pts_scores, dict)
        assert isinstance(bg_coords, dict)


# ---------------------------------------------------------------------------
# SAMVisualPrompter
# ---------------------------------------------------------------------------


def _make_mock_encoder_decoder():
    encoder = MagicMock()
    encoder.params = MagicMock()
    encoder.params.image_size = 1024
    encoder.params.resize_type = "standard"

    decoder = MagicMock()
    decoder.params = MagicMock()
    decoder.params.embed_dim = 256
    decoder.params.mask_threshold = 0.0
    return encoder, decoder


class TestSAMVisualPrompter:
    def test_init(self):
        enc, dec = _make_mock_encoder_decoder()
        prompter = SAMVisualPrompter(enc, dec)
        assert prompter.encoder is enc
        assert prompter.decoder is dec

    def test_infer_no_prompts_raises(self):
        enc, dec = _make_mock_encoder_decoder()
        prompter = SAMVisualPrompter(enc, dec)
        with pytest.raises(RuntimeError, match="boxes or points prompts are required"):
            prompter.infer(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_call_delegates_to_infer(self):
        enc, dec = _make_mock_encoder_decoder()
        prompter = SAMVisualPrompter(enc, dec)
        with pytest.raises(RuntimeError, match="boxes or points prompts are required"):
            prompter(np.zeros((100, 100, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# SAMLearnableVisualPrompter
# ---------------------------------------------------------------------------


class TestSAMLearnableVisualPrompter:
    def test_init_default(self):
        enc, dec = _make_mock_encoder_decoder()
        prompter = SAMLearnableVisualPrompter(enc, dec)
        assert prompter._threshold == 0.65
        assert not prompter.has_reference_features()

    def test_init_with_reference_features(self):
        enc, dec = _make_mock_encoder_decoder()
        feats = VisualPromptingFeatures(
            feature_vectors=np.random.rand(3, 1, 256).astype(np.float32),
            used_indices=np.array([0, 1, 2]),
        )
        prompter = SAMLearnableVisualPrompter(enc, dec, reference_features=feats)
        assert prompter.has_reference_features()

    def test_init_invalid_threshold(self):
        enc, dec = _make_mock_encoder_decoder()
        with pytest.raises(ValueError, match="Confidence threshold"):
            SAMLearnableVisualPrompter(enc, dec, threshold=1.5)

    def test_reference_features_property_success(self):
        enc, dec = _make_mock_encoder_decoder()
        feats = VisualPromptingFeatures(
            feature_vectors=np.random.rand(2, 1, 256).astype(np.float32),
            used_indices=np.array([0, 1]),
        )
        prompter = SAMLearnableVisualPrompter(enc, dec, reference_features=feats)
        result = prompter.reference_features
        assert isinstance(result, VisualPromptingFeatures)

    def test_reference_features_property_error(self):
        enc, dec = _make_mock_encoder_decoder()
        prompter = SAMLearnableVisualPrompter(enc, dec)
        with pytest.raises(RuntimeError, match="Reference features are not generated"):
            _ = prompter.reference_features

    def test_reset_reference_info(self):
        enc, dec = _make_mock_encoder_decoder()
        prompter = SAMLearnableVisualPrompter(enc, dec)
        prompter.reset_reference_info()
        assert prompter._reference_features is not None
        assert prompter._reference_features.shape == (0, 1, 256)
        assert len(prompter._used_indices) == 0

    def test_gather_prompts_with_labels(self):
        enc, dec = _make_mock_encoder_decoder()
        prompter = SAMLearnableVisualPrompter(enc, dec)
        prompts = [
            {"label": np.int64(0), "data": np.array([1, 2])},
            {"label": np.int64(1), "data": np.array([3, 4])},
            {"label": np.int64(0), "data": np.array([5, 6])},
        ]
        result = prompter._gather_prompts_with_labels(prompts)
        assert 0 in result
        assert 1 in result
        assert len(result[0]) == 2
        assert len(result[1]) == 1

    def test_expand_reference_info(self):
        enc, dec = _make_mock_encoder_decoder()
        prompter = SAMLearnableVisualPrompter(enc, dec)
        prompter.reset_reference_info()
        prompter._expand_reference_info(5)
        assert prompter._reference_features.shape[0] == 6  # 0..5

    def test_learn_no_prompts_raises(self):
        enc, dec = _make_mock_encoder_decoder()
        prompter = SAMLearnableVisualPrompter(enc, dec)
        with pytest.raises(RuntimeError, match="boxes, polygons or points prompts are required"):
            prompter.learn(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_infer_without_reference_features_raises(self):
        enc, dec = _make_mock_encoder_decoder()
        prompter = SAMLearnableVisualPrompter(enc, dec)
        with pytest.raises(RuntimeError, match="Reference features are not defined"):
            prompter.infer(np.zeros((100, 100, 3), dtype=np.uint8))
