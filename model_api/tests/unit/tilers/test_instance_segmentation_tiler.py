# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from model_api.models import InstanceSegmentationResult
from model_api.models.instance_segmentation import (
    DETRInstanceSegmentation,
    InstanceSegmentationModel,
    MaskRCNNModel,
    _full_image_mask_postprocess,
    _segm_postprocess,
)
from model_api.tilers.instance_segmentation import InstanceSegmentationTiler


def _legacy_expand_box(box, scale):
    """Copy of `_expand_box`, so the reference below shares no production code."""
    w_half = (box[2] - box[0]) * 0.5 * scale
    h_half = (box[3] - box[1]) * 0.5 * scale
    x_c = (box[2] + box[0]) * 0.5
    y_c = (box[3] + box[1]) * 0.5
    box_exp = np.zeros(box.shape)
    box_exp[0] = x_c - w_half
    box_exp[2] = x_c + w_half
    box_exp[1] = y_c - h_half
    box_exp[3] = y_c + h_half
    return box_exp


def _legacy_segm_postprocess(box, raw_cls_mask, im_h, im_w):
    """Verbatim copy of the pre-crop `_segm_postprocess` (and `_expand_box`): the independent reference."""
    raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), "constant", constant_values=0)
    extended_box = _legacy_expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
    x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])
    mask = (cv2.resize(raw_cls_mask.astype(np.float32), (w, h)) > 0.5).astype(np.uint8)
    im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    im_mask[y0:y1, x0:x1] = mask[
        (y0 - extended_box[1]) : (y1 - extended_box[1]),
        (x0 - extended_box[0]) : (x1 - extended_box[0]),
    ]
    return im_mask


# Inside the image, clipped on every side, partly outside, and one pixel wide.
_BOXES = [[10, 20, 50, 60], [-15, -10, 30, 25], [80, 70, 130, 120], [-5, 60, 40, 140], [30, 30, 30, 30]]


def _make_model():
    model = MagicMock()
    model.load = MagicMock()
    model.inference_adapter = MagicMock()
    model.inference_adapter.get_rt_info.side_effect = RuntimeError(
        "Cannot get runtime attribute. Path to runtime attribute is incorrect.",
    )
    model.get_label_name = MagicMock(side_effect=lambda x: f"label_{x}")
    return model


def _make_instance_seg_result(n=2, offset=(0, 0)):
    bboxes = np.array([[10 + offset[0], 20 + offset[1], 50 + offset[0], 60 + offset[1]]] * n, dtype=np.float32)
    labels = np.array([0] * n, dtype=np.int32)
    scores = np.array([0.9] * n, dtype=np.float32)
    masks = [np.ones((40, 40), dtype=np.uint8) for _ in range(n)]
    return InstanceSegmentationResult(
        bboxes=bboxes,
        labels=labels,
        scores=scores,
        masks=np.array(masks),
        saliency_map=[np.zeros((10, 10), dtype=np.float32)],
        feature_vector=np.zeros((1,), dtype=np.float32),
    )


def _make_detr_model():
    model = MagicMock(spec=DETRInstanceSegmentation)
    model.inference_adapter = MagicMock()
    model.inference_adapter.get_rt_info.side_effect = RuntimeError(
        "Cannot get runtime attribute. Path to runtime attribute is incorrect.",
    )
    model.params = MagicMock()
    model.params.postprocess_semantic_masks = True
    model.get_label_name.return_value = "object"
    model._postprocess_single_mask.side_effect = lambda *args: DETRInstanceSegmentation._postprocess_single_mask(  # noqa: SLF001
        model,
        *args,
    )
    model._postprocess_tile_mask.side_effect = lambda *args: DETRInstanceSegmentation._postprocess_tile_mask(  # noqa: SLF001
        model,
        *args,
    )
    model._postprocess_tile_mask_crop.side_effect = lambda *args: DETRInstanceSegmentation._postprocess_tile_mask_crop(  # noqa: SLF001
        model,
        *args,
    )
    return model


class TestInstanceSegmentationTilerInit:
    def test_init(self):
        model = _make_model()
        tiler = InstanceSegmentationTiler(model, execution_mode="sync")
        assert tiler.tile_classifier_model is None

    def test_init_with_classifier(self):
        model = _make_model()
        classifier = MagicMock()
        tiler = InstanceSegmentationTiler(model, execution_mode="sync", tile_classifier_model=classifier)
        assert tiler.tile_classifier_model is classifier


class TestInstanceSegmentationTilerFilterTiles:
    def test_filter_tiles_no_classifier(self):
        model = _make_model()
        tiler = InstanceSegmentationTiler(model, execution_mode="sync")
        coords = [[0, 0, 100, 100], [100, 0, 200, 100]]
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        assert tiler._filter_tiles(image, coords) == coords  # noqa: SLF001

    def test_filter_tiles_with_classifier(self):
        model = _make_model()
        model.base_preprocess.return_value = ({"data": "preprocessed"}, {})
        classifier = MagicMock()
        # First tile (i==0) always kept, second filtered by threshold
        classifier.infer_sync.side_effect = [
            {"tile_prob": 0.1},  # tile 0 - kept (always first)
            {"tile_prob": 0.5},  # tile 1 - kept (above threshold)
            {"tile_prob": 0.1},  # tile 2 - filtered
        ]
        tiler = InstanceSegmentationTiler(model, execution_mode="sync", tile_classifier_model=classifier)
        image = np.zeros((100, 300, 3), dtype=np.uint8)
        coords = [[0, 0, 100, 100], [100, 0, 200, 100], [200, 0, 300, 100]]
        result = tiler._filter_tiles(image, coords)  # noqa: SLF001
        assert len(result) == 2
        assert coords[0] in result
        assert coords[1] in result


class TestInstanceSegmentationTilerPostprocessTile:
    def test_postprocess_tile(self):
        model = _make_model()
        tiler = InstanceSegmentationTiler(model, execution_mode="async")
        pred = _make_instance_seg_result(n=1)
        coord = [10, 20, 110, 120]
        result = tiler._postprocess_tile(pred, coord)  # noqa: SLF001
        assert "masks" in result
        assert "bboxes" in result
        assert len(result["masks"]) == 1


class TestInstanceSegmentationTilerMergeResults:
    @patch("model_api.tilers.instance_segmentation.multiclass_nms")
    @patch("model_api.tilers.instance_segmentation._segm_postprocess_crop")
    def test_merge_results_with_detections(self, mock_segm, mock_nms):
        model = _make_model()
        tiler = InstanceSegmentationTiler(model, execution_mode="sync")
        mock_nms.return_value = np.array([0])
        mock_segm.return_value = (np.ones((100, 100), dtype=np.uint8), 0, 0)

        results = [
            {
                "bboxes": np.array([[0, 0.9, 10, 20, 50, 60]], dtype=np.float32),
                "features": np.array([1.0]),
                "saliency_map": [],
                "coords": [0, 0, 100, 100],
                "masks": [np.ones((40, 40), dtype=np.uint8)],
            },
        ]
        merged = tiler._merge_results(results, (100, 100, 3))  # noqa: SLF001
        assert isinstance(merged, InstanceSegmentationResult)

    @pytest.mark.parametrize("box", _BOXES)
    def test_segm_postprocess_matches_legacy(self, box):
        raw = np.random.default_rng(0).random((28, 28)).astype(np.float32)
        box = np.array(box, dtype=np.float32)
        np.testing.assert_array_equal(
            _segm_postprocess(box, raw, 100, 120),
            _legacy_segm_postprocess(box, raw, 100, 120),
        )

    @pytest.mark.parametrize("box", _BOXES)
    def test_merge_results_crops_match_legacy_dense(self, box):
        """Merged result keeps compact crops; lazily densified `masks` equals the legacy dense output."""
        tiler = InstanceSegmentationTiler(_make_model(), execution_mode="sync")
        raw = np.random.default_rng(1).random((28, 28)).astype(np.float32)
        results = [
            {
                "bboxes": np.array([[0, 0.9, *box]], dtype=np.float32),
                "features": np.array([1.0]),
                "saliency_map": [],
                "coords": [0, 0, 120, 100],
                "masks": [raw],
            },
        ]
        merged = tiler._merge_results(results, (100, 120, 3))  # noqa: SLF001
        assert merged._masks is None  # noqa: SLF001
        assert merged.mask_crops[0][0].size < 100 * 120
        expected = _legacy_segm_postprocess(np.array(box, dtype=np.float32), raw, 100, 120)
        assert merged.masks.shape == (1, 100, 120)
        np.testing.assert_array_equal(merged.masks[0], expected)

    @pytest.mark.parametrize("box", _BOXES)
    def test_maskrcnn_crop_hook_matches_legacy_dense(self, box):
        model = object.__new__(MaskRCNNModel)
        raw = np.random.default_rng(2).random((28, 28)).astype(np.float32)
        box = np.array(box, dtype=np.float32)
        crop, x0, y0 = model._postprocess_tile_mask_crop(box, raw, [0, 0, 120, 100], 100, 120)  # noqa: SLF001
        dense = np.zeros((100, 120), dtype=np.uint8)
        dense[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
        np.testing.assert_array_equal(dense, _legacy_segm_postprocess(box, raw, 100, 120))

    def test_detr_crop_is_trimmed_and_matches_dense(self):
        """A full-image tile must not keep an (H, W) mask per detection; dense output stays identical."""
        model = object.__new__(DETRInstanceSegmentation)
        raw = np.zeros((96, 96), dtype=np.float32)
        raw[40:50, 60:75] = 1
        box, coord = np.array([0, 0, 5, 5], dtype=np.float32), [20, 10, 220, 210]
        crop, x0, y0 = model._postprocess_tile_mask_crop(box, raw, coord, 400, 300)  # noqa: SLF001
        assert crop.size < 0.1 * 200 * 200
        dense = np.zeros((400, 300), dtype=np.uint8)
        dense[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
        np.testing.assert_array_equal(dense, model._postprocess_tile_mask(box, raw, coord, 400, 300))  # noqa: SLF001
        assert crop.base is None

    def test_detr_crop_empty_mask(self):
        model = object.__new__(DETRInstanceSegmentation)
        crop, x0, y0 = model._postprocess_tile_mask_crop(  # noqa: SLF001
            np.zeros(4, dtype=np.float32),
            np.zeros((96, 96), dtype=np.float32),
            [20, 10, 220, 210],
            400,
            300,
        )
        assert crop.shape == (0, 0)
        assert (x0, y0) == (20, 10)

    def test_base_crop_hook_respects_subclass_decoding(self):
        """The base hook goes through `_postprocess_single_mask`, so custom decoding is not bypassed."""

        class _Custom(InstanceSegmentationModel):
            __model__ = "CustomIS"

            def _postprocess_single_mask(self, box, raw_cls_mask, im_h, im_w):
                mask = np.zeros((im_h, im_w), dtype=np.uint8)
                mask[5:8, 9:12] = 1
                return mask

        crop, x0, y0 = object.__new__(_Custom)._postprocess_tile_mask_crop(  # noqa: SLF001
            np.zeros(4),
            np.zeros((4, 4)),
            [0, 0, 50, 50],
            50,
            50,
        )
        assert crop.shape == (3, 3)
        assert (x0, y0) == (9, 5)

    def test_merge_results_empty(self):
        model = _make_model()
        tiler = InstanceSegmentationTiler(model, execution_mode="sync")
        results = [
            {
                "bboxes": np.empty((0, 6), dtype=np.float32),
                "features": np.array([1.0]),
                "saliency_map": [],
                "coords": [0, 0, 100, 100],
                "masks": [],
            },
        ]
        merged = tiler._merge_results(results, (100, 100, 3))  # noqa: SLF001
        assert isinstance(merged, InstanceSegmentationResult)


class TestInstanceSegmentationTilerMergeSaliencyMaps:
    def test_merge_saliency_maps_empty(self):
        model = _make_model()
        tiler = InstanceSegmentationTiler(model, execution_mode="sync")
        assert tiler._merge_saliency_maps([], (100, 100, 3), []) is None  # noqa: SLF001

    def test_merge_saliency_maps_falsy_first(self):
        model = _make_model()
        tiler = InstanceSegmentationTiler(model, execution_mode="sync")
        result = tiler._merge_saliency_maps([[]], (100, 100, 3), [[0, 0, 100, 100]])  # noqa: SLF001
        assert result == []

    def test_merge_saliency_maps_empty_numpy_array(self):
        tiler = InstanceSegmentationTiler(_make_model(), execution_mode="sync")
        result = tiler._merge_saliency_maps([np.array([])], (100, 100, 3), [[0, 0, 100, 100]])  # noqa: SLF001
        assert result.size == 0

    def test_merge_saliency_maps_with_tiles(self):
        model = _make_model()
        tiler = InstanceSegmentationTiler(
            model,
            configuration={"tile_size": 50, "tile_with_full_img": True},
            execution_mode="sync",
        )
        smap_full = [np.ones((10, 20), dtype=np.float32) * 100, np.ones((10, 20), dtype=np.float32) * 50]
        smap_tile = [np.ones((10, 10), dtype=np.float32) * 80, np.ones((10, 10), dtype=np.float32) * 40]
        coords = [[0, 0, 100, 50], [0, 0, 50, 50]]
        result = tiler._merge_saliency_maps([smap_full, smap_tile], (50, 100, 3), coords)  # noqa: SLF001
        assert len(result) == 2

    def test_merge_saliency_maps_1d_class_map_skipped(self):
        model = _make_model()
        tiler = InstanceSegmentationTiler(
            model,
            configuration={"tile_size": 50, "tile_with_full_img": False},
            execution_mode="sync",
        )
        smap = [np.array([1.0]), np.array([2.0])]
        coords = [[0, 0, 50, 50]]
        result = tiler._merge_saliency_maps([smap], (50, 50, 3), coords)  # noqa: SLF001
        # 1d maps - rounded and checked for sum==0
        assert len(result) == 2

    def test_merge_saliency_maps_no_full_img(self):
        model = _make_model()
        tiler = InstanceSegmentationTiler(
            model,
            configuration={"tile_size": 50, "tile_with_full_img": False},
            execution_mode="sync",
        )
        smap1 = [np.ones((10, 10), dtype=np.float32) * 100]
        smap2 = [np.ones((10, 10), dtype=np.float32) * 50]
        coords = [[0, 0, 50, 50], [50, 0, 100, 50]]
        result = tiler._merge_saliency_maps([smap1, smap2], (50, 100, 3), coords)  # noqa: SLF001
        assert len(result) == 1

    def test_merge_saliency_maps_zero_sum(self):
        model = _make_model()
        tiler = InstanceSegmentationTiler(
            model,
            configuration={"tile_size": 50, "tile_with_full_img": False},
            execution_mode="sync",
        )
        # All zeros - should result in ndarray(0) for 1d case
        smap = [np.array([0.0])]
        coords = [[0, 0, 50, 50]]
        result = tiler._merge_saliency_maps([smap], (50, 50, 3), coords)  # noqa: SLF001
        # 1d map with sum 0 gets replaced with ndarray(0)
        assert result[0].shape == (0,)


class TestInstanceSegmentationTilerPredictTiles:
    @pytest.mark.parametrize("external_tiles", [False, True])
    def test_detr_masks_cover_tile_not_box(self, external_tiles):
        """DETR masks use tile coordinates in both full-image and pre-tiled inference."""
        model = _make_detr_model()

        masks = np.zeros((2, 10, 10), dtype=np.float32)
        masks[0, 3:6, 2:5] = 1
        masks[1, 1:4, 6:9] = 1

        def result(i):
            box = [[20, 30, 50, 60], [60, 10, 90, 40]][i]
            return InstanceSegmentationResult(
                bboxes=np.array([box], dtype=np.float32),
                labels=np.array([0], dtype=np.int32),
                scores=np.array([0.9 - i * 0.1], dtype=np.float32),
                masks=masks[i : i + 1],
                saliency_map=[],
                feature_vector=np.empty(0),
            )

        model.side_effect = [result(0), result(1)]
        tiler = InstanceSegmentationTiler(
            model,
            configuration={"tile_size": 100, "tiles_overlap": 0.0, "tile_with_full_img": False},
            execution_mode="sync",
        )
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        if external_tiles:
            merged = tiler.predict_tiles(
                [image[:, :100], image[:, 100:]],
                [[0, 0, 100, 100], [100, 0, 200, 100]],
                image.shape,
            )
        else:
            merged = tiler(image)
        expected = np.zeros((2, 100, 200), dtype=np.uint8)
        expected[0, :, :100] = _full_image_mask_postprocess(masks[0], 100, 100)
        expected[1, :, 100:] = _full_image_mask_postprocess(masks[1], 100, 100)
        np.testing.assert_array_equal(merged.bboxes, [[20, 30, 50, 60], [160, 10, 190, 40]])
        np.testing.assert_array_equal(merged.masks, expected)

    def test_detr_masks_follow_nms_and_rectangular_edge_tile(self):
        model = _make_detr_model()
        coords = [[0, 0, 100, 100], [50, 0, 150, 100], [150, 60, 200, 100]]
        masks = np.zeros((3, 10, 10), dtype=np.float32)
        masks[0, 2:6, 6:10] = 1
        masks[1, 2:6, 1:5] = 1
        masks[2, 2:7, 4:8] = 1
        boxes = [[60, 20, 100, 60], [10, 20, 50, 60], [20, 8, 40, 28]]
        scores = [0.5, 0.9, 0.8]
        model.side_effect = [
            InstanceSegmentationResult(
                bboxes=np.array([box], dtype=np.float32),
                labels=np.array([0], dtype=np.int32),
                scores=np.array([score], dtype=np.float32),
                masks=mask[None],
                saliency_map=[],
                feature_vector=np.empty(0),
            )
            for box, score, mask in zip(boxes, scores, masks)
        ]
        tiler = InstanceSegmentationTiler(model, execution_mode="sync")
        tiles = [np.zeros((100, 100, 3)), np.zeros((100, 100, 3)), np.zeros((40, 50, 3))]
        merged = tiler.predict_tiles(tiles, coords, (100, 200, 3))
        np.testing.assert_array_equal(merged.bboxes, [[60, 20, 100, 60], [170, 68, 190, 88]])
        expected = np.zeros((2, 100, 200), dtype=np.uint8)
        expected[0, :, 50:150] = _full_image_mask_postprocess(masks[1], 100, 100)
        expected[1, 60:100, 150:200] = _full_image_mask_postprocess(masks[2], 40, 50)
        np.testing.assert_array_equal(merged.masks, expected)

    def test_maskrcnn_keeps_existing_box_mask_placement(self):
        model = object.__new__(MaskRCNNModel)
        box = np.array([45, 10, 60, 30])
        mask = np.ones((10, 10), dtype=np.float32)
        actual = model._postprocess_tile_mask(box, mask, [0, 0, 50, 50], 50, 100)  # noqa: SLF001
        np.testing.assert_array_equal(actual, _segm_postprocess(box, mask, 50, 100))

    @patch("model_api.tilers.instance_segmentation.multiclass_nms")
    @patch("model_api.tilers.instance_segmentation._segm_postprocess_crop")
    def test_predict_tiles_sync_end_to_end(self, mock_segm, mock_nms):
        """predict_tiles infers each pre-cropped tile once and merges masks + boxes."""
        model = _make_model()
        mock_nms.return_value = np.array([0])
        mock_segm.return_value = (np.ones((50, 100), dtype=np.uint8), 0, 0)
        tiler = InstanceSegmentationTiler(model, execution_mode="sync")
        model.side_effect = [_make_instance_seg_result(n=1), _make_instance_seg_result(n=1)]
        tiles = [np.zeros((50, 50, 3), dtype=np.uint8), np.zeros((50, 50, 3), dtype=np.uint8)]
        coords = [[0, 0, 50, 50], [50, 0, 100, 50]]
        # Saliency-map merging is exercised separately; keep this focused on box/mask merging.
        with patch.object(tiler, "_merge_saliency_maps", return_value=[]):
            merged = tiler.predict_tiles(tiles, coords, (50, 100, 3))
        assert isinstance(merged, InstanceSegmentationResult)
        assert model.call_count == 2

    def test_setup_model_toggles_postprocess_flag(self):
        """_setup_model disables semantic-mask postprocessing while tiling, then restores it."""
        from model_api.models.instance_segmentation import MaskRCNNModel

        model = MagicMock(spec=MaskRCNNModel)
        model.load = MagicMock()
        model.inference_adapter = MagicMock()
        model.inference_adapter.get_rt_info.side_effect = RuntimeError(
            "Cannot get runtime attribute. Path to runtime attribute is incorrect.",
        )
        model.params = MagicMock()
        model.params.postprocess_semantic_masks = True

        tiler = InstanceSegmentationTiler(model, execution_mode="sync")
        with tiler._setup_model():  # noqa: SLF001
            assert model._postprocess_semantic_masks is False  # noqa: SLF001
        assert model._postprocess_semantic_masks is True  # noqa: SLF001


class TestInstanceSegmentationTilerCall:
    def test_call_with_maskrcnn(self):
        model = _make_model()
        model.__class__ = MagicMock
        # Mock isinstance check
        with patch("model_api.tilers.instance_segmentation.isinstance") as mock_isinstance:
            mock_isinstance.side_effect = (
                lambda obj, cls: cls == MagicMock if cls != InstanceSegmentationTiler else False
            )
            # Just test that __call__ delegates to super().__call__
            tiler = InstanceSegmentationTiler(model, execution_mode="sync")
            with patch.object(InstanceSegmentationTiler.__bases__[0], "__call__", return_value="result"):
                # Can't easily mock isinstance for MaskRCNNModel, test non-MaskRCNN path
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                tiler(image)

    def test_call_with_actual_maskrcnn_mock(self):
        from model_api.models.instance_segmentation import MaskRCNNModel

        model = MagicMock(spec=MaskRCNNModel)
        model.load = MagicMock()
        model.inference_adapter = MagicMock()
        model.inference_adapter.get_rt_info.side_effect = RuntimeError(
            "Cannot get runtime attribute. Path to runtime attribute is incorrect.",
        )
        model.params = MagicMock()
        model.params.postprocess_semantic_masks = True
        model.return_value = _make_instance_seg_result(n=1)

        tiler = InstanceSegmentationTiler(
            model,
            execution_mode="sync",
            configuration={"tile_size": 100, "tiles_overlap": 0.0, "tile_with_full_img": False},
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(tiler, "_merge_results", return_value="merged"):
            result = tiler(image)
            assert result == "merged"

    def test_call_non_maskrcnn(self):
        model = _make_model()
        model.return_value = _make_instance_seg_result(n=1)
        tiler = InstanceSegmentationTiler(
            model,
            execution_mode="sync",
            configuration={"tile_size": 100, "tiles_overlap": 0.0, "tile_with_full_img": False},
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(tiler, "_merge_results", return_value="merged"):
            result = tiler(image)
            assert result == "merged"
