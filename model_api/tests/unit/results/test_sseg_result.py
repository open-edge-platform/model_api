#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from unittest.mock import patch

import numpy as np
import pytest
from model_api.models.result import Contour
from model_api.models.result.segmentation import ImageResultWithSoftPrediction


class TestImageResultWithSoftPredictionHist:
    """Tests for hist() method supporting OpenCV 4 and 5 return types."""

    @pytest.fixture
    def result_image(self):
        """Create a simple test image with known pixel distribution."""
        # 10x10 image: 50 pixels of value 0, 30 pixels of value 128, 20 pixels of value 255
        img = np.zeros((10, 10), dtype=np.uint8)
        img[0:5, :] = 0  # 50 pixels
        img[5:8, :] = 128  # 30 pixels
        img[8:10, :] = 255  # 20 pixels
        return img

    @pytest.fixture
    def image_result(self, result_image):
        """Create ImageResultWithSoftPrediction instance."""
        return ImageResultWithSoftPrediction(
            resultImage=result_image,
            soft_prediction=np.zeros((10, 10, 3)),
            saliency_map=np.zeros((10, 10)),
            feature_vector=np.zeros((256,)),
        )

    def test_hist_with_opencv4_column_vector(self, image_result):
        """OpenCV 4 returns column vector [N, 1] from calcHist."""
        # Simulate OpenCV 4 output: column vector shape [256, 1]
        opencv4_hist = np.zeros((256, 1), dtype=np.float32)
        opencv4_hist[0, 0] = 50.0  # 50 pixels of value 0
        opencv4_hist[128, 0] = 30.0  # 30 pixels of value 128
        opencv4_hist[255, 0] = 20.0  # 20 pixels of value 255

        with patch("cv2.calcHist", return_value=opencv4_hist):
            hist = image_result.hist()

        assert hist == {"0": 0.5, "128": 0.3, "255": 0.2}

    def test_hist_with_opencv5_1d_array(self, image_result):
        """OpenCV 5+ returns 1D array [N] from calcHist."""
        # Simulate OpenCV 5 output: 1D array shape [256]
        opencv5_hist = np.zeros((256,), dtype=np.float32)
        opencv5_hist[0] = 50.0  # 50 pixels of value 0
        opencv5_hist[128] = 30.0  # 30 pixels of value 128
        opencv5_hist[255] = 20.0  # 20 pixels of value 255

        with patch("cv2.calcHist", return_value=opencv5_hist):
            hist = image_result.hist()

        assert hist == {"0": 0.5, "128": 0.3, "255": 0.2}

    def test_hist_both_formats_produce_same_result(self, image_result):
        """Both OpenCV formats produce identical histogram output."""
        opencv4_hist = np.zeros((256, 1), dtype=np.float32)
        opencv4_hist[10, 0] = 25.0
        opencv4_hist[20, 0] = 75.0

        opencv5_hist = np.zeros((256,), dtype=np.float32)
        opencv5_hist[10] = 25.0
        opencv5_hist[20] = 75.0

        with patch("cv2.calcHist", return_value=opencv4_hist):
            hist_v4 = image_result.hist()

        with patch("cv2.calcHist", return_value=opencv5_hist):
            hist_v5 = image_result.hist()

        assert hist_v4 == hist_v5


def test_contour_type():
    contour = Contour(
        shape=[(100, 100)],
        label=1,
        probability=0.9,
        excluded_shapes=[[(50, 50)], [(60, 60)]],
    )

    assert isinstance(contour.shape, np.ndarray)
    assert isinstance(contour.excluded_shapes, list)
    assert isinstance(contour.excluded_shapes[0], np.ndarray)
    assert contour.label == 1
    assert contour.probability == 0.9
    assert np.array_equal(contour.excluded_shapes, np.array([[(50, 50)], [(60, 60)]]))
