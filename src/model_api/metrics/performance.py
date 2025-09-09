#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import logging

from .time_stat import TimeStat

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    A class to represent performance metrics for a model.
    """

    def __init__(self):
        """
        Initializes performance metrics.
        """
        self.load_time = TimeStat()
        self.preprocess_time = TimeStat()
        self.inference_time = TimeStat()
        self.postprocess_time = TimeStat()
        self.total_time = TimeStat()

    def __add__(self, other):
        """
        Adds two PerformanceMetrics objects.
        """
        if not isinstance(other, PerformanceMetrics):
            return NotImplemented

        new_metrics = PerformanceMetrics()
        new_metrics.load_time = self.load_time + other.load_time
        new_metrics.preprocess_time = self.preprocess_time + other.preprocess_time
        new_metrics.inference_time = self.inference_time + other.inference_time
        new_metrics.postprocess_time = self.postprocess_time + other.postprocess_time
        return new_metrics

    def reset(self) -> None:
        """
        Resets performance metrics to the initial state.
        """
        self.preprocess_time.reset()
        self.inference_time.reset()
        self.postprocess_time.reset()
        self.total_time.reset()

    def get_load_time(self) -> TimeStat:
        """
        Returns the load time statistics.

        Returns:
            TimeStat: Load time statistics object.
        """
        return self.load_time

    def get_preprocess_time(self) -> TimeStat:
        """
        Returns the preprocessing time statistics.

        Returns:
            TimeStat: Preprocessing time statistics object.
        """
        return self.preprocess_time

    def get_inference_time(self) -> TimeStat:
        """
        Returns the inference time statistics.

        Returns:
            TimeStat: Inference time statistics object.
        """
        return self.inference_time

    def get_postprocess_time(self) -> TimeStat:
        """
        Returns the postprocessing time statistics.

        Returns:
            TimeStat: Postprocessing time statistics object.
        """
        return self.postprocess_time

    def get_total_frames(self) -> int:
        """
        Returns the total number of frames processed.

        Returns:
            int: Total number of frames processed.
        """
        return len(self.total_time.durations)

    def get_fps(self) -> float:
        """
        Returns the Frames Per Second (FPS) statistics.

        Returns:
            float: Frames Per Second.
        """
        return self.get_total_frames() / sum(self.total_time.durations) if sum(self.total_time.durations) > 0 else 0.0

    def get_total_time_min(self) -> float:
        """
        Returns the minimum total time for processing a frame.

        Returns:
            float: Minimum total time in seconds.
        """
        return min(self.total_time.durations) if self.total_time.durations else 0.0

    def get_total_time_max(self) -> float:
        """
        Returns the maximum total time for processing a frame.

        Returns:
            float: Maximum total time in seconds.
        """
        return max(self.total_time.durations) if self.total_time.durations else 0.0

    def log_metrics(self) -> None:
        """
        Logs all performance metrics using the logging module.
        """
        # Create the metrics report as a multi-line string
        report_lines = [
            "",
            "=" * 60,
            "🚀 PERFORMANCE METRICS REPORT 🚀".center(60),
            "=" * 60,
            "",
            "📊 Model Loading:",
            f"   Load Time: {self.load_time.mean():.3f}s",
            "",
            "⚙️  Processing Times (mean ± std):",
            f"   Preprocess:  {self.preprocess_time.mean():.3f}s ± {self.preprocess_time.stddev():.3f}s",
            f"   Inference:   {self.inference_time.mean():.3f}s ± {self.inference_time.stddev():.3f}s",
            f"   Postprocess: {self.postprocess_time.mean():.3f}s ± {self.postprocess_time.stddev():.3f}s",
            "",
            "📈 Total Time Statistics:",
            f"   Mean:  {self.total_time.mean():.3f}s ± {self.total_time.stddev():.3f}s",
            f"   Min:   {self.get_total_time_min():.3f}s",
            f"   Max:   {self.get_total_time_max():.3f}s",
            "",
            "🎯 Performance Summary:",
            f"   Total Frames: {self.get_total_frames():,}",
            f"   FPS:          {self.get_fps():.2f}",
            "",
            "=" * 60,
            "",
        ]

        # Log the entire report as a single info message
        logger.info("\n".join(report_lines))
