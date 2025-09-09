#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from time import perf_counter


class TimeStat:
    """
    A class to represent a statistical time metric.
    """

    def __init__(self):
        """
        Initializes the TimeStat object.
        """
        self.time = 0.0
        self.durations = []
        self.count = 0
        self.last_update_time = None

    def __add__(self, other):
        """
        Adds two TimeStat objects.

        Returns:
            TimeStat: A new TimeStat object representing the sum of the two.
        """
        if not isinstance(other, TimeStat):
            return NotImplemented

        new_stat = TimeStat()
        new_stat.time = self.time + other.time
        new_stat.durations = self.durations + other.durations
        new_stat.count = self.count + other.count
        return new_stat

    def update(self) -> None:
        """
        Updates the statistics with the latest duration.
        """
        time = perf_counter()
        if self.last_update_time:
            diff = time - self.last_update_time
            self.time += diff
            self.durations.append(diff)
            self.count += 1
            self.last_update_time = None
        else:
            self.last_update_time = time

    def reset(self) -> None:
        """
        Resets the statistics to their initial state.
        """
        self.time = 0.0
        self.durations = []
        self.count = 0
        self.last_update_time = None

    def mean(self) -> float:
        """
        Calculates the mean of the recorded durations.

        Returns:
            float: The mean of the recorded durations.
        """
        return self.time / self.count if self.count != 0 else 0.0

    def stddev(self) -> float:
        """
        Calculates the standard deviation of the recorded durations.

        Returns:
            float: The standard deviation of the recorded durations.
        """
        if self.count == 0:
            return 0.0
        mean = self.mean()
        variance = sum((x - mean) ** 2 for x in self.durations) / self.count
        return variance**0.5
