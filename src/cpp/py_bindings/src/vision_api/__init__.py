#!/usr/bin/env python3

try:
    from openvino import Core

    _ = Core()  # Triggers loading of shared libs like libopenvino.so
except Exception as e:
    raise ImportError(f"Failed to initialize OpenVINO runtime: {e}")

from ._vision_api import ClassificationModel

__all__ = [ClassificationModel]
