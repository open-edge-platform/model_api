#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Converter registry mapping model_library names to converter classes."""

from model_converter.converters.base import BaseConverter
from model_converter.converters.getitune import GetituneConverter
from model_converter.converters.timm import TimmConverter
from model_converter.converters.torchvision import TorchvisionConverter
from model_converter.converters.yolo import YoloConverter

CONVERTER_REGISTRY: dict[str, type[BaseConverter]] = {
    "torchvision": TorchvisionConverter,
    "timm": TimmConverter,
    "yolo": YoloConverter,
    "getitune": GetituneConverter,
}
