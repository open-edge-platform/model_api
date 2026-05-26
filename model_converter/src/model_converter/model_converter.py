#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
Models to OpenVINO Model Converter

Usage:
    uv run python model_converter.py config.json -o ./output_models

"""

import sys

from model_converter.cli import main

if __name__ == "__main__":
    sys.exit(main())
