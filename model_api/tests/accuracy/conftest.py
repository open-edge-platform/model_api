#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import json
from pathlib import Path

import pytest
from tests.accuracy.comparator.pytest_plugin import (
    add_comparator_options,
    enforce_ci_guard,
    reference_mode,
)

__all__ = ["reference_mode"]


def pytest_addoption(parser):
    parser.addoption("--data", action="store", help="data folder with dataset")
    parser.addoption(
        "--model_data",
        action="store",
        default="public_scope.json",
        help="path to model data JSON file for test parameterization",
    )
    parser.addoption(
        "--device",
        action="store",
        default="CPU",
        help="device to run tests on (in case of OpenvinoAdapter)",
    )
    parser.addoption(
        "--dump",
        action="store_true",
        default=False,
        help="whether to dump results into json file",
    )
    parser.addoption(
        "--results-dir",
        action="store",
        default="",
        help="directory to store inference result",
    )
    add_comparator_options(parser)


def pytest_configure(config):
    config.test_results = []
    enforce_ci_guard(config)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()

    if result.when == "call":
        test_results = item.config.test_results

        if not test_results:
            return

        with Path("test_scope.json").open("w") as outfile:
            json.dump(test_results, outfile, indent=4)
