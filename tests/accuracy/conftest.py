#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import json

import pytest


def pytest_addoption(parser):
    parser.addoption("--data", action="store", help="data folder with dataset")
    parser.addoption(
        "--dump",
        action="store_true",
        default=False,
        help="whether to dump results into json file",
    )


def pytest_configure(config):
    config.test_results = []


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()

    if result.when == "call":
        test_results = item.config.test_results
        with open("test_scope.json", "w") as outfile:
            json.dump(test_results, outfile, indent=4)
