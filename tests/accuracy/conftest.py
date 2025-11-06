#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


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
        "--results-dir",
        action="store",
        default="",
        help="directory to store inference result",
    )


def pytest_configure(config):
    config.test_results = []
