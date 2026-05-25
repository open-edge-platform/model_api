"""Pytest plugin entrypoint for comparator-based accuracy assertions."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest

_UPDATE_REFERENCES = "--update-references"
_STRICT_TOLERANCES = "--strict-tolerances"
_FORCE_UPDATE_IN_CI = "--force-update-in-ci"


def add_comparator_options(parser: pytest.Parser) -> None:
    """Register comparator-specific CLI options on ``parser``."""
    parser.addoption(
        _UPDATE_REFERENCES,
        action="store_true",
        default=False,
        help="Regenerate reference artifacts instead of comparing",
    )
    parser.addoption(
        _STRICT_TOLERANCES,
        action="store_true",
        default=False,
        help="Use strict tolerances for all comparisons",
    )
    parser.addoption(
        _FORCE_UPDATE_IN_CI,
        action="store_true",
        default=False,
        help="Override CI guard for --update-references (use with caution)",
    )


def enforce_ci_guard(config: pytest.Config) -> None:
    """Refuse ``--update-references`` when ``CI=true`` unless override is set."""
    if not config.getoption(_UPDATE_REFERENCES, default=False):
        return
    if os.environ.get("CI") != "true":
        return
    if config.getoption(_FORCE_UPDATE_IN_CI, default=False):
        return
    msg = "--update-references is forbidden when CI=true; pass --force-update-in-ci to override"
    raise pytest.UsageError(msg)


def pytest_addoption(parser: pytest.Parser) -> None:
    add_comparator_options(parser)


def pytest_configure(config: pytest.Config) -> None:
    enforce_ci_guard(config)


@pytest.fixture
def reference_mode(pytestconfig: pytest.Config) -> str:
    """Return ``'update'`` if ``--update-references`` is set, else ``'assert'``."""
    if pytestconfig.getoption(_UPDATE_REFERENCES, default=False):
        return "update"
    return "assert"
