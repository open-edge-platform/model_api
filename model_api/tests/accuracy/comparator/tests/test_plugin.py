"""Tests for the comparator pytest plugin."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]


CONFTEST = """
from tests.accuracy.comparator.pytest_plugin import (
    add_comparator_options,
    enforce_ci_guard,
    reference_mode,
)

__all__ = ["reference_mode"]


def pytest_addoption(parser):
    add_comparator_options(parser)


def pytest_configure(config):
    enforce_ci_guard(config)
"""


@pytest.fixture
def plugin_pytester(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest(CONFTEST)
    return pytester


def test_update_references_flag_parsed(plugin_pytester: pytest.Pytester) -> None:
    plugin_pytester.makepyfile(
        test_x="""
        def test_flag(request):
            assert request.config.getoption('--update-references') is True
            assert request.config.getoption('--strict-tolerances') is True
        """,
    )
    result = plugin_pytester.runpytest("--update-references", "--strict-tolerances")
    assert result.ret == 0, f"Expected exit 0, got {result.ret}\n{result.stdout.str()}"


def test_reference_mode_fixture_default(plugin_pytester: pytest.Pytester) -> None:
    plugin_pytester.makepyfile(
        test_x="""
        def test_mode(reference_mode):
            assert reference_mode == 'assert'
        """,
    )
    result = plugin_pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_reference_mode_fixture_update(plugin_pytester: pytest.Pytester) -> None:
    plugin_pytester.makepyfile(
        test_x="""
        def test_mode(reference_mode):
            assert reference_mode == 'update'
        """,
    )
    result = plugin_pytester.runpytest("--update-references")
    assert result.ret == 0, f"Expected exit 0, got {result.ret}\n{result.stdout.str()}"


def test_ci_guard_blocks_update(
    plugin_pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CI", "true")
    plugin_pytester.makepyfile(test_x="def test_noop(): pass")
    result = plugin_pytester.runpytest("--update-references")
    assert result.ret != 0
    result.stderr.fnmatch_lines(["*--update-references is forbidden when CI=true*"])


def test_ci_guard_override_allows_update(
    plugin_pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CI", "true")
    plugin_pytester.makepyfile(test_x="def test_noop(): pass")
    result = plugin_pytester.runpytest("--update-references", "--force-update-in-ci")
    result.assert_outcomes(passed=1)


def test_ci_guard_inactive_when_not_ci(
    plugin_pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CI", raising=False)
    plugin_pytester.makepyfile(test_x="def test_noop(): pass")
    result = plugin_pytester.runpytest("--update-references")
    result.assert_outcomes(passed=1)
