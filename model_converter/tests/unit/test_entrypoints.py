#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for model_converter entry points."""

import runpy
import sys
from unittest.mock import patch


class TestInitExports:
    """Tests for model_converter.__init__ exports."""

    def test_exports_model_converter(self):
        """__init__ exports ModelConverter class."""
        from model_converter import ModelConverter

        assert ModelConverter is not None

    def test_exports_list_models(self):
        """__init__ exports list_models function."""
        from model_converter import list_models

        assert callable(list_models)

    def test_exports_main(self):
        """__init__ exports main function."""
        from model_converter import main

        assert callable(main)

    def test_all_attribute(self):
        """__init__ defines __all__ correctly."""
        import model_converter

        assert "ModelConverter" in model_converter.__all__
        assert "list_models" in model_converter.__all__
        assert "main" in model_converter.__all__


class TestMainModule:
    """Tests for model_converter.__main__."""

    def test_calls_main(self):
        """__main__ calls cli.main() and sys.exit."""
        with (
            patch("model_converter.cli.main", return_value=0) as mock_main,
            patch.object(sys, "exit") as mock_exit,
        ):
            runpy.run_module("model_converter", run_name="__main__", alter_sys=True)

        mock_main.assert_called_once()
        mock_exit.assert_called_once_with(0)


class TestModelConverterScript:
    """Tests for model_converter.model_converter legacy entry point."""

    def test_calls_main_when_run(self):
        """model_converter.py calls main() when run as __main__."""
        with (
            patch("model_converter.cli.main", return_value=0) as mock_main,
            patch.object(sys, "exit") as mock_exit,
        ):
            runpy.run_module("model_converter.model_converter", run_name="__main__", alter_sys=True)

        mock_main.assert_called_once()
        mock_exit.assert_called_once_with(0)
