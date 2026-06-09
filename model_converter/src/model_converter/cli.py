#!/usr/bin/env -S uv run --script
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
PyTorch to OpenVINO Model Converter

Usage:
    uv run python model_converter.py config.json -o ./output_models

"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from model_converter.converters import CONVERTER_REGISTRY, BaseConverter
from model_converter.converters.getitune import GetituneConverter
from model_converter.dataset_registry import DatasetRegistry
from model_converter.reporting import (
    ConversionResult,
    format_console_table,
)


class ModelConverter:
    """Facade for model conversion that dispatches to specialized converters.

    Routes each model configuration to the appropriate converter based on
    the ``model_library`` field. Maintains backward compatibility with
    existing usage patterns.
    """

    def __init__(
        self,
        output_dir: Path,
        cache_dir: Path,
        verbose: bool = False,
        dataset_registry: DatasetRegistry | None = None,
        training_extensions_dir: Path | None = None,
        report_path: Path | None = None,
    ):
        """Initialize the ModelConverter.

        Args:
            output_dir: Directory to save converted models
            cache_dir: Directory to cache downloaded weights
            verbose: Enable verbose logging
            dataset_registry: Dataset registry for resolving dataset paths
            training_extensions_dir: Path to training_extensions repo (for getitune models)
            report_path: Path to the Markdown report file.  When set, each
                non-skipped conversion result is written to the file immediately
                after export.
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.dataset_registry = dataset_registry
        self.training_extensions_dir = Path(training_extensions_dir) if training_extensions_dir else None
        self.report_path = Path(report_path) if report_path else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Initialize converters
        self._converters: dict[str, BaseConverter] = {}
        self._verbose = verbose

    def _get_converter(self, model_library: str) -> BaseConverter:
        """Get or create a converter for the given model library.

        Args:
            model_library: Library name (torchvision, timm, yolo, getitune)

        Returns:
            Appropriate converter instance
        """
        if model_library not in self._converters:
            converter_cls = CONVERTER_REGISTRY.get(model_library)
            if converter_cls is None:
                error_msg = (
                    f"Unsupported model_library: '{model_library}'. "
                    f"Supported libraries: {list(CONVERTER_REGISTRY.keys())}"
                )
                raise ValueError(error_msg)

            kwargs: dict[str, Any] = {
                "output_dir": self.output_dir,
                "cache_dir": self.cache_dir,
                "verbose": self._verbose,
                "dataset_registry": self.dataset_registry,
                "report_path": self.report_path,
            }

            if converter_cls == GetituneConverter:
                kwargs["training_extensions_dir"] = self.training_extensions_dir

            self._converters[model_library] = converter_cls(**kwargs)

        return self._converters[model_library]

    def process_model_config(self, config: dict[str, Any]) -> bool:
        """Process a single model configuration by dispatching to the appropriate converter.

        Args:
            config: Model configuration dictionary

        Returns:
            True if successful, False otherwise
        """
        model_library = config.get("model_library", "torchvision")
        try:
            converter = self._get_converter(model_library)
            return converter.process_model_config(config)
        except ValueError as e:
            self.logger.error(f"✗ {e}")
            return False

    def process_config_file(
        self,
        config_path: Path,
        model_filter: str | None = None,
        library_filter: list[str] | None = None,
    ) -> tuple[int, int]:
        """Process models from a configuration file.

        Args:
            config_path: Path to JSON configuration file
            model_filter: Optional model short name to process (process only this model)
            library_filter: Optional list of library names to process (process only these libraries)

        Returns:
            Tuple of (successful_count, failed_count)
        """
        try:
            with Path(config_path).open() as f:
                config = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load configuration file: {e}")
            raise

        models = config.get("models", [])

        if not models:
            self.logger.warning("No models found in configuration file")
            return 0, 0

        self.logger.info(f"Configuration validated: {len(models)} models found")

        # Filter by library if requested
        if library_filter:
            unknown_libs = set(library_filter) - set(CONVERTER_REGISTRY.keys())
            for lib in unknown_libs:
                self.logger.warning(f"Unknown library '{lib}' (available: {list(CONVERTER_REGISTRY.keys())})")
            models = [m for m in models if m.get("model_library") in library_filter]
            if not models:
                self.logger.error(f"No models found for libraries: {library_filter}")
                return 0, 0
            self.logger.info(f"Filtering by libraries: {library_filter} ({len(models)} models)")

        # Filter models if requested
        if model_filter:
            models = [m for m in models if m.get("model_short_name") == model_filter]
            if not models:
                self.logger.error(f"Model '{model_filter}' not found in configuration")
                return 0, 0
            self.logger.info(f"Processing only model: {model_filter}")

        successful = 0
        failed = 0

        for model_config in models:
            if self.process_model_config(model_config):
                successful += 1
            else:
                failed += 1

        return successful, failed

    def collect_results(self) -> list[ConversionResult]:
        """Aggregate conversion results from every instantiated converter.

        Returns:
            The combined list of per-model conversion results.
        """
        results: list[ConversionResult] = []
        for converter in self._converters.values():
            results.extend(converter.results)
        return results


def list_models(config_path: Path, library_filter: list[str] | None = None) -> None:
    """List all models in a configuration file.

    Args:
        config_path: Path to JSON configuration file
        library_filter: Optional list of library names to filter by
    """
    try:
        with config_path.open() as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return

    models = config.get("models", [])

    if library_filter:
        unknown_libs = set(library_filter) - set(CONVERTER_REGISTRY.keys())
        for lib in unknown_libs:
            print(f"Warning: Unknown library '{lib}' (available: {list(CONVERTER_REGISTRY.keys())})", file=sys.stderr)
        models = [m for m in models if m.get("model_library") in library_filter]

    if not models:
        print("No models found in configuration")
        return

    print(f"\nFound {len(models)} models:\n")
    print(f"{'Short Name':<30} {'Full Name':<40} {'Library':<15} {'Type':<20}")
    print("-" * 105)

    for model in models:
        short_name = model.get("model_short_name", "N/A")
        full_name = model.get("model_full_name", "N/A")
        model_type = model.get("model_type", "N/A")
        library = model.get("model_library", "N/A")
        print(f"{short_name:<30} {full_name:<40} {library:<15} {model_type:<20}")

    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to OpenVINO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all models in config
  uv run python model_converter.py config.json -o ./models

  # Convert a specific model
  uv run python model_converter.py config.json -o ./models --model resnet50

  # List all models in config
  uv run python model_converter.py config.json --list

  # Enable verbose logging
  uv run python model_converter.py config.json -o ./models -v
        """,
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to JSON configuration file",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./converted_models"),
        help="Output directory for converted models (default: ./converted_models)",
    )

    parser.add_argument(
        "-c",
        "--cache",
        type=Path,
        default=Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
        help="Cache directory for downloaded weights (default: ~/.cache/torch/hub/checkpoints)",
    )

    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=Path("presets/datasets.json"),
        help="Path to datasets configuration JSON file (default: presets/datasets.json)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Process only the specified model (by model_short_name)",
    )

    parser.add_argument(
        "--library",
        type=str,
        default=None,
        help=(
            "Comma-separated list of model libraries to process (e.g., getitune,timm). "
            "If not provided, all libraries are exported."
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all models in the configuration file and exit",
    )

    parser.add_argument(
        "--report",
        nargs="?",
        const="",
        default="",
        metavar="PATH",
        help=(
            "Generate a conversion summary report (enabled by default). Pass the flag alone to write to "
            "<output>/conversion_report.md, or provide a PATH to override the location. "
            "The report is printed to the console and saved as Markdown. "
            "Use --no-report to disable."
        ),
    )

    parser.add_argument(
        "--no-report",
        dest="report",
        action="store_const",
        const=None,
        help="Disable the conversion summary report.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--training-extensions-dir",
        type=Path,
        default=None,
        help="Path to cloned training_extensions repo (required for getitune models)",
    )

    args = parser.parse_args()

    # Parse library filter
    library_filter: list[str] | None = None
    if args.library:
        library_filter = [lib.strip() for lib in args.library.split(",") if lib.strip()]

    # Check if config file exists
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
        return 1

    # List models and exit
    if args.list:
        list_models(args.config, library_filter=library_filter)
        return 0

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading configuration from: {args.config}")

    try:
        report_path: Path | None = None
        if args.report is not None:
            report_path = Path(args.report) if args.report else args.output / "conversion_report.md"

        # Load dataset registry
        dataset_registry: DatasetRegistry | None = None
        if args.datasets_config.exists():
            try:
                dataset_registry = DatasetRegistry(args.datasets_config)
                logger.info(f"Loaded dataset registry from: {args.datasets_config}")
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Failed to load dataset registry: {e}")
                logger.warning("Quantization will be skipped for all models")
        else:
            logger.warning(f"Dataset configuration not found: {args.datasets_config}")
            logger.warning("Quantization will be skipped for all models")

        # Create converter
        converter = ModelConverter(
            output_dir=args.output,
            cache_dir=args.cache,
            verbose=args.verbose,
            dataset_registry=dataset_registry,
            training_extensions_dir=args.training_extensions_dir,
            report_path=report_path,
        )

        logger.info(f"Output directory: {args.output}")
        logger.info(f"Cache directory: {args.cache}")

        # Process models
        successful, failed = converter.process_config_file(
            config_path=args.config,
            model_filter=args.model,
            library_filter=library_filter,
        )

        # Print summary
        logger.info("=" * 80)
        logger.info("Conversion Summary:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total: {successful + failed}")
        logger.info("=" * 80)

        if report_path is not None:
            results = converter.collect_results()
            print(format_console_table(results))
            logger.info(f"Conversion report written to: {report_path}")

        return 0 if failed == 0 else 1
    except (ValueError, RuntimeError, ImportError, FileNotFoundError) as e:
        logger.error(f"Failed to process model: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
