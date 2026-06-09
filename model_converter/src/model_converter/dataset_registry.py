#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Dataset registry for managing calibration dataset paths."""

import json
import logging
from pathlib import Path
from typing import Any


class DatasetRegistry:
    """Registry for mapping dataset types to local filesystem paths.

    Loads a JSON configuration file that maps dataset type identifiers
    (e.g., "imagenet-1k", "coco-detection") to local directory paths.
    Provides validation and helpful error messages when datasets are missing.

    Example datasets.json format:
        {
            "datasets": {
                "imagenet-1k": "/path/to/imagenet/validation",
                "imagenet-21k": "/path/to/imagenet21k/validation",
                "coco-detection": "/path/to/coco2017/val"
            }
        }
    """

    def __init__(self, config_path: Path):
        """Initialize the DatasetRegistry from a JSON configuration file.

        Args:
            config_path: Path to the datasets configuration JSON file

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the JSON is invalid or missing required structure
        """
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)

        if not self.config_path.exists():
            error_msg = f"Dataset configuration file not found: {self.config_path}"
            raise FileNotFoundError(error_msg)

        try:
            with self.config_path.open() as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in dataset configuration file {self.config_path}: {e}"
            raise ValueError(error_msg) from e
        except (OSError, PermissionError) as e:
            error_msg = f"Failed to read dataset configuration file {self.config_path}: {e}"
            raise ValueError(error_msg) from e

        if not isinstance(config, dict) or "datasets" not in config:
            error_msg = (
                f"Invalid dataset configuration format in {self.config_path}. "
                'Expected JSON with "datasets" key: {"datasets": {...}}'
            )
            raise ValueError(error_msg)

        self._datasets: dict[str, Path] = {}
        datasets = config["datasets"]

        if not isinstance(datasets, dict):
            error_msg = (
                f"Invalid datasets format in {self.config_path}. Expected dictionary mapping dataset types to paths."
            )
            raise ValueError(error_msg)

        # Convert string paths to Path objects
        for dataset_type, path in datasets.items():
            if not isinstance(dataset_type, str) or not isinstance(path, str):
                error_msg = (
                    f"Invalid entry in datasets configuration: {dataset_type} -> {path}. "
                    "Both type and path must be strings."
                )
                raise ValueError(error_msg)
            self._datasets[dataset_type] = Path(path)

        self.logger.info(f"Loaded dataset registry with {len(self._datasets)} dataset types")
        self.logger.debug(f"Available dataset types: {list(self._datasets.keys())}")

    def get_path(self, dataset_type: str, *, validate_exists: bool = False) -> Path:
        """Get the filesystem path for a dataset type.

        Args:
            dataset_type: Dataset type identifier (e.g., "imagenet-1k")
            validate_exists: If True, verify the path exists on the filesystem

        Returns:
            Path to the dataset directory

        Raises:
            ValueError: If the dataset type is not registered
            FileNotFoundError: If validate_exists=True and path doesn't exist
        """
        if dataset_type not in self._datasets:
            available = ", ".join(sorted(self._datasets.keys()))
            error_msg = (
                f"Dataset type '{dataset_type}' not found in registry. "
                f"Available types: {available if available else '(none)'}"
            )
            raise ValueError(error_msg)

        path = self._datasets[dataset_type]

        if validate_exists and not path.exists():
            error_msg = (
                f"Dataset path for type '{dataset_type}' does not exist: {path}. "
                f"Please verify the path in {self.config_path}"
            )
            raise FileNotFoundError(error_msg)

        return path

    def has_type(self, dataset_type: str) -> bool:
        """Check if a dataset type is registered.

        Args:
            dataset_type: Dataset type identifier to check

        Returns:
            True if the type is registered, False otherwise
        """
        return dataset_type in self._datasets

    def list_types(self) -> list[str]:
        """Get a list of all registered dataset types.

        Returns:
            Sorted list of dataset type identifiers
        """
        return sorted(self._datasets.keys())

    def resolve_from_config(self, model_config: dict[str, Any]) -> Path | None:
        """Resolve dataset path from a model configuration.

        Extracts the 'dataset_type' field from the model config and
        returns the corresponding path. Returns None if no dataset_type
        is specified (model doesn't require calibration dataset).

        Args:
            model_config: Model configuration dictionary

        Returns:
            Path to the dataset, or None if no dataset_type specified

        Raises:
            ValueError: If dataset_type is specified but not in registry
        """
        dataset_type = model_config.get("dataset_type")

        if dataset_type is None:
            return None

        if not isinstance(dataset_type, str) or not dataset_type.strip():
            model_name = model_config.get("model_short_name", "unknown")
            error_msg = f"Model '{model_name}' has invalid dataset_type: {dataset_type}"
            raise ValueError(error_msg)

        return self.get_path(dataset_type.strip())
