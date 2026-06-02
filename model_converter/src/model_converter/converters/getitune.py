#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Getitune (training_extensions) model converter."""

import json
import shutil
import subprocess  # nosec B404 — fixed-argv invocation of `uv run`, no shell, no untrusted input
import tempfile
from pathlib import Path
from typing import Any

from model_converter.converters.base import BaseConverter
from model_converter.reporting import AccuracyResults


class GetituneConverter(BaseConverter):
    """Converter for getitune models from training_extensions.

    Invokes the export_pretrained_models.py script as a subprocess to export
    models, then repackages the output and applies INT8 quantization.
    """

    def __init__(self, training_extensions_dir: Path | None = None, **kwargs: Any):
        """Initialize GetituneConverter.

        Args:
            training_extensions_dir: Path to cloned training_extensions repository
            **kwargs: Arguments passed to BaseConverter
        """
        super().__init__(**kwargs)
        self.training_extensions_dir = Path(training_extensions_dir) if training_extensions_dir else None

    def process_model_config(self, config: dict[str, Any]) -> bool:
        """Process a getitune model configuration.

        Runs the export_pretrained_models.py script, repackages the output
        into the standard layout, and optionally quantizes to INT8.

        Args:
            config: Model configuration dictionary

        Returns:
            True if successful, False otherwise
        """
        model_short_name = config.get("model_short_name", "unknown")

        # Check if both FP16 and INT8 models already exist
        fp16_model_path = self.output_dir / f"{model_short_name}-fp16-ov" / f"{model_short_name}.xml"
        int8_model_path = self.output_dir / f"{model_short_name}-int8-ov" / f"{model_short_name}.xml"

        if fp16_model_path.exists() and int8_model_path.exists():
            self.logger.info(f"Skipping {model_short_name}: FP16 and INT8 models already exist")
            self._record_result(self._build_result(config), converted=False, quantized=False, skipped=True)
            return True

        try:
            if not self.training_extensions_dir:
                error_msg = (
                    "training_extensions_dir is required for getitune models. "
                    "Use --training-extensions-dir to specify the path."
                )
                raise ValueError(error_msg)

            if not self.training_extensions_dir.exists():
                error_msg = f"training_extensions directory not found: {self.training_extensions_dir}"
                raise FileNotFoundError(error_msg)

            model_license = config.get("license")
            model_license_link = config.get("license_link")

            if not model_license:
                error_msg = f"Model '{model_short_name}' must define 'license' in configuration"
                raise ValueError(error_msg)
            if not model_license_link:
                error_msg = f"Model '{model_short_name}' must define 'license_link' in configuration"
                raise ValueError(error_msg)

            self.logger.info("=" * 80)
            self.logger.info(f"Processing getitune model: {config.get('model_full_name', model_short_name)}")
            self.logger.info(f"Short name: {model_short_name}")
            if "description" in config:
                self.logger.info(f"Description: {config['description']}")
            self.logger.info("=" * 80)

            # Export model using training_extensions script
            exported_model_path = self._run_export(config)

            # Repackage into standard layout
            self._repackage_model(config, exported_model_path)

            # Quantize if enabled and dataset is available
            accuracy: AccuracyResults | None = None
            quantization_attempted = bool(
                config.get("quantize", True) and self.dataset_path and self.dataset_path.exists(),
            )
            if quantization_attempted:
                accuracy = self._quantize_exported_model(config)

            quantized = accuracy.int8_succeeded if quantization_attempted and accuracy is not None else True
            self._record_result(
                self._build_result(config),
                converted=True,
                quantized=quantized,
                accuracy=accuracy,
            )

            self.logger.info(f"✓ Successfully converted {model_short_name}")
            return True

        except (ValueError, RuntimeError, FileNotFoundError, OSError, subprocess.CalledProcessError) as e:
            self.logger.error(f"✗ Failed to process getitune model {model_short_name}: {e}")
            self._record_result(self._build_result(config), converted=False, quantized=False)
            import traceback

            self.logger.debug(traceback.format_exc())
            return False

    def _run_export(self, config: dict[str, Any]) -> Path:
        """Run the export_pretrained_models.py script.

        Invokes the script within the training_extensions/library uv-managed
        environment so that all ``getitune`` dependencies are available.

        Args:
            config: Model configuration dictionary

        Returns:
            Path to the exported model XML file
        """
        assert self.training_extensions_dir is not None

        getitune_task = config.get("getitune_task")
        getitune_recipe = config.get("getitune_recipe")

        if not getitune_task:
            error_msg = f"Model '{config.get('model_short_name')}' must define 'getitune_task'"
            raise ValueError(error_msg)
        if not getitune_recipe:
            error_msg = f"Model '{config.get('model_short_name')}' must define 'getitune_recipe'"
            raise ValueError(error_msg)

        # Create temporary directory for export output
        temp_dir = tempfile.mkdtemp(prefix="getitune_export_")
        temp_output = Path(temp_dir)

        export_script = self.training_extensions_dir / "export_pretrained_models.py"
        if not export_script.exists():
            error_msg = f"Export script not found: {export_script}"
            raise FileNotFoundError(error_msg)

        library_dir = self.training_extensions_dir / "library"
        library_pyproject = library_dir / "pyproject.toml"
        if not library_pyproject.exists():
            error_msg = (
                f"getitune library project not found at {library_dir}. "
                f"Ensure training_extensions contains a 'library/' subdirectory with a pyproject.toml."
            )
            raise FileNotFoundError(error_msg)

        cmd = [
            "uv",
            "run",
            "--project",
            str(library_dir),
            "--extra",
            "cpu",
            "python",
            str(export_script),
            "--task",
            getitune_task,
            "--model",
            getitune_recipe,
            "--output-dir",
            str(temp_output),
            "--format",
            "OPENVINO",
            "--precision",
            "FP16",
        ]

        self.logger.info(f"Running export command: {' '.join(cmd)}")

        result = subprocess.run(  # noqa: S603  # nosec B603 — cmd is built from validated config, no shell
            cmd,
            cwd=str(self.training_extensions_dir),
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            self.logger.error(f"Export script stderr: {result.stderr}")
            error_msg = f"Export script failed with return code {result.returncode}: {result.stderr}"
            raise RuntimeError(error_msg)

        self.logger.debug(f"Export script stdout: {result.stdout}")

        # Locate the exported model
        # Expected path: temp_output/{task_lower}/{recipe}/exported_model.xml
        expected_path = temp_output / getitune_task.lower() / getitune_recipe / "exported_model.xml"

        if not expected_path.exists():
            # Try searching for any .xml file in the output
            xml_files = list(temp_output.rglob("*.xml"))
            if xml_files:
                expected_path = xml_files[0]
            else:
                error_msg = f"No exported model found in {temp_output}"
                raise FileNotFoundError(error_msg)

        self.logger.info(f"Found exported model: {expected_path}")
        return expected_path

    def _repackage_model(self, config: dict[str, Any], exported_model_path: Path) -> None:
        """Repackage exported model into standard layout.

        Args:
            config: Model configuration dictionary
            exported_model_path: Path to the exported model XML
        """
        model_short_name = config.get("model_short_name", "unknown")

        # Create output folder
        output_folder = self.output_dir / f"{model_short_name}-fp16-ov"
        output_folder.mkdir(parents=True, exist_ok=True)

        # Copy model files (XML + BIN)
        target_xml = output_folder / f"{model_short_name}.xml"
        shutil.copy2(exported_model_path, target_xml)

        bin_path = exported_model_path.with_suffix(".bin")
        if bin_path.exists():
            shutil.copy2(bin_path, output_folder / f"{model_short_name}.bin")

        # Also save an FP32 copy for quantization
        fp32_xml = output_folder / f"{model_short_name}_fp32.xml"
        shutil.copy2(exported_model_path, fp32_xml)
        if bin_path.exists():
            shutil.copy2(bin_path, output_folder / f"{model_short_name}_fp32.bin")

        self.logger.info(f"✓ Model repackaged to: {target_xml}")

        # Extract and save model_info as config.json
        try:
            import openvino as ov

            core = ov.Core()
            model = core.read_model(target_xml)
            model_info = model.get_rt_info(["model_info"]).value
            with (output_folder / "config.json").open("w") as f:
                json.dump(model_info, f, indent=4)
        except (ImportError, RuntimeError, KeyError) as e:
            self.logger.warning(f"Could not extract model_info metadata: {e}")

        # Copy .gitattributes file
        gitattributes_template = Path(__file__).parent.parent / "templates" / ".gitattributes"
        if gitattributes_template.exists():
            shutil.copy2(gitattributes_template, output_folder / ".gitattributes")

        # Copy README
        self.copy_readme(config, output_folder, variant="fp16")

        # Cleanup temp directory
        temp_dir = exported_model_path.parent
        while temp_dir.name != "getitune_export_" and "getitune_export_" not in temp_dir.name:
            temp_dir = temp_dir.parent
            if temp_dir == temp_dir.parent:
                break
        if "getitune_export_" in temp_dir.name:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _read_preprocessing_from_model(
        ov: Any,
        model_path: Path,
    ) -> tuple[list[int], str, str, bool]:
        """Read preprocessing parameters from the model's rt_info metadata.

        Args:
            ov: The openvino module
            model_path: Path to the OpenVINO model XML file

        Returns:
            Tuple of (input_shape, mean_values, scale_values, reverse_input_channels)
        """
        core = ov.Core()
        model = core.read_model(model_path)

        # Get input shape from the model's input layer
        input_shape = list(model.input(0).shape)

        # Read preprocessing params from model_info rt_info
        def _get_rt_str(key: str, default: str) -> str:
            try:
                return model.get_rt_info(["model_info", key]).astype(str)
            except RuntimeError:
                return default

        mean_values = _get_rt_str("mean_values", "0 0 0")
        scale_values = _get_rt_str("scale_values", "1 1 1")
        reverse_input_channels = _get_rt_str("reverse_input_channels", "True").lower() in ("true", "1", "yes")

        return input_shape, mean_values, scale_values, reverse_input_channels

    def _quantize_exported_model(self, config: dict[str, Any]) -> AccuracyResults:
        """Quantize the exported FP16 model to INT8.

        Reads preprocessing parameters (input_shape, mean_values, scale_values,
        reverse_input_channels) from the exported model's rt_info metadata,
        which is embedded by the getitune exporter.

        Args:
            config: Model configuration dictionary

        Returns:
            The accuracies measured during quantization and the INT8 success flag.
        """
        import openvino as ov

        model_short_name = config.get("model_short_name", "unknown")
        accuracy = AccuracyResults()

        # Use FP32 model for quantization (better quality)
        fp32_model_path = self.output_dir / f"{model_short_name}-fp16-ov" / f"{model_short_name}_fp32.xml"

        if not fp32_model_path.exists():
            # Fall back to the FP16 model
            fp32_model_path = self.output_dir / f"{model_short_name}-fp16-ov" / f"{model_short_name}.xml"

        # Extract preprocessing parameters from model rt_info
        input_shape, mean_values, scale_values, reverse_input_channels = self._read_preprocessing_from_model(
            ov,
            fp32_model_path,
        )

        self.logger.info("Creating calibration dataset for INT8 quantization")
        calibration_data, _ = self.create_calibration_dataset(
            input_shape=input_shape,
            mean_values=mean_values,
            scale_values=scale_values,
            reverse_input_channels=reverse_input_channels,
            subset_size=300,
            return_labels=False,
        )

        if calibration_data:
            self.quantize_model(
                model_path=fp32_model_path,
                calibration_data=calibration_data,
                model_config=config,
                preset="mixed",
                accuracy_results=accuracy,
            )

        # Clean up temporary FP32 model after quantization
        try:
            fp32_path = self.output_dir / f"{model_short_name}-fp16-ov" / f"{model_short_name}_fp32.xml"
            if fp32_path.exists():
                fp32_path.unlink()
                self.logger.debug(f"Removed temporary FP32 model: {fp32_path}")
            fp32_bin_path = fp32_path.with_suffix(".bin")
            if fp32_bin_path.exists():
                fp32_bin_path.unlink()
                self.logger.debug(f"Removed temporary FP32 weights: {fp32_bin_path}")
        except OSError as e:
            self.logger.warning(f"Failed to remove temporary FP32 files: {e}")

        return accuracy
