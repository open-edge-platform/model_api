#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Getitune (training_extensions) model converter."""

import shutil
import subprocess  # nosec B404 — fixed-argv invocation of `uv run`, no shell, no untrusted input
import tempfile
from pathlib import Path
from typing import Any, ClassVar

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

        if self._skip_if_already_converted(config, model_short_name):
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

            self._validate_license(config, model_short_name)

            self._log_model_banner(config, model_short_name, label="getitune model")

            # Export model using training_extensions script
            exported_model_path = self._run_export(config)

            # Repackage into standard layout
            self._repackage_model(config, exported_model_path)

            # Quantize if enabled and dataset is available
            accuracy: AccuracyResults | None = None
            quantization_attempted = bool(config.get("quantize", True) and config.get("dataset_type"))
            if quantization_attempted:
                accuracy = self._quantize_exported_model(config)

            return self._finalize_success(
                config,
                model_short_name,
                accuracy=accuracy,
                quantization_attempted=quantization_attempted,
            )

        except (ValueError, RuntimeError, FileNotFoundError, OSError, subprocess.CalledProcessError) as e:
            return self._record_failure(config, model_short_name, e, label="getitune model")

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

        export_script = (self.training_extensions_dir / "export_pretrained_models.py").resolve()
        if not export_script.exists():
            error_msg = f"Export script not found: {export_script}"
            raise FileNotFoundError(error_msg)

        library_dir = (self.training_extensions_dir / "library").resolve()
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

        # Overwrite the exporter's placeholder labels with real class names.
        self._apply_config_labels(config, target_xml, fp32_xml)
        self._apply_preprocessing_overrides(config, target_xml, fp32_xml)

        # Extract and save model_info as config.json
        try:
            import openvino as ov

            core = ov.Core()
            model = core.read_model(target_xml)
            model_info = model.get_rt_info(["model_info"]).value
            self._write_config_json(output_folder, model_info)
        except (ImportError, RuntimeError, KeyError) as e:
            self.logger.warning(f"Could not extract model_info metadata: {e}")

        # Copy .gitattributes file
        self._copy_gitattributes(output_folder)

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

    def _apply_config_labels(self, config: dict[str, Any], *model_paths: Path) -> None:
        """Overwrite ``model_info/labels`` rt_info with configured class names.

        The getitune exporter embeds only numeric placeholder ids and no
        human-readable labels.  When the configuration defines a ``labels`` set
        (e.g. ``IMAGENET1K_V1``), resolve it to the real class names and rewrite
        the rt_info of each given OpenVINO model in place so the repackaged
        ``config.json`` (and any model quantized from these files) carries the
        correct labels.

        Args:
            config: Model configuration dictionary.
            *model_paths: OpenVINO model XML files to update in place.
        """
        labels_config = config.get("labels")
        if not labels_config:
            return

        labels = self.get_labels(labels_config)
        if not labels:
            self.logger.warning(f"Could not load labels for: {labels_config}")
            return

        try:
            import openvino as ov

            core = ov.Core()
            for model_path in model_paths:
                if not model_path.exists():
                    continue
                model = core.read_model(model_path)
                model.set_rt_info(labels, ["model_info", "labels"])
                # Save to a temporary path first: OpenVINO memory-maps the
                # source weights file, so writing back to the same path would
                # truncate the file while it is still being read, corrupting
                # the model (and crashing the process).
                tmp_xml = model_path.with_name(f"{model_path.stem}_labeled_tmp.xml")
                ov.save_model(model, tmp_xml, compress_to_fp16=not model_path.stem.endswith("_fp32"))
                tmp_xml.replace(model_path)
                tmp_xml.with_suffix(".bin").replace(model_path.with_suffix(".bin"))
            self.logger.info(f"✓ Applied {labels_config} labels to exported model(s)")
        except (ImportError, RuntimeError) as e:
            self.logger.warning(f"Could not apply labels to exported model: {e}")

    def _apply_preprocessing_overrides(self, config: dict[str, Any], *model_paths: Path) -> None:
        """Override ``model_info`` preprocessing rt_info with configured values.

        Args:
            config: Model configuration dictionary.
            *model_paths: OpenVINO model XML files to update in place.
        """
        overrides = config.get("preprocessing_overrides")
        if not overrides:
            return

        try:
            import openvino as ov

            core = ov.Core()
            for model_path in model_paths:
                if not model_path.exists():
                    continue
                model = core.read_model(model_path)
                for key, value in overrides.items():
                    model.set_rt_info(str(value), ["model_info", key])

                tmp_xml = model_path.with_name(f"{model_path.stem}_preproc_tmp.xml")
                ov.save_model(model, tmp_xml, compress_to_fp16=not model_path.stem.endswith("_fp32"))
                tmp_xml.replace(model_path)
                tmp_xml.with_suffix(".bin").replace(model_path.with_suffix(".bin"))
            self.logger.info(f"✓ Applied preprocessing overrides to exported model(s): {overrides}")
        except (ImportError, RuntimeError) as e:
            self.logger.warning(f"Could not apply preprocessing overrides to exported model: {e}")

    # Maps the exported model's ``input_dtype`` to the divisor that maps its raw
    # integer range to [0, 1]. Mirrors Model API's intensity inference so the
    # converter normalizes calibration/validation tensors the same way the model
    # does at inference time.
    _DTYPE_MAX_VALUE: ClassVar[dict[str, float]] = {"u8": 255.0, "u16": 65535.0, "i16": 32767.0, "f32": 1.0}

    @staticmethod
    def _read_preprocessing_from_model(
        ov: Any,
        model_path: Path,
    ) -> tuple[list[int], str, str, bool, float]:
        """Read preprocessing parameters from the model's rt_info metadata.

        Args:
            ov: The openvino module
            model_path: Path to the OpenVINO model XML file

        Returns:
            Tuple of (input_shape, mean_values, scale_values, reverse_input_channels,
            intensity_scale). ``intensity_scale`` is the divisor applied to raw
            pixels before mean/scale (e.g. ``255`` for a ``scale_to_unit`` ``u8``
            model) and ``1.0`` when the model performs no intensity scaling.
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

        # Reproduce the model's intensity step. getitune models normalize with a
        # ``scale_to_unit`` step (÷255 for u8) before applying [0, 1]-range
        # mean/scale; without it the calibration/validation tensors are ~255x too
        # large and accuracy collapses to ~0%.
        intensity_scale = 1.0
        if _get_rt_str("intensity_mode", "") == "scale_to_unit":
            max_value = _get_rt_str("intensity_max_value", "")
            if max_value:
                intensity_scale = float(max_value)
            else:
                intensity_scale = GetituneConverter._DTYPE_MAX_VALUE.get(_get_rt_str("input_dtype", ""), 1.0)

        return input_shape, mean_values, scale_values, reverse_input_channels, intensity_scale

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
        (
            input_shape,
            mean_values,
            scale_values,
            reverse_input_channels,
            intensity_scale,
        ) = self._read_preprocessing_from_model(
            ov,
            fp32_model_path,
        )

        # Pick a metric strategy for this dataset/model_type combo. Top-1
        # uses the preprocessed-tensor classification path; other metrics
        # (multilabel mAP, COCO mAP, mIoU) flow through Model API via
        # :meth:`_measure_metric` with raw image samples.
        dataset_path = self._resolve_dataset_path(config)
        metric, is_top1 = self._select_accuracy_metric(config, dataset_path, accuracy)

        self.logger.info("Creating calibration dataset for INT8 quantization")
        if is_top1:
            self.logger.info("Creating validation dataset for accuracy measurement")

        calibration_data, validation_labels = self.create_calibration_dataset(
            input_shape=input_shape,
            mean_values=mean_values,
            scale_values=scale_values,
            reverse_input_channels=reverse_input_channels,
            subset_size=500,
            return_labels=is_top1,
            dataset_path=dataset_path,
            dataset_type=config.get("dataset_type"),
            intensity_scale=intensity_scale,
        )

        validation_samples = self._collect_metric_validation_samples(
            metric,
            is_top1,
            dataset_path,
            config.get("dataset_type"),
        )

        if calibration_data:
            self.quantize_model(
                model_path=fp32_model_path,
                calibration_data=calibration_data,
                model_config=config,
                preset="mixed",
                validation_data=calibration_data if validation_labels else None,
                validation_labels=validation_labels or None,
                validation_samples=validation_samples,
                metric=metric,
                accuracy_results=accuracy,
            )

        # Clean up temporary FP32 model after quantization
        fp32_path = self.output_dir / f"{model_short_name}-fp16-ov" / f"{model_short_name}_fp32.xml"
        self._cleanup_fp32(fp32_path)

        return accuracy
