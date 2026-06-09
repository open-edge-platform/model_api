#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Timm (HuggingFace) model converter."""

from typing import TYPE_CHECKING, Any

import torch.nn as nn

from model_converter.converters.pytorch import PyTorchConverter

if TYPE_CHECKING:
    from model_converter.reporting import AccuracyResults


class TimmConverter(PyTorchConverter):
    """Converter for timm models hosted on HuggingFace Hub.

    Loads models via timm/transformers from HuggingFace,
    exports to OpenVINO, and optionally quantizes to INT8.
    """

    def load_huggingface_model(
        self,
        repo_id: str,
        revision: str,
        model_library: str = "timm",
        model_params: dict[str, Any] | None = None,
    ) -> nn.Module:
        """Load a model from Hugging Face Hub.

        Args:
            repo_id: Hugging Face repository ID
            revision: Immutable revision/commit SHA for the Hugging Face repository
            model_library: Library to use ('timm', 'transformers', etc.)
            model_params: Optional parameters for model loading

        Returns:
            Loaded model instance
        """
        try:
            if model_library == "timm":
                import timm

                repo_ref = f"hf-hub:{repo_id}@{revision}"
                self.logger.info(f"Loading timm model: {repo_ref}")
                model = timm.create_model(
                    repo_ref,
                    pretrained=True,
                    cache_dir=self.cache_dir,
                    **(model_params or {}),
                )
            elif model_library == "transformers":
                from transformers import AutoModel

                self.logger.info(f"Loading transformers model: {repo_id}@{revision}")
                model = AutoModel.from_pretrained(
                    repo_id,
                    revision=revision,
                    cache_dir=self.cache_dir,
                    **(model_params or {}),
                )
            else:
                error_msg = f"Unsupported model library: {model_library}"
                raise ValueError(error_msg)

            model.eval()
            self.logger.info("✓ Hugging Face model loaded successfully")
            return model

        except Exception as e:
            self.logger.error(f"Failed to load Hugging Face model: {e}")
            raise

    def _apply_timm_data_config(self, model: nn.Module, config: dict[str, Any]) -> None:
        """Override preprocessing config with timm's canonical values.

        Reads the model's ``pretrained_cfg`` via :func:`timm.data.resolve_data_config`
        and updates ``mean_values``, ``scale_values`` and ``input_shape`` in
        ``config`` in place. timm stores ``mean``/``std`` as 0..1 floats, so they
        are scaled to the 0..255 pixel range used by Model API metadata.

        ``reverse_input_channels`` is forced to ``True`` because timm models are
        trained on RGB images while images are decoded as BGR by OpenCV, so the
        channels must always be swapped. timm does not expose this through
        ``resolve_data_config``, but it is an invariant for these models.

        Args:
            model: The loaded timm model.
            config: Model configuration dictionary, mutated in place.
        """
        # timm models consume RGB; OpenCV decodes BGR, so the swap is always needed.
        self._override_config_value(config, "reverse_input_channels", resolved=True)

        try:
            from timm.data import resolve_data_config
        except ImportError:
            self.logger.warning("timm not available; keeping configured preprocessing values")
            return

        try:
            data_config = resolve_data_config({}, model=model)
        except (RuntimeError, ValueError, KeyError, TypeError) as e:
            self.logger.warning(f"Could not resolve timm data config, keeping configured values: {e}")
            return

        mean = data_config.get("mean")
        std = data_config.get("std")
        input_size = data_config.get("input_size")

        if mean is not None:
            resolved_mean = " ".join(f"{value * 255:g}" for value in mean)
            self._override_config_value(config, "mean_values", resolved_mean)
        if std is not None:
            resolved_scale = " ".join(f"{value * 255:g}" for value in std)
            self._override_config_value(config, "scale_values", resolved_scale)
        if input_size is not None and len(input_size) == 3:
            channels, height, width = input_size
            resolved_shape = [1, int(channels), int(height), int(width)]
            self._override_config_value(config, "input_shape", resolved_shape)

    def _override_config_value(self, config: dict[str, Any], key: str, resolved: Any) -> None:
        """Set ``config[key]`` to ``resolved``, logging when it changes a value."""
        existing = config.get(key)
        if existing is not None and existing != resolved:
            self.logger.info(f"Overriding {key}: config={existing!r} -> timm={resolved!r}")
        config[key] = resolved

    def process_model_config(self, config: dict[str, Any]) -> bool:
        """Process a timm/HuggingFace model configuration.

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
            model_license = config.get("license")
            model_license_link = config.get("license_link")

            if not model_license:
                error_msg = f"Model '{model_short_name}' must define 'license' in configuration"
                raise ValueError(error_msg)
            if not model_license_link:
                error_msg = f"Model '{model_short_name}' must define 'license_link' in configuration"
                raise ValueError(error_msg)

            self.logger.info("=" * 80)
            self.logger.info(f"Processing model: {config.get('model_full_name', model_short_name)}")
            self.logger.info(f"Short name: {model_short_name}")
            if "description" in config:
                self.logger.info(f"Description: {config['description']}")
            self.logger.info("=" * 80)

            # Load model from HuggingFace
            huggingface_repo = config.get("huggingface_repo")
            huggingface_revision = config.get("huggingface_revision")

            if not huggingface_repo:
                error_msg = f"Timm model '{model_short_name}' must define 'huggingface_repo'"
                raise ValueError(error_msg)
            if not huggingface_revision:
                error_msg = "Hugging Face models must define 'huggingface_revision' with an immutable commit SHA"
                raise ValueError(error_msg)

            model_library = config.get("model_library", "timm")
            model_params = config.get("model_params")
            model = self.load_huggingface_model(
                repo_id=huggingface_repo,
                revision=huggingface_revision,
                model_library=model_library,
                model_params=model_params,
            )

            # Override preprocessing parameters with the values timm ships for
            # this specific checkpoint.  Hand-maintained config values are easy
            # to get wrong (e.g. ImageNet vs. inception normalization), which
            # silently destroys accuracy, so timm's pretrained_cfg is treated as
            # the source of truth.
            if model_library == "timm":
                self._apply_timm_data_config(model, config)

            # Prepare export parameters
            input_shape = config.get("input_shape", [1, 3, 224, 224])
            input_names = config.get("input_names", ["input"])
            output_names = config.get("output_names", ["result"])
            reverse_input_channels = config.get("reverse_input_channels", True)
            mean_values = config.get("mean_values", "123.675 116.28 103.53")
            scale_values = config.get("scale_values", "58.395 57.12 57.375")
            model_type = config.get("model_type", "")

            metadata = self._build_metadata(config)

            output_path = self.output_dir / model_short_name
            fp16_model_path, fp32_model_path = self.export_to_openvino(
                model=model,
                input_shape=input_shape,
                output_path=output_path,
                model_config=config,
                input_names=input_names,
                output_names=output_names,
                metadata=metadata,
            )

            # Quantize the model if dataset is available
            accuracy: AccuracyResults | None = None
            quantization_attempted = bool(config.get("quantize", True) and config.get("dataset_type"))
            if quantization_attempted:
                accuracy = self._quantize_and_cleanup(
                    config,
                    fp32_model_path,
                    model_type=model_type,
                    input_shape=input_shape,
                    mean_values=mean_values,
                    scale_values=scale_values,
                    reverse_input_channels=reverse_input_channels,
                    torch_model=model,
                )

            quantized = accuracy.int8_succeeded if quantization_attempted and accuracy is not None else True
            self._record_result(
                self._build_result(config),
                converted=True,
                quantized=quantized,
                accuracy=accuracy,
            )

            self.logger.info(f"✓ Successfully converted {model_short_name}")
            return True

        except (ValueError, RuntimeError, ImportError, FileNotFoundError) as e:
            self.logger.error(f"✗ Failed to process model {model_short_name}: {e}")
            self._record_result(self._build_result(config), converted=False, quantized=False)
            import traceback

            self.logger.debug(traceback.format_exc())
            return False
