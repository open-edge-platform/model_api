#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Torchvision model converter."""

from typing import TYPE_CHECKING, Any

from model_converter.converters.pytorch import PyTorchConverter
from model_converter.downloaders import URLDownloader

if TYPE_CHECKING:
    from model_converter.reporting import AccuracyResults


class TorchvisionConverter(PyTorchConverter):
    """Converter for torchvision models.

    Downloads weights from URL, loads the model class dynamically,
    exports to OpenVINO, and optionally quantizes to INT8.
    """

    def __init__(self, **kwargs: Any):
        """Initialize TorchvisionConverter."""
        super().__init__(**kwargs)
        self._url_downloader = URLDownloader(cache_dir=self.cache_dir)

    def process_model_config(self, config: dict[str, Any]) -> bool:
        """Process a torchvision model configuration.

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

            # Download weights and load model
            weights_url = config["weights_url"]
            weights_path = self._url_downloader.download(url=weights_url)

            model_class_name = config.get("model_class_name", "torch.nn.Module")
            model_class = self.load_model_class(model_class_name)

            checkpoint = self.load_checkpoint(weights_path)

            model_params = config.get("model_params")
            model = self.create_model(model_class, checkpoint, model_params)

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
            quantization_attempted = bool(
                config.get("quantize", True) and self.dataset_path and self.dataset_path.exists(),
            )
            if quantization_attempted:
                accuracy = self._quantize_and_cleanup(
                    config,
                    fp32_model_path,
                    model_type=model_type,
                    input_shape=input_shape,
                    mean_values=mean_values,
                    scale_values=scale_values,
                    reverse_input_channels=reverse_input_channels,
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
