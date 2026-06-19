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

        if self._skip_if_already_converted(config, model_short_name):
            return True

        try:
            self._validate_license(config, model_short_name)

            self._log_model_banner(config, model_short_name)

            # Download weights and load model
            weights_url = config["weights_url"]
            weights_path = self._url_downloader.download(url=weights_url)

            model_class_name = config.get("model_class_name", "torch.nn.Module")
            model_class = self.load_model_class(model_class_name)

            checkpoint = self.load_checkpoint(weights_path)

            model_params = config.get("model_params")
            model = self.create_model(model_class, checkpoint, model_params)

            # Prepare export parameters
            params = self._extract_export_params(config)

            metadata = self._build_metadata(config)

            output_path = self.output_dir / model_short_name
            _fp16_model_path, fp32_model_path = self.export_to_openvino(
                model=model,
                input_shape=params.input_shape,
                output_path=output_path,
                model_config=config,
                input_names=params.input_names,
                output_names=params.output_names,
                metadata=metadata,
            )

            # Quantize the model if dataset is available
            accuracy: AccuracyResults | None = None
            quantization_attempted = bool(config.get("quantize", True) and config.get("dataset_type"))
            if quantization_attempted:
                accuracy = self._quantize_and_cleanup(
                    config,
                    fp32_model_path,
                    model_type=params.model_type,
                    input_shape=params.input_shape,
                    mean_values=params.mean_values,
                    scale_values=params.scale_values,
                    reverse_input_channels=params.reverse_input_channels,
                    torch_model=model,
                )

            return self._finalize_success(
                config,
                model_short_name,
                accuracy=accuracy,
                quantization_attempted=quantization_attempted,
            )

        except (ValueError, RuntimeError, ImportError, FileNotFoundError) as e:
            return self._record_failure(config, model_short_name, e)
