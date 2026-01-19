#!/usr/bin/env -S uv run --script
#
# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
PyTorch to OpenVINO Model Converter

Usage:
    uv run python model_converter.py config.json -o ./output_models

"""

import argparse
import importlib
import json
import logging
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download


class ModelConverter:
    """Handles conversion of PyTorch models to OpenVINO format."""

    def __init__(
        self,
        output_dir: Path,
        cache_dir: Path,
        verbose: bool = False,
        dataset_path: Optional[Path] = None,
    ):
        """
        Initialize the ModelConverter.

        Args:
            output_dir: Directory to save converted models
            cache_dir: Directory to cache downloaded weights
            verbose: Enable verbose logging
            dataset_path: Path to calibration dataset for quantization
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def get_labels(self, label_set: str) -> Optional[str]:
        """
        Get label list for a given label set.

        Args:
            label_set: Name of the label set (e.g., "IMAGENET1K_V1")

        Returns:
            Space-separated string of labels, or None if not found
        """
        if label_set == "IMAGENET1K_V1":
            from torchvision.models._meta import _IMAGENET_CATEGORIES

            categories = _IMAGENET_CATEGORIES
            categories = [label.replace(" ", "_") for label in categories]
            return " ".join(categories)

        return None

    def download_from_huggingface(
        self,
        repo_id: str,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Download model from Hugging Face Hub with caching.

        Args:
            repo_id: Hugging Face repository ID (e.g., 'timm/mobilenetv2_050.lamb_in1k')
            filename: Optional specific file to download (if None, downloads the whole repo)

        Returns:
            Path to the downloaded model file or directory
        """
        self.logger.info(f"Downloading from Hugging Face Hub: {repo_id}")

        try:
            if filename:
                # Download a specific file
                cached_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=self.cache_dir,
                )
                self.logger.info(f"✓ Downloaded file: {cached_file}")
                return Path(cached_file)
            else:
                # Download the entire repository
                cached_dir = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=self.cache_dir,
                )
                self.logger.info(f"✓ Downloaded repository to: {cached_dir}")
                return Path(cached_dir)
        except Exception as e:
            self.logger.error(f"Failed to download from Hugging Face: {e}")
            raise

    def download_weights(
        self,
        url: str,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Download model weights from URL with caching.

        Args:
            url: URL to download weights from
            filename: Optional filename to save as (default: extract from URL)

        Returns:
            Path to the downloaded/cached weights file
        """
        if filename is None:
            filename = url.split("/")[-1]

        cached_file = self.cache_dir / filename

        if cached_file.exists():
            self.logger.info(f"Using cached weights: {cached_file}")
            return cached_file

        self.logger.info(f"Downloading weights from: {url}")
        self.logger.info(f"Saving to: {cached_file}")

        try:
            urllib.request.urlretrieve(  # noqa: S310  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
                url,
                cached_file,
            )
            self.logger.info("✓ Download complete")
            return cached_file
        except Exception as e:
            self.logger.error(f"Failed to download weights: {e}")
            raise

    def load_model_class(
        self,
        class_path: str,
    ) -> type:
        """
        Dynamically load a model class from a Python path.

        Args:
            class_path: Full Python path to the class (e.g., 'torchvision.models.resnet.resnet50')

        Returns:
            The model class
        """
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            self.logger.debug(f"Importing module: {module_path}")
            # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
            module = importlib.import_module(
                module_path,
            )
            model_class = getattr(module, class_name)
            self.logger.debug(f"Loaded class: {class_name}")
            return model_class
        except Exception as e:
            self.logger.error(f"Failed to import module {module_path}: {e}")
            raise

    def load_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> Dict[str, Any]:
        """
        Load PyTorch checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint dictionary
        """
        try:
            checkpoint = torch.load(  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.logger.debug(f"Loaded checkpoint from: {checkpoint_path}")
            return checkpoint
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise

    def load_huggingface_model(
        self,
        repo_id: str,
        model_library: str = "timm",
        model_params: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """
        Load a model from Hugging Face Hub.

        Args:
            repo_id: Hugging Face repository ID
            model_library: Library to use ('timm', 'transformers', etc.)
            model_params: Optional parameters for model loading

        Returns:
            Loaded model instance
        """
        try:
            if model_library == "timm":
                import timm
                self.logger.info(f"Loading timm model: {repo_id}")
                model = timm.create_model(
                    repo_id,
                    pretrained=True,
                    **(model_params or {}),
                )
            elif model_library == "transformers":
                from transformers import AutoModel
                self.logger.info(f"Loading transformers model: {repo_id}")
                model = AutoModel.from_pretrained(
                    repo_id,
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

    def create_model(
        self,
        model_class: type,
        checkpoint: Dict[str, Any],
        model_params: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """
        Create and initialize model instance.

        Args:
            model_class: Model class to instantiate
            checkpoint: Checkpoint containing model weights
            model_params: Optional parameters for model initialization

        Returns:
            Initialized model instance
        """
        try:
            # Handle torch.nn.Module case (checkpoint contains full model)
            if model_class == torch.nn.Module:
                if "model" in checkpoint:
                    model = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    # Cannot reconstruct architecture from state_dict alone
                    error_msg = (
                        "Checkpoint contains only state_dict. "
                        "Please specify the model class instead of torch.nn.Module"
                    )
                    raise ValueError(error_msg)
                else:
                    # Assume checkpoint is the model itself
                    model = checkpoint

                if not isinstance(model, nn.Module):
                    error_msg = "Checkpoint does not contain a valid model"
                    raise ValueError(error_msg)
            else:
                # Instantiate model class
                model = model_class(**model_params) if model_params else model_class()

                # Load weights
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    if isinstance(checkpoint["model"], nn.Module):
                        return checkpoint["model"]
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint

                model.load_state_dict(state_dict, strict=False)

            model.eval()
            self.logger.info("✓ Model created and loaded successfully")
            return model

        except Exception as e:
            self.logger.error(f"Failed to create model: {e}")
            raise

    def copy_readme(
        self,
        model_name: str,
        output_folder: Path,
        variant: str = "fp16",
        model_library: str = "timm",
    ) -> None:
        """
        Copy README template to model folder and replace placeholders.

        Args:
            model_name: Name of the model
            output_folder: Folder where the model is saved
            variant: Model variant ('fp16' or 'int8')
            model_library: Model library ('timm' or 'torchvision')
        """
        try:
            # Determine which README template to use based on model library
            template_name = f"README-{model_library}-{variant}.md"
            template_path = Path(__file__).parent / "templates" / template_name
            
            if not template_path.exists():
                self.logger.warning(f"README template not found: {template_path}")
                return
            
            # Read template
            readme_content = template_path.read_text()
            
            # Replace {model_name} placeholder
            readme_content = readme_content.replace("{model_name}", model_name)
            
            # Write to model folder
            output_readme = output_folder / "README.md"
            output_readme.write_text(readme_content)
            self.logger.debug(f"Copied README to: {output_readme}")
            
        except Exception as e:
            self.logger.warning(f"Failed to copy README: {e}")

    def create_calibration_dataset(
        self,
        input_shape: List[int],
        mean_values: Optional[str] = None,
        scale_values: Optional[str] = None,
        reverse_input_channels: bool = True,
        subset_size: int = 5000,
        return_labels: bool = False,
    ) -> tuple[List[np.ndarray], List[int]] | List[np.ndarray]:
        """
        Create calibration dataset from ImageNet validation images.

        Args:
            input_shape: Target input shape [batch, channels, height, width]
            mean_values: Space-separated mean values for normalization
            scale_values: Space-separated scale values for normalization
            reverse_input_channels: Whether to reverse RGB to BGR
            subset_size: Number of images to use for calibration
            return_labels: Whether to return labels along with images

        Returns:
            List of preprocessed image arrays, or tuple of (images, labels) if return_labels=True
        """
        if not self.dataset_path or not self.dataset_path.exists():
            self.logger.warning("Dataset path not provided or doesn't exist. Skipping quantization.")
            return []

        # Parse mean and scale values
        mean = np.array([float(x) for x in mean_values.split()]) if mean_values else np.array([0, 0, 0])
        scale = np.array([float(x) for x in scale_values.split()]) if scale_values else np.array([1, 1, 1])

        _, _, height, width = input_shape
        calibration_data = []
        labels = [] if return_labels else None

        # Find all images in the dataset
        image_dir = self.dataset_path / "ILSVRC2012_img_val_subset"
        if not image_dir.exists():
            self.logger.error(f"Image directory not found: {image_dir}")
            return ([], []) if return_labels else []

        image_files = []
        image_labels = [] if return_labels else None
        for class_dir in sorted(image_dir.iterdir()):
            if class_dir.is_dir():
                class_label = int(class_dir.name)  # Directory name is the class ID
                for pattern in ["*.JPEG", "*.jpg", "*.png"]:
                    for img_path in class_dir.glob(pattern):
                        image_files.append(img_path)
                        if return_labels:
                            image_labels.append(class_label)

        if not image_files:
            self.logger.error("No images found in dataset")
            return ([], []) if return_labels else []

        self.logger.info(f"Found {len(image_files)} images in dataset")
        self.logger.info(f"Using {min(subset_size, len(image_files))} images for calibration")

        # Process images
        for i in range(min(subset_size, len(image_files))):
            img_path = image_files[i]
            try:
                # Read and resize image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Resize to target size
                img = cv2.resize(img, (width, height))

                # Convert to float32
                img = img.astype(np.float32)

                # Reverse channels if needed (BGR to RGB)
                if reverse_input_channels:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Normalize: (img - mean) / scale
                img = (img - mean) / scale

                # Transpose to NCHW format
                img = img.transpose(2, 0, 1)

                # Add batch dimension
                img = np.expand_dims(img, axis=0)

                calibration_data.append(img)
                if return_labels:
                    labels.append(image_labels[i])

                if (i + 1) % 50 == 0:
                    self.logger.debug(f"Processed {i + 1}/{subset_size} images")

            except Exception as e:
                self.logger.warning(f"Failed to process {img_path}: {e}")
                continue

        self.logger.info(f"✓ Created calibration dataset with {len(calibration_data)} images")
        return (calibration_data, labels) if return_labels else calibration_data

    def validate_model(
        self,
        model_path: Path,
        validation_data: List[np.ndarray],
        labels: List[int],
    ) -> float:
        """
        Validate OpenVINO model and compute top-1 accuracy.

        Args:
            model_path: Path to the OpenVINO model (.xml)
            validation_data: List of validation images
            labels: List of ground truth labels

        Returns:
            Top-1 accuracy (0.0 to 1.0)
        """
        try:
            import openvino as ov

            core = ov.Core()
            model = core.read_model(model_path)
            compiled_model = core.compile_model(model, device_name="CPU")
            output_layer = compiled_model.outputs[0]

            predictions = []
            for img in validation_data:
                result = compiled_model(img)[output_layer]
                pred_class = np.argmax(result, axis=1)[0]
                predictions.append(pred_class)

            # Compute accuracy
            correct = sum(p == l for p, l in zip(predictions, labels))
            accuracy = correct / len(labels)

            return accuracy

        except Exception as e:
            self.logger.error(f"Failed to validate model: {e}")
            return 0.0

    def quantize_model(
        self,
        model_path: Path,
        calibration_data: List[np.ndarray],
        preset: str = "accuracy",
        model_library: str = "timm",
        validation_data: Optional[List[np.ndarray]] = None,
        validation_labels: Optional[List[int]] = None,
    ) -> Path:
        """
        Quantize OpenVINO model to INT8 using NNCF.

        Args:
            model_path: Path to the FP32 OpenVINO model (.xml)
            calibration_data: List of calibration images
            preset: Quantization preset ('accuracy', 'performance', 'mixed')
            model_library: Model library ('timm' or 'torchvision')
            validation_data: Optional validation images for accuracy measurement
            validation_labels: Optional validation labels for accuracy measurement

        Returns:
            Path to the quantized model
        """
        if not calibration_data:
            self.logger.warning("No calibration data provided. Skipping quantization.")
            return model_path

        try:
            import nncf
            import openvino as ov

            self.logger.info(f"Quantizing model with {len(calibration_data)} calibration samples")
            self.logger.info(f"Using preset: {preset}")

            # Load the model
            core = ov.Core()
            model = core.read_model(model_path)

            # Create calibration dataset generator
            def calibration_dataset():
                for data in calibration_data:
                    yield data

            # Map preset string to NNCF enum
            preset_map = {
                "performance": nncf.QuantizationPreset.PERFORMANCE,
                "mixed": nncf.QuantizationPreset.MIXED,
            }
            nncf_preset = preset_map.get(preset.lower(), nncf.QuantizationPreset.MIXED)
            
            # Quantize the model
            quantized_model = nncf.quantize(
                model,
                calibration_dataset=nncf.Dataset(calibration_dataset()),
                preset=nncf_preset,
                subset_size=len(calibration_data),
            )

            # Extract model name from the FP32 model path
            # The FP32 path is like: output_dir/model_name-fp16-ov/model_name_fp32.xml
            model_name = model_path.stem  # Gets model_name_fp32 from model_name_fp32.xml
            # Remove _fp32 suffix if present
            if model_name.endswith("_fp32"):
                model_name = model_name[:-5]
            
            # Create output folder with -int8-ov suffix
            output_folder = model_path.parent.parent / f"{model_name}-int8-ov"
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Save quantized model with model name inside the folder
            output_path = output_folder / f"{model_name}.xml"
            ov.save_model(quantized_model, output_path, True)
            self.logger.info(f"✓ Quantized model saved: {output_path}")
            
            # Validate accuracy if validation data provided
            if validation_data and validation_labels:
                self.logger.info("Validating FP32 model accuracy...")
                fp32_accuracy = self.validate_model(model_path, validation_data, validation_labels)
                self.logger.info(f"FP32 Top-1 Accuracy: {fp32_accuracy * 100:.2f}%")
                
                self.logger.info("Validating INT8 model accuracy...")
                int8_accuracy = self.validate_model(output_path, validation_data, validation_labels)
                self.logger.info(f"INT8 Top-1 Accuracy: {int8_accuracy * 100:.2f}%")
                
                accuracy_drop = (fp32_accuracy - int8_accuracy) * 100
                self.logger.info(f"Accuracy Drop: {accuracy_drop:.2f}%")
            
            # Copy .gitattributes file
            gitattributes_template = Path(__file__).parent / "templates" / ".gitattributes"
            if gitattributes_template.exists():
                shutil.copy2(gitattributes_template, output_folder / ".gitattributes")
                self.logger.debug(f"Copied .gitattributes to: {output_folder}")
            
            # Copy README for INT8 model
            self.copy_readme(model_name, output_folder, variant="int8", model_library=model_library)

            return output_path

        except ImportError:
            self.logger.error("NNCF not installed. Install with: pip install nncf")
            return model_path
        except Exception as e:
            self.logger.error(f"Failed to quantize model: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return model_path

    def export_to_openvino(
        self,
        model: nn.Module,
        input_shape: List[int],
        output_path: Path,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        metadata: Optional[Dict[tuple, str]] = None,
        model_library: str = "timm",
    ) -> tuple[Path, Path]:
        """
        Export PyTorch model to OpenVINO format.

        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape [batch, channels, height, width]
            output_path: Path to save the model (without extension)
            input_names: Names for input tensors
            output_names: Names for output tensors
            metadata: Metadata to embed in the model
            model_library: Model library ('timm' or 'torchvision')

        Returns:
            Tuple of (fp16_model_path, fp32_model_path) - FP16 for final use, FP32 for quantization
        """
        import openvino as ov

        try:
            model.eval()
            dummy_input = torch.randn(*input_shape)
            self.logger.info("Direct PyTorch to OpenVINO conversion")
            ov_model = ov.convert_model(model, example_input=dummy_input)
            self.logger.info("✓ PyTorch to OpenVINO conversion complete")

            # Reshape model to fixed input shape (remove dynamic dimensions)
            first_input = ov_model.input(0)
            input_name_for_reshape = next(iter(first_input.get_names())) if first_input.get_names() else 0

            self.logger.debug(f"Setting fixed input shape: {input_shape}")
            ov_model.reshape({input_name_for_reshape: input_shape})

            # Post-process the model
            ov_model = self._postprocess_openvino_model(
                ov_model,
                input_names=input_names,
                output_names=output_names,
                metadata=metadata,
            )

            # Create output folder with -fp16-ov suffix
            model_name = output_path.name
            output_folder = output_path.parent / f"{model_name}-fp16-ov"
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Save FP32 model for quantization (temporary)
            fp32_xml_path = output_folder / f"{model_name}_fp32.xml"
            ov.save_model(ov_model, fp32_xml_path, compress_to_fp16=False)
            self.logger.debug(f"Saved FP32 model for quantization: {fp32_xml_path}")
            
            # Save the FP16 model (final)
            xml_path = output_folder / f"{model_name}.xml"
            ov.save_model(ov_model, xml_path, compress_to_fp16=True)
            self.logger.info(f"✓ Model saved: {xml_path}")
            
            # Copy .gitattributes file
            gitattributes_template = Path(__file__).parent / "templates" / ".gitattributes"
            if gitattributes_template.exists():
                shutil.copy2(gitattributes_template, output_folder / ".gitattributes")
                self.logger.debug(f"Copied .gitattributes to: {output_folder}")
            
            # Copy README for FP16 model
            self.copy_readme(model_name, output_folder, variant="fp16", model_library=model_library)

            return xml_path, fp32_xml_path

        except Exception as e:
            self.logger.error(f"Failed to export model: {e}")
            raise

    def _postprocess_openvino_model(
        self,
        model: Any,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        metadata: Optional[Dict[tuple, str]] = None,
    ) -> Any:
        """
        Post-process OpenVINO model (set names, add metadata).

        Args:
            model: OpenVINO model
            input_names: Names for input tensors
            output_names: Names for output tensors
            metadata: Metadata to embed

        Returns:
            Post-processed model
        """
        # Set input names
        if input_names:
            for i, name in enumerate(input_names):
                if i < len(model.inputs):
                    model.input(i).set_names({name})
                    self.logger.debug(f"Set input {i} name to: {name}")

        # Set output names
        if output_names:
            for i, name in enumerate(output_names):
                if i < len(model.outputs):
                    model.output(i).set_names({name})
                    self.logger.debug(f"Set output {i} name to: {name}")

        # Add metadata
        if metadata:
            for key, value in metadata.items():
                model.set_rt_info(value, list(key))
                self.logger.debug(f"Set metadata {key}: {value}")

        return model

    def process_model_config(self, config: Dict[str, Any]) -> bool:
        """
        Process a single model configuration.

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
            return True

        try:
            self.logger.info("=" * 80)
            self.logger.info(f"Processing model: {config.get('model_full_name', model_short_name)}")
            self.logger.info(f"Short name: {model_short_name}")
            if "description" in config:
                self.logger.info(f"Description: {config['description']}")
            self.logger.info("=" * 80)

            # Check if this is a Hugging Face model
            huggingface_repo = config.get("huggingface_repo")
            if huggingface_repo:
                # Load model from Hugging Face
                model_library = config.get("model_library", "timm")
                model_params = config.get("model_params")
                model = self.load_huggingface_model(
                    repo_id=huggingface_repo,
                    model_library=model_library,
                    model_params=model_params,
                )
            else:
                # Traditional PyTorch model workflow
                # Download weights
                weights_url = config["weights_url"]
                weights_path = self.download_weights(weights_url)

                # Load model class
                model_class_name = config.get("model_class_name", "torch.nn.Module")
                model_class = self.load_model_class(model_class_name)

                # Load checkpoint
                checkpoint = self.load_checkpoint(weights_path)

                # Create model
                model_params = config.get("model_params")
                model = self.create_model(model_class, checkpoint, model_params)

            # Prepare export parameters
            input_shape = config.get("input_shape", [1, 3, 224, 224])
            input_names = config.get("input_names", ["input"])
            output_names = config.get("output_names", ["result"])

            # Prepare metadata from config (with defaults for ImageNet normalization)
            reverse_input_channels = config.get("reverse_input_channels", True)
            mean_values = config.get("mean_values", "123.675 116.28 103.53")
            scale_values = config.get("scale_values", "58.395 57.12 57.375")

            metadata = {
                ("model_info", "model_type"): config.get("model_type", ""),
                ("model_info", "model_short_name"): model_short_name,
                ("model_info", "reverse_input_channels"): str(reverse_input_channels),
                ("model_info", "mean_values"): mean_values,
                ("model_info", "scale_values"): scale_values,
            }

            # Add labels if specified in config
            labels_config = config.get("labels")
            if labels_config:
                labels = self.get_labels(labels_config)
                if labels:
                    metadata["model_info", "labels"] = labels
                    self.logger.info(f"Added {labels_config} labels to metadata")
                else:
                    self.logger.warning(f"Could not load labels for: {labels_config}")
            
            # Get model library (default to 'timm' for backward compatibility)
            model_library = config.get("model_library", "timm")
            
            output_path = self.output_dir / model_short_name
            fp16_model_path, fp32_model_path = self.export_to_openvino(
                model=model,
                input_shape=input_shape,
                output_path=output_path,
                input_names=input_names,
                output_names=output_names,
                metadata=metadata,
                model_library=model_library,
            )

            # Quantize the model if dataset is available
            if self.dataset_path:
                self.logger.info("Creating calibration dataset for INT8 quantization")

                self.logger.info("Creating validation dataset for accuracy measurement")
                validation_data, validation_labels = self.create_calibration_dataset(
                    input_shape=input_shape,
                    mean_values=mean_values,
                    scale_values=scale_values,
                    reverse_input_channels=reverse_input_channels,
                    subset_size=300,
                    return_labels=True,
                )

                if validation_data:
                    # Use FP32 model for better quantization accuracy
                    self.quantize_model(
                        model_path=fp32_model_path,
                        calibration_data=validation_data,
                        preset="mixed",
                        model_library=model_library,
                        validation_data=validation_data if validation_labels else None,
                        validation_labels=validation_labels,
                    )
                
                # Clean up temporary FP32 model after quantization
                try:
                    if fp32_model_path.exists():
                        fp32_model_path.unlink()
                        self.logger.debug(f"Removed temporary FP32 model: {fp32_model_path}")
                    # Also remove the .bin file
                    fp32_bin_path = fp32_model_path.with_suffix(".bin")
                    if fp32_bin_path.exists():
                        fp32_bin_path.unlink()
                        self.logger.debug(f"Removed temporary FP32 weights: {fp32_bin_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove temporary FP32 files: {e}")

            self.logger.info(f"✓ Successfully converted {model_short_name}")
            return True

        except (ValueError, RuntimeError, ImportError, FileNotFoundError) as e:
            self.logger.error(f"✗ Failed to process model {model_short_name}: {e}")
            import traceback

            self.logger.debug(traceback.format_exc())
            return False

    def process_config_file(
        self,
        config_path: Path,
        model_filter: Optional[str] = None,
    ) -> tuple[int, int]:
        """
        Process models from a configuration file.

        Args:
            config_path: Path to JSON configuration file
            model_filter: Optional model short name to process (process only this model)

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


def list_models(config_path: Path):
    """List all models in a configuration file."""
    try:
        with config_path.open() as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return

    models = config.get("models", [])

    if not models:
        print("No models found in configuration")
        return

    print(f"\nFound {len(models)} models:\n")
    print(f"{'Short Name':<30} {'Full Name':<40} {'Type':<20}")
    print("-" * 90)

    for model in models:
        short_name = model.get("model_short_name", "N/A")
        full_name = model.get("model_full_name", "N/A")
        model_type = model.get("model_type", "N/A")
        print(f"{short_name:<30} {full_name:<40} {model_type:<20}")

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
        "-d",
        "--dataset",
        type=Path,
        default=Path.home() / "model_api" / "Small-ImageNet-Validation-Dataset-1000-Classes",
        help="Path to calibration dataset for INT8 quantization (default: ~/model_api/Small-ImageNet-Validation-Dataset-1000-Classes)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Process only the specified model (by model_short_name)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all models in the configuration file and exit",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Check if config file exists
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
        return 1

    # List models and exit
    if args.list:
        list_models(args.config)
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
        # Create converter
        converter = ModelConverter(
            output_dir=args.output,
            cache_dir=args.cache,
            verbose=args.verbose,
            dataset_path=args.dataset,
        )

        logger.info(f"Output directory: {args.output}")
        logger.info(f"Cache directory: {args.cache}")
        if args.dataset:
            logger.info(f"Calibration dataset: {args.dataset}")

        # Process models
        successful, failed = converter.process_config_file(
            config_path=args.config,
            model_filter=args.model,
        )

        # Print summary
        logger.info("=" * 80)
        logger.info("Conversion Summary:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total: {successful + failed}")
        logger.info("=" * 80)

        return 0 if failed == 0 else 1
    except (ValueError, RuntimeError, ImportError, FileNotFoundError) as e:
        logger.error(f"Failed to process model: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
