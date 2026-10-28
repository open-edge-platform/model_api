#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Getitune model converter."""

import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

from jsonargparse import ArgumentParser as JArgParser
from jsonargparse import Namespace

from model_converter.converters.base import BaseConverter
from model_converter.reporting import AccuracyResults

# Location where the getitune backbones cache their downloaded pre-trained checkpoints.
CHECKPOINT_CACHE_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"

CLASSIFICATION_TASKS = {"MULTI_CLASS_CLS", "MULTI_LABEL_CLS"}

# Number of classes in the official DINOv2 ImageNet-1k linear head.
_IMAGENET_NUM_CLASSES = 1000

# Official facebookresearch/dinov2 ImageNet-1k linear-probe head for the single
# DINOv2 variant exported here. The runtime ``2 * embed_dim`` shape check guards
# against the head not matching the backbone.
_DINOV2_LINEAR_HEAD_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_linear_head.pth"


def _dinov2_backbone_forward(self, x, out_type: str = "cls_token"):
    """Backbone forward returning ``concat(cls_token, mean(patch_tokens))`` for the default call.

    Mirrors ``create_linear_input`` of facebookresearch/dinov2 with ``n_last_blocks=1``
    and ``use_avgpool=True``. Other ``out_type`` values (e.g. ``"raw"``) fall through to
    the original implementation.
    """
    import torch

    original_forward = type(self).forward
    if out_type != "cls_token":
        return original_forward(self, x, out_type=out_type)
    tokens = original_forward(self, x, out_type="raw")[-1]  # [B, N, D], already normalized
    cls_token = tokens[:, 0]
    patch_mean = tokens[:, self.num_prefix_tokens :].mean(dim=1)  # skip cls + register tokens
    return (torch.cat([cls_token, patch_mean], dim=-1),)


class GetituneConverter(BaseConverter):
    """Converter for getitune models.

    Invokes the export_pretrained_models.py script as a subprocess to export
    models, then repackages the output and applies INT8 quantization.
    """

    def __init__(self, **kwargs: Any):
        """Initialize GetituneConverter.

        Args:
            **kwargs: Arguments passed to BaseConverter
        """
        super().__init__(**kwargs)

    @staticmethod
    def _last_linear(module):
        """Return the last ``nn.Linear`` submodule of ``module`` (e.g. the classifier), or ``None``."""
        from torch import nn

        last = None
        for sub in module.modules():
            if isinstance(sub, nn.Linear):
                last = sub
        return last

    @staticmethod
    def _head_weights_pytorchcv_efficientnet(model_name: str):
        """Pre-trained ImageNet-1k head for pytorchcv EfficientNet backbones (``output.fc.*``)."""
        import torch
        from pytorchcv.models.common.model_store import get_model_file

        path = get_model_file(model_name=model_name, local_model_store_dir_path=str(CHECKPOINT_CACHE_DIR))
        # weights_only=True restricts unpickling to plain tensors/state dicts, avoiding the
        # arbitrary-code-execution risk of full pickle deserialization (these checkpoints are
        # just weight tensors).
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "output.fc.weight" not in state_dict or "output.fc.bias" not in state_dict:
            return None
        return state_dict["output.fc.weight"], state_dict["output.fc.bias"]

    @staticmethod
    def _head_weights_torchvision(model_name: str):
        """Pre-trained ImageNet head for torchvision backbones (last ``nn.Linear`` of the classifier)."""
        from torchvision.models import get_model, get_model_weights

        net = get_model(name=model_name, weights=get_model_weights(model_name))
        linear = GetituneConverter._last_linear(net)
        if linear is None:
            return None
        return linear.weight.detach().clone(), linear.bias.detach().clone()

    @staticmethod
    def _head_weights_timm(model_name: str):
        """Pre-trained head for timm backbones, if the checkpoint exposes a plain ``nn.Linear`` classifier."""
        import timm
        from torch import nn

        net = timm.create_model(model_name, pretrained=True)
        classifier = net.get_classifier()
        if not isinstance(classifier, nn.Linear) or classifier.bias is None:
            return None
        return classifier.weight.detach().clone(), classifier.bias.detach().clone()

    @staticmethod
    def _head_weights_vit(model_name: str):
        """Pre-trained ImageNet head for augreg ViT ``.npz`` checkpoints (``head/kernel`` / ``head/bias``).

        DINOv2 checkpoints are self-supervised and contain no classification head.
        """
        import numpy as np
        import torch
        from getitune.backend.lightning.models.classification.utils.pretrained_urls import VIT_PRETRAINED_URLS

        url = VIT_PRETRAINED_URLS.get(model_name)
        if url is None or not url.endswith(".npz"):
            return None  # dinov2 (.pth) and unknown variants have no usable head

        cache_file = CHECKPOINT_CACHE_DIR / Path(urlparse(url).path).name
        if not cache_file.exists():
            return None

        weights = np.load(cache_file)
        for prefix in ("", "opt/target/", "params/"):
            kernel_key, bias_key = f"{prefix}head/kernel", f"{prefix}head/bias"
            if kernel_key in weights and bias_key in weights:
                kernel = torch.from_numpy(weights[kernel_key]).t().contiguous()  # (in, out) -> (out, in)
                bias = torch.from_numpy(weights[bias_key])
                return kernel, bias
        return None

    # Maps the backbone class name to (handler, needs_model_name).
    _HEAD_SOURCES: ClassVar[dict[str, Any]] = {
        "EfficientNetFeatureExtractor": _head_weights_pytorchcv_efficientnet,
        "TorchvisionBackbone": _head_weights_torchvision,
        "TimmBackbone": _head_weights_timm,
        "VisionTransformerBackbone": _head_weights_vit,
    }

    @staticmethod
    def _get_recipe_paths(task: str | None = None) -> list[str]:
        """Get all recipe YAML paths for a given task, excluding OpenVINO and tile variants."""
        from getitune.types.task import TaskType
        from getitune.utils import list_models

        recipes = list_models(task=TaskType(task) if task is not None else None, return_recipes=True)
        # Filter out openvino_model and tile variants
        filtered = []
        for r in recipes:
            name = Path(r).stem
            if name == "openvino_model":
                continue
            if "_tile" in name:
                continue
            filtered.append(r)
        return sorted(filtered)

    @staticmethod
    def _load_recipe_config(recipe_path: str) -> dict:
        """Load and resolve a recipe YAML configuration."""
        from getitune.cli.utils.jsonargparse import get_configuration

        return get_configuration(recipe_path)

    @staticmethod
    def _get_label_info_from_config(config: dict) -> int | None:
        """Extract label_info (num_classes) from a resolved recipe config."""
        model_config = config.get("model", {})
        init_args = model_config.get("init_args", {})
        label_info = init_args.get("label_info")
        if isinstance(label_info, int):
            return label_info
        return None

    @staticmethod
    def _get_input_size_from_config(config: dict) -> tuple[int, int] | None:
        """Extract input_size from a resolved recipe config (checks overrides first, then data)."""
        # Check overrides first
        overrides = config.get("overrides", {})
        if isinstance(overrides, dict):
            data_overrides = overrides.get("data", {})
            if isinstance(data_overrides, dict):
                input_size = data_overrides.get("input_size")
                if input_size is not None:
                    return tuple(input_size)

        # Fallback to data config
        data_config = config.get("data", {})
        if isinstance(data_config, dict):
            input_size = data_config.get("input_size")
            if input_size is not None:
                return tuple(input_size)

        return None

    @staticmethod
    def _create_model_from_config(config: dict, label_info: int, input_size: tuple[int, int] | None = None):
        """Create a LightningModel from config with original label_info and optional input_size.

        The model is created with the original pre-trained number of classes,
        so pre-trained weights (including the head) are loaded as-is.
        """
        from dataclasses import asdict

        from getitune.backend.lightning.models.base import LightningModel
        from getitune.config.data import IntensityConfig
        from getitune.utils.utils import get_model_cls_from_config, should_pass_label_info

        model_config = deepcopy(config["model"])

        # Remove non-serializable fields that get_configuration may resolve to function objects.
        # These are optional fields with defaults in the model class and are not needed for export.
        init_args = model_config.get("init_args", {})
        for key in list(init_args.keys()):
            if callable(init_args[key]) and not isinstance(init_args[key], (type, dict)):
                del init_args[key]

        # Remove pre-resolved None data_input_params so we can inject ours
        if init_args.get("data_input_params") is None:
            init_args.pop("data_input_params", None)

        # Inject data_input_params so the export embeds correct preprocessing.
        # `intensity_config` (scale_to_unit, ÷255) is mandatory: getitune trains with
        # a two-stage pipeline.
        data_input_params: dict = {"intensity_config": asdict(IntensityConfig())}
        if input_size is not None:
            data_input_params["input_size"] = input_size
        model_config["init_args"]["data_input_params"] = data_input_params

        model_cls = get_model_cls_from_config(Namespace(model_config))

        skip = set()
        if should_pass_label_info(model_cls):
            model_config["init_args"]["label_info"] = label_info
            skip.add("label_info")

        model_parser = JArgParser()
        model_parser.add_subclass_arguments(
            LightningModel,
            "model",
            skip=skip,
            required=False,
            fail_untyped=False,
        )
        return model_parser.instantiate_classes(Namespace(model=model_config)).get("model")

    @staticmethod
    def _find_head_linear(model):
        """Return the final ``nn.Linear`` of a classification model's head, or ``None``.

        Supports ``LinearClsHead`` / ``MultiLabelLinearClsHead`` (``head.fc``) and
        ``VisionTransformerClsHead`` (``head.layers.head``).
        """
        from torch import nn

        head = getattr(getattr(model, "model", None), "head", None)
        if head is None:
            return None

        fc = getattr(head, "fc", None)
        if isinstance(fc, nn.Linear):
            return fc

        layers = getattr(head, "layers", None)
        vit_head = getattr(layers, "head", None) if layers is not None else None
        if isinstance(vit_head, nn.Linear):
            return vit_head

        return None

    @staticmethod
    def _is_dinov2_multiclass(model) -> bool:
        """Return ``True`` for a multi-class DINOv2 model needing the official linear head.

        DINOv2 backbones are self-supervised ``.pth`` checkpoints (detected via
        ``VIT_PRETRAINED_URLS``) that ship no classification head, so getitune leaves a
        randomly initialized ``VisionTransformerClsHead`` (``head.layers.head``). Other
        ViT variants (e.g. augreg ``.npz``) keep their usual head-recovery path.
        """
        from getitune.backend.lightning.models.classification.utils.pretrained_urls import VIT_PRETRAINED_URLS
        from torch import nn

        backbone = getattr(getattr(model, "model", None), "backbone", None)
        if type(backbone).__name__ != "VisionTransformerBackbone":
            return False

        model_name = getattr(model, "model_name", None)
        url = VIT_PRETRAINED_URLS.get(model_name) if model_name else None
        if url is None or not url.endswith(".pth"):
            return False

        # Multi-class head only (VisionTransformerClsHead -> head.layers.head);
        # multi-label DINOv2 uses a different head and keeps the skip path.
        head = getattr(getattr(model, "model", None), "head", None)
        layers = getattr(head, "layers", None)
        return isinstance(getattr(layers, "head", None), nn.Linear)

    @staticmethod
    def _apply_dinov2_linear_head(model) -> str:
        """Replace the DINOv2 model's head with the official facebookresearch linear head.

        Downloads (and caches) the official ImageNet-1k ``*_reg4_linear_head.pth`` head,
        rewires the backbone to emit ``concat(cls_token, mean(patch_tokens))`` and swaps the
        final ``nn.Linear`` for the pre-trained one. Mutates ``model`` in place.

        Returns:
            A human-readable detail string describing the applied head.

        Raises:
            ValueError: If the head does not match the backbone (``2 * embed_dim`` inputs),
                is not the 1000-class ImageNet head, or the model is not a 1000-class model.
        """
        import types

        import torch
        from torch import nn
        from torch.hub import download_url_to_file

        classifier = model.model
        backbone = classifier.backbone
        head = classifier.head

        CHECKPOINT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        head_weights = CHECKPOINT_CACHE_DIR / Path(urlparse(_DINOV2_LINEAR_HEAD_URL).path).name
        if not head_weights.exists():
            download_url_to_file(_DINOV2_LINEAR_HEAD_URL, str(head_weights), progress=True)

        # weights_only=True restricts unpickling to plain tensors (these checkpoints are
        # just weight tensors), avoiding arbitrary-code-execution on load.
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        state = torch.load(head_weights, map_location="cpu", weights_only=True)

        embed_dim = backbone.embed_dim
        out_features, in_features = state["weight"].shape
        if in_features != 2 * embed_dim:
            message = (
                f"DINOv2 head expects {in_features} input features but backbone provides "
                f"2*{embed_dim}; the cached linear head does not match this backbone"
            )
            raise ValueError(message)
        if out_features != _IMAGENET_NUM_CLASSES:
            message = (
                f"DINOv2 head weights have {out_features} outputs, expected {_IMAGENET_NUM_CLASSES}; "
                "the official ImageNet-1k *_linear_head.pth is required"
            )
            raise ValueError(message)
        num_classes = getattr(model, "num_classes", None)
        if num_classes != _IMAGENET_NUM_CLASSES:
            message = (
                f"Official DINOv2 head has {_IMAGENET_NUM_CLASSES} classes but the model has "
                f"{num_classes}; the DINOv2 recipe must use the 1000 ImageNet classes"
            )
            raise ValueError(message)

        # 1) Backbone: default call returns concat(cls, mean(patch tokens)) instead of cls only.
        backbone.forward = types.MethodType(_dinov2_backbone_forward, backbone)

        # 2) Head: replace the final Linear with the official one.
        linear = nn.Linear(in_features, out_features)
        linear.load_state_dict(state)
        linear.eval()
        head.layers.head = linear
        head.in_channels = in_features

        return f"applied official DINOv2 linear head {tuple(state['weight'].shape)} from {head_weights}"

    @staticmethod
    def _recover_pretrained_head(model) -> tuple[str, str]:
        """Restore the pre-trained classification head weights onto ``model`` in-place.

        The classification backbones are feature extractors that silently drop the
        checkpoint's classifier head, after which ``_create_model()`` builds a fresh,
        randomly initialized head. This copies the original pre-trained head weights
        back onto the model's head so the exported model is accurate.

        Returns:
            A ``(status, detail)`` tuple where status is one of ``"recovered"``,
            ``"skipped"`` or ``"failed"``.
        """
        import torch

        target = GetituneConverter._find_head_linear(model)
        if target is None:
            head = getattr(getattr(model, "model", None), "head", None)
            head_type = type(head).__name__ if head is not None else "<none>"
            return "skipped", f"unsupported head type '{head_type}' (no plain nn.Linear classifier)"

        # DINOv2 ships no classification head; load the official facebookresearch linear
        # head (and rewire the backbone) instead of copying weights. Errors hard-fail.
        if GetituneConverter._is_dinov2_multiclass(model):
            return "recovered", GetituneConverter._apply_dinov2_linear_head(model)

        backbone = getattr(getattr(model, "model", None), "backbone", None)
        backbone_type = type(backbone).__name__ if backbone is not None else "<none>"
        handler = GetituneConverter._HEAD_SOURCES.get(backbone_type)
        if handler is None:
            return "skipped", f"no head-recovery handler for backbone '{backbone_type}'"

        model_name = getattr(model, "model_name", None)
        if not model_name:
            return "skipped", "model has no 'model_name'"

        try:
            source = handler(model_name)
        except Exception as exc:  # noqa: BLE001
            # best effort: failures are surfaced in the export summary
            return "failed", f"error fetching pre-trained head: {exc}"

        if source is None:
            return "skipped", f"no compatible pre-trained head available for '{model_name}'"

        weight, bias = source
        if tuple(weight.shape) != tuple(target.weight.shape) or tuple(bias.shape) != tuple(target.bias.shape):
            return (
                "skipped",
                f"head shape mismatch (checkpoint {tuple(weight.shape)} vs model {tuple(target.weight.shape)})",
            )

        with torch.no_grad():
            target.weight.copy_(weight.to(target.weight.dtype))
            target.bias.copy_(bias.to(target.bias.dtype))
        return "recovered", f"loaded pre-trained head {tuple(weight.shape)} from '{model_name}'"

    @staticmethod
    def _export_model(model, output_dir: Path, export_format: str = "OPENVINO", precision: str = "FP32") -> Path:
        """Export a model with its current (pre-trained) weights."""
        from getitune.types.export import ExportFormat
        from getitune.types.precision import Precision

        fmt = ExportFormat(export_format)
        prec = Precision(precision)

        output_dir.mkdir(parents=True, exist_ok=True)
        return model.export(
            output_dir=output_dir,
            base_name="exported_model",
            export_format=fmt,
            precision=prec,
        )

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
            self._validate_license(config, model_short_name)

            self._log_model_banner(config, model_short_name, label="getitune model")

            # Export model
            exported_model_path = self._run_export(config)

            # Repackage into standard layout
            self._repackage_model(config, exported_model_path)

            # Quantize if enabled and dataset is available
            accuracy: AccuracyResults | None = None
            quantization_attempted = bool(config.get("quantize", False) and config.get("dataset_type"))
            if quantization_attempted:
                accuracy = self._quantize_exported_model(config)

            return self._finalize_success(
                config,
                model_short_name,
                accuracy=accuracy,
                quantization_attempted=quantization_attempted,
            )

        except (ValueError, RuntimeError, FileNotFoundError) as e:
            return self._record_failure(config, model_short_name, e, label="getitune model")

    def _run_export(self, config: dict[str, Any]) -> Path:
        """Run the model export.

        Args:
            config: Model configuration dictionary

        Returns:
            Path to the exported model XML file
        """
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

        recipes = GetituneConverter._get_recipe_paths(getitune_task)
        recipe_path = next((r for r in recipes if Path(r).stem == getitune_recipe), None)

        if not recipe_path:
            error_msg = f"Recipe '{getitune_recipe}' not found"
            raise ValueError(error_msg)

        # Load and parse recipe config
        config = GetituneConverter._load_recipe_config(recipe_path)

        # Get original label_info (num_classes from pre-trained dataset)
        label_info = GetituneConverter._get_label_info_from_config(config)
        if label_info is None:
            error_msg = "No label_info in recipe (requires dataset-specific labels)"
            raise ValueError(error_msg)

        # Get input_size from config
        input_size = GetituneConverter._get_input_size_from_config(config)

        self.logger.info(f"label_info (num_classes): {label_info}")
        if input_size:
            self.logger.info(f"input_size: {input_size}")

        # Create model with original pre-trained weights
        self.logger.info("Creating model with original pre-trained weights...")
        model = GetituneConverter._create_model_from_config(config, label_info=label_info, input_size=input_size)

        # Restore the pre-trained classification head (backbones drop it on load,
        # leaving a randomly initialized head -> ~0% accuracy otherwise).
        if getitune_task in CLASSIFICATION_TASKS:
            head_status, head_detail = GetituneConverter._recover_pretrained_head(model)
            if head_status == "recovered":
                self.logger.info(f"Pre-trained head: {head_detail}")
            elif head_status == "skipped":
                self.logger.info(f"Pre-trained head NOT recovered: {head_detail}")
            else:  # "failed"
                self.logger.info(f"Pre-trained head fetching failed: {head_detail}")
                raise ValueError(head_detail)

        # Export
        self.logger.info(f"Exporting to: {temp_output}")
        exported_path = GetituneConverter._export_model(
            model,
            output_dir=temp_output,
            precision="FP16",
        )
        self.logger.info(f"SUCCESS: {exported_path}")
        return exported_path

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
        self._apply_label_info(config, target_xml, fp32_xml)
        self._apply_model_info_overrides(config, target_xml, fp32_xml)
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

    @staticmethod
    def _make_label_names_unique(names: list[str]) -> list[str]:
        """Disambiguate duplicate label names the way datumaro does on import.

        ImageNet-1k contains two distinct ``crane`` synsets; datumaro keeps the
        first occurrence as-is and suffixes later duplicates with an incrementing
        count (``crane`` -> ``crane2``). getitune datasets therefore carry the
        de-duplicated names, so the exported ``label_info`` must match them
        exactly for the test/predict label check to pass.
        """
        seen: dict[str, int] = {}
        unique: list[str] = []
        for name in names:
            count = seen.get(name, 0) + 1
            seen[name] = count
            unique.append(name if count == 1 else f"{name}{count}")
        return unique

    def _resolve_label_info_names(self, config: dict[str, Any]) -> list[str] | None:
        """Resolve the class names for the embedded ``label_info`` (getitune convention).

        getitune/datumaro name ImageNet-1k classes with their full WordNet synset
        descriptions (e.g. ``"tench, Tinca tinca"``), de-duplicated via
        :meth:`_make_label_names_unique`. This differs from the Model API
        ``labels`` field (short, underscore-joined tokens), so ``label_info`` is
        resolved separately to match the dataset the model is evaluated against.
        """
        labels_config = config.get("labels")
        if not labels_config:
            return None

        if labels_config == "IMAGENET1K_V1":
            from timm.data import ImageNetInfo

            names = list(ImageNetInfo("imagenet1k").label_descriptions())
            return self._make_label_names_unique(names)

        # Fallback: reuse the Model API label strings (better than placeholders,
        # though they may not match a specific dataset's naming convention).
        resolved = self.get_labels(labels_config)
        if not resolved:
            return None
        return resolved.split(" ")

    def _apply_label_info(self, config: dict[str, Any], *model_paths: Path) -> None:
        """Overwrite the exporter's placeholder ``model_info/label_info`` with real names.

        The getitune exporter embeds ``label_info`` with placeholder
        ``label_0..label_{N-1}`` names because the model is built from a recipe
        without a dataset. getitune's test/predict pipeline reads this JSON blob
        (not ``model_info/labels``) and rejects the model unless its
        ``label_info`` matches the evaluation dataset. Rewrite the ``label_names``
        (and ``label_groups``, preserving grouping) with the resolved class names,
        keeping the existing ``label_ids``.

        Args:
            config: Model configuration dictionary.
            *model_paths: OpenVINO model XML files to update in place.
        """
        import json

        names = self._resolve_label_info_names(config)
        if not names:
            return

        try:
            import openvino as ov

            core = ov.Core()
            for model_path in model_paths:
                if not model_path.exists():
                    continue
                model = core.read_model(model_path)
                try:
                    serialized = model.get_rt_info(["model_info", "label_info"]).astype(str)
                except RuntimeError:
                    continue

                info = json.loads(serialized)
                old_names = info.get("label_names", [])
                if len(old_names) != len(names):
                    self.logger.warning(
                        f"label_info has {len(old_names)} names but resolved {len(names)}; "
                        "skipping label_info update",
                    )
                    continue

                name_map = dict(zip(old_names, names))
                info["label_names"] = names
                if isinstance(info.get("label_groups"), list):
                    info["label_groups"] = [[name_map.get(n, n) for n in group] for group in info["label_groups"]]

                model.set_rt_info(json.dumps(info), ["model_info", "label_info"])
                tmp_xml = model_path.with_name(f"{model_path.stem}_label_info_tmp.xml")
                ov.save_model(model, tmp_xml, compress_to_fp16=not model_path.stem.endswith("_fp32"))
                tmp_xml.replace(model_path)
                tmp_xml.with_suffix(".bin").replace(model_path.with_suffix(".bin"))
            self.logger.info("✓ Applied real label_info to exported model(s)")
        except (ImportError, RuntimeError) as e:
            self.logger.warning(f"Could not apply label_info to exported model: {e}")

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

    def _apply_model_info_overrides(self, config: dict[str, Any], *model_paths: Path) -> None:
        """Override arbitrary ``model_info`` rt_info fields with configured values.

        Args:
            config: Model configuration dictionary.
            *model_paths: OpenVINO model XML files to update in place.
        """
        overrides = config.get("model_info_overrides")
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

                tmp_xml = model_path.with_name(f"{model_path.stem}_model_info_tmp.xml")
                ov.save_model(model, tmp_xml, compress_to_fp16=not model_path.stem.endswith("_fp32"))
                tmp_xml.replace(model_path)
                tmp_xml.with_suffix(".bin").replace(model_path.with_suffix(".bin"))
            self.logger.info(f"✓ Applied model_info overrides to exported model(s): {overrides}")
        except (ImportError, RuntimeError) as e:
            self.logger.warning(f"Could not apply model_info overrides to exported model: {e}")

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

        # Get input shape from the model's input layer. Use ``partial_shape``
        # rather than ``.shape`` because some backbones (e.g. DINOv2/ViT) export
        # with a dynamic batch dimension; ``.shape`` calls ``get_shape()`` which
        # raises "to_shape was called on a dynamic shape". Resolve any dynamic
        # dimension to a concrete size (batch -> 1) so downstream calibration can
        # build fixed-size tensors.
        partial_shape = model.input(0).partial_shape
        input_shape = [dim.get_length() if dim.is_static else 1 for dim in partial_shape]

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
