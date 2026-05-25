"""Reference storage helpers for future comparator workflows."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import json
import platform
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from shutil import which

import numpy as np
from PIL import Image


def save_class_map(path: Path, arr: np.ndarray) -> None:
    """Save a class-map as 16-bit PNG (mode I;16). Supports up to 65535 classes."""
    if arr.ndim != 2:
        msg = f"class map must be 2-D, got shape {arr.shape}"
        raise ValueError(msg)
    img = Image.fromarray(arr.astype(np.uint16), mode="I;16")
    img.save(str(path), format="PNG")


def load_class_map(path: Path) -> np.ndarray:
    """Load a 16-bit PNG class-map. Returns uint16 ndarray."""
    with Image.open(str(path)) as img:
        return np.array(img, dtype=np.uint16)


def save_binary_mask(path: Path, arr: np.ndarray) -> None:
    """Save a boolean / {0,1} 2-D mask as 1-bit PNG (mode '1')."""
    if arr.ndim != 2:
        msg = f"binary mask must be 2-D, got shape {arr.shape}"
        raise ValueError(msg)
    bool_arr = arr.astype(bool)
    img = Image.fromarray((bool_arr.astype(np.uint8) * 255), mode="L").convert("1")
    img.save(str(path), format="PNG")


def load_binary_mask(path: Path) -> np.ndarray:
    """Load a 1-bit PNG mask as boolean ndarray."""
    with Image.open(str(path)) as img:
        return np.array(img.convert("1"), dtype=bool)


def save_instance_masks(path: Path, masks: np.ndarray, threshold: float = 0.5) -> None:
    """Binarize per-instance masks at threshold and store stacked as compressed NPZ.

    Expects masks of shape (N, H, W) (float probabilities or already-binary).
    """
    if masks.ndim != 3:
        msg = f"instance masks must be 3-D (N,H,W), got shape {masks.shape}"
        raise ValueError(msg)
    binary = masks >= threshold
    np.savez_compressed(str(path), masks=binary.astype(np.bool_))


def load_instance_masks(path: Path) -> np.ndarray:
    """Load stacked instance masks. Returns boolean ndarray of shape (N, H, W)."""
    with np.load(str(path)) as npz:
        return np.asarray(npz["masks"], dtype=bool)


def compute_mean_iou(pred: np.ndarray, ref: np.ndarray) -> float:
    """Binary IoU = |intersection| / |union|. Returns 1.0 if both empty."""
    if pred.shape != ref.shape:
        msg = f"shape mismatch: {pred.shape} vs {ref.shape}"
        raise ValueError(msg)
    p = pred.astype(bool)
    r = ref.astype(bool)
    intersection = np.logical_and(p, r).sum()
    union = np.logical_or(p, r).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def compute_class_map_iou(
    pred: np.ndarray,
    ref: np.ndarray,
    num_classes: int | None = None,
) -> float:
    """Mean per-class IoU over class-maps. Skips classes absent from both pred and ref."""
    if pred.shape != ref.shape:
        msg = f"shape mismatch: {pred.shape} vs {ref.shape}"
        raise ValueError(msg)
    classes = (
        np.union1d(np.unique(pred), np.unique(ref))
        if num_classes is None
        else np.arange(num_classes)
    )
    ious: list[float] = []
    for c in classes:
        p = pred == c
        r = ref == c
        if not p.any() and not r.any():
            continue
        inter = np.logical_and(p, r).sum()
        union = np.logical_or(p, r).sum()
        ious.append(float(inter) / float(union) if union else 1.0)
    if not ious:
        return 1.0
    return float(np.mean(ious))


@dataclass
class ReferenceBundle:
    """In-memory representation of a reference directory."""

    result_json: dict
    mask_paths: dict[str, Path] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    generated_by: dict = field(default_factory=dict)


def build_generated_by() -> dict:
    """Best-effort capture of environment provenance for a reference bundle."""
    try:
        git_exe = which("git")
        if git_exe:
            git_commit = subprocess.run(  # noqa: S603
                [git_exe, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip() or "unknown"
        else:
            git_commit = "unknown"
    except (OSError, subprocess.SubprocessError):
        git_commit = "unknown"

    try:
        import openvino  # type: ignore[import-not-found]

        openvino_version = getattr(openvino, "__version__", "unknown")
    except ImportError:
        openvino_version = "unknown"

    try:
        import model_api  # type: ignore[import-not-found]

        model_api_version = getattr(model_api, "__version__", "unknown")
    except ImportError:
        model_api_version = "unknown"

    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "openvino_version": openvino_version,
        "model_api_version": model_api_version,
        "git_commit": git_commit,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
    }


def _save_mask(path_stem: Path, arr: np.ndarray) -> Path:
    """Dispatch mask saving by array shape/dtype. Returns the saved path."""
    if arr.ndim == 3 and arr.dtype == np.bool_:
        out = path_stem.with_suffix(".npz")
        save_instance_masks(out, arr.astype(np.float32))
        return out
    if arr.ndim == 2:
        if arr.dtype == np.uint16 or (np.issubdtype(arr.dtype, np.integer) and arr.max(initial=0) > 1):
            out = path_stem.with_suffix(".png")
            save_class_map(out, arr)
            return out
        out = path_stem.with_suffix(".png")
        save_binary_mask(out, arr)
        return out
    msg = f"unsupported mask shape/dtype for field {path_stem.name!r}: {arr.shape} {arr.dtype}"
    raise ValueError(msg)


def save_reference(
    reference_dir: Path,
    *,
    result_json: dict,
    masks: dict[str, np.ndarray] | None,
    metadata: dict,
    generated_by: dict,
    overwrite: bool = False,
) -> None:
    """Persist a reference bundle on disk.

    Layout:
      reference_dir/
        result.json
        metadata.json
        generated_by.json
        masks/<field>.{png,npz}
    """
    reference_dir = Path(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)

    result_path = reference_dir / "result.json"
    if result_path.exists() and not overwrite:
        msg = f"reference already exists at {reference_dir} (pass overwrite=True to replace)"
        raise FileExistsError(msg)

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, sort_keys=True)
    with (reference_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    with (reference_dir / "generated_by.json").open("w", encoding="utf-8") as f:
        json.dump(generated_by, f, indent=2, sort_keys=True)

    if masks:
        masks_dir = reference_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        for field_name, arr in masks.items():
            _save_mask(masks_dir / field_name, np.asarray(arr))


def load_reference(reference_dir: Path) -> ReferenceBundle:
    """Load a reference bundle previously written by :func:`save_reference`."""
    reference_dir = Path(reference_dir)
    result_path = reference_dir / "result.json"
    if not result_path.exists():
        msg = f"no reference found at {reference_dir}: missing result.json"
        raise FileNotFoundError(msg)

    with result_path.open(encoding="utf-8") as f:
        result_json = json.load(f)

    metadata: dict = {}
    metadata_path = reference_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)

    generated_by: dict = {}
    generated_by_path = reference_dir / "generated_by.json"
    if generated_by_path.exists():
        with generated_by_path.open(encoding="utf-8") as f:
            generated_by = json.load(f)

    mask_paths: dict[str, Path] = {}
    masks_dir = reference_dir / "masks"
    if masks_dir.is_dir():
        for p in sorted(masks_dir.iterdir()):
            if p.suffix in {".png", ".npz"}:
                mask_paths[p.stem] = p

    return ReferenceBundle(
        result_json=result_json,
        mask_paths=mask_paths,
        metadata=metadata,
        generated_by=generated_by,
    )
