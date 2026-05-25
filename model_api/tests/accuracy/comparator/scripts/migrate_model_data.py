#!/usr/bin/env python3
#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""One-time migration: add reference_dir to model_data JSON configs."""
import hashlib
import json
import re
from pathlib import Path


def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)


def migrate_config(path: Path) -> None:
    cfg = json.loads(path.read_text())
    config_basename = path.stem
    seen = set()
    for model_idx, model in enumerate(cfg):
        for test_idx, td in enumerate(model["test_data"]):
            model_slug = sanitize(model.get("name", f"model_{model_idx}"))
            image_slug = sanitize(Path(td["image"]).stem)
            # to calculate stable reference_dir, we need to ignore the existing reference_dir (if any) and reference (if any)
            td["reference_dir"] = ""
            options_hash = hashlib.sha1(
                json.dumps(td, sort_keys=True).encode(), usedforsecurity=False,
            ).hexdigest()[:8]
            ref_dir = f"{config_basename}/{model_idx}_{model_slug}/{test_idx}_{image_slug}_{options_hash}"
            assert ref_dir not in seen, f"COLLISION: {ref_dir}"
            seen.add(ref_dir)
            td["reference_dir"] = ref_dir
            td.pop("reference", None)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"Migrated {path}: {len(seen)} entries, all unique")


if __name__ == "__main__":
    configs = [
        Path("tests/accuracy/public_scope.json"),
        Path("tests/accuracy/otx_2.6.0_models.json"),
        Path("tests/accuracy/geti_2.13.0_models.json"),
    ]
    for c in configs:
        migrate_config(c)
