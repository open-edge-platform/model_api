#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""YOLO model converter."""

import json
import shutil
from pathlib import Path
from typing import Any

from defusedxml.ElementTree import ParseError, parse

from model_converter.converters.base import BaseConverter

MODEL_VERSIONS = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]


class YoloConverter(BaseConverter):
    """Converter for Ultralytics YOLO models.

    Uses the Ultralytics library to export YOLO models to OpenVINO format,
    then repackages and adds metadata.
    """

    def process_model_config(self, config: dict[str, Any]) -> bool:
        """Process a YOLO model configuration.

        Args:
            config: Model configuration dictionary with 'yolo_version' field

        Returns:
            True if successful, False otherwise
        """
        model_short_name = config.get("model_short_name", "unknown")
        yolo_version = config.get("yolo_version", model_short_name)

        # Check if both FP16 and INT8 models already exist
        fp16_folder = self.output_dir / f"{model_short_name}-fp16-ov"
        int8_folder = self.output_dir / f"{model_short_name}-int8-ov"

        if (fp16_folder / f"{yolo_version}.xml").exists() and (int8_folder / f"{yolo_version}.xml").exists():
            self.logger.info(f"Skipping {model_short_name}: FP16 and INT8 models already exist")
            return True

        try:
            from ultralytics import YOLO

            self.logger.info("=" * 80)
            self.logger.info(f"Processing YOLO model: {model_short_name}")
            self.logger.info("=" * 80)

            yolo_size = yolo_version[-1]  # n, s, m, l, or x

            # Load model from cache directory to avoid polluting the working directory
            model_path = self.cache_dir / f"{yolo_version}.pt"
            model = YOLO(str(model_path))

            # Export regular OpenVINO model (FP16)
            self.logger.info(f"Exporting {yolo_version} to OpenVINO FP16 format...")
            model.export(format="openvino", half=True)

            # Ultralytics exports next to the .pt file in the cache directory
            old_name = self.cache_dir / f"{yolo_version}_openvino_model"
            fp16_folder.mkdir(parents=True, exist_ok=True)
            if old_name.exists():
                if fp16_folder.exists():
                    shutil.rmtree(fp16_folder)
                old_name.rename(fp16_folder)

                # Update model_type in XML metadata
                xml_file = fp16_folder / f"{yolo_version}.xml"
                if xml_file.exists():
                    self._update_model_type_in_xml(xml_file, "YOLO11")

                # Copy README template for fp16
                self._copy_yolo_readme("README-yolo-fp16.md", fp16_folder, yolo_size)

            # Export INT8 quantized model
            self.logger.info(f"Exporting {yolo_version} to OpenVINO INT8 format...")
            model.export(format="openvino", int8=True, data="coco128.yaml")

            # Rename output folder for INT8
            old_name_int8 = self.cache_dir / f"{yolo_version}_int8_openvino_model"
            if old_name_int8.exists():
                if int8_folder.exists():
                    shutil.rmtree(int8_folder)
                old_name_int8.rename(int8_folder)

                # Update model_type in XML metadata
                xml_file = int8_folder / f"{yolo_version}.xml"
                if xml_file.exists():
                    self._update_model_type_in_xml(xml_file, "YOLO11")

                # Copy README template for int8
                self._copy_yolo_readme("README-yolo-int8.md", int8_folder, yolo_size)

            self.logger.info(f"✓ Successfully converted {model_short_name}")
            return True

        except (ValueError, RuntimeError, ImportError, FileNotFoundError, OSError) as e:
            self.logger.error(f"✗ Failed to process YOLO model {model_short_name}: {e}")
            import traceback

            self.logger.debug(traceback.format_exc())
            return False

    def _copy_yolo_readme(self, template_name: str, dest_dir: Path, yolo_size: str) -> None:
        """Copy a YOLO README template, replacing <<yolo_size>>."""
        template_path = Path(__file__).parent.parent / "templates" / template_name
        if not template_path.exists():
            self.logger.warning(f"YOLO README template not found: {template_path}")
            return
        content = template_path.read_text()
        content = content.replace("<<yolo_size>>", yolo_size)
        (dest_dir / "README.md").write_text(content)
        self.logger.debug(f"Copied {template_name} -> {dest_dir / 'README.md'} (size={yolo_size})")

    def _update_model_type_in_xml(self, xml_path: Path, model_type: str = "YOLO11") -> None:
        """Update the model_type value in the OpenVINO XML file."""
        try:
            tree = parse(xml_path)
            root = tree.getroot()

            for rt_info in root.findall(".//rt_info"):
                for model_info in rt_info.findall(".//model_info"):
                    for model_type_elem in model_info.findall(".//model_type"):
                        model_type_elem.set("value", model_type)

            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            self.logger.debug(f"Updated model_type to {model_type} in {xml_path}")

            model_info_dict = {}
            for rt_info in root.findall(".//rt_info"):
                for model_info in rt_info.findall(".//model_info"):
                    for child in model_info:
                        model_info_dict[child.tag] = child.attrib["value"]
            with (xml_path.parent / "config.json").open("w") as f:
                json.dump(model_info_dict, f, indent=4)

        except (OSError, ParseError) as error:
            self.logger.warning(f"Failed to update {xml_path}: {error}")
