#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
import shutil
from pathlib import Path

from defusedxml.ElementTree import ParseError, parse
from ultralytics import YOLO

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def copy_readme_template(template_name: str, dest_dir: Path, yolo_size: str) -> None:
    """Copy a README template to dest_dir/README.md, replacing <<yolo_size>>."""
    template_path = TEMPLATES_DIR / template_name
    content = template_path.read_text()
    content = content.replace("<<yolo_size>>", yolo_size)
    (dest_dir / "README.md").write_text(content)
    print(f"Copied {template_name} -> {dest_dir / 'README.md'} (size={yolo_size})")


def update_model_type_in_xml(xml_path: Path, model_type: str = "YOLO11") -> None:
    """Update the model_type value in the OpenVINO XML file."""
    try:
        tree = parse(xml_path)
        root = tree.getroot()

        # Find and update model_type in rt_info
        for rt_info in root.findall(".//rt_info"):
            for model_info in rt_info.findall(".//model_info"):
                for model_type_elem in model_info.findall(".//model_type"):
                    model_type_elem.set("value", model_type)

        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        print(f"Updated model_type to {model_type} in {xml_path}")

        model_info_dict = {}
        for rt_info in root.findall(".//rt_info"):
            for model_info in rt_info.findall(".//model_info"):
                for child in model_info:
                    model_info_dict[child.tag] = child.attrib["value"]
        with (xml_path.parent / "config.json").open("w") as f:
            json.dump(model_info_dict, f, indent=4)

    except (OSError, ParseError) as error:
        print(f"Failed to update {xml_path}: {error}")


# YOLO11 model versions
MODEL_VERSIONS = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]


def convert_yolo_models(model_versions: list[str] | None = None) -> None:
    """Convert YOLO11 model variants to OpenVINO IR folders."""
    for version in model_versions or MODEL_VERSIONS:
        print(f"Processing {version}...")
        yolo_size = version[-1]  # n, s, m, l, or x

        # Load model
        model = YOLO(f"{version}.pt")

        # Export regular OpenVINO model
        print(f"Exporting {version} to OpenVINO format...")
        model.export(format="openvino", half=True)

        # Rename output folder for regular model
        old_name = Path(f"{version}_openvino_model")
        new_name = Path(f"YOLO{version[4:]}-fp16-ov")
        if old_name.exists():
            if new_name.exists():
                shutil.rmtree(new_name)
            old_name.rename(new_name)
            print(f"Renamed {old_name} to {new_name}")

            # Update model_type in XML metadata
            xml_file = new_name / f"{version}.xml"
            if xml_file.exists():
                update_model_type_in_xml(xml_file, "YOLO11")

            # Copy README template for fp16
            copy_readme_template("README-yolo-fp16.md", new_name, yolo_size)

        # Export INT8 quantized OpenVINO model
        print(f"Exporting {version} to OpenVINO INT8 format...")
        model.export(format="openvino", int8=True, data="coco128.yaml")

        # Rename output folder for INT8 model
        old_name_int8 = Path(f"{version}_int8_openvino_model")
        new_name_int8 = Path(f"YOLO{version[4:]}-int8-ov")
        if old_name_int8.exists():
            if new_name_int8.exists():
                shutil.rmtree(new_name_int8)
            old_name_int8.rename(new_name_int8)
            print(f"Renamed {old_name_int8} to {new_name_int8}")

            # Update model_type in XML metadata
            xml_file = new_name_int8 / f"{version}.xml"
            if xml_file.exists():
                update_model_type_in_xml(xml_file, "YOLO11")

            # Copy README template for int8
            copy_readme_template("README-yolo-int8.md", new_name_int8, yolo_size)

        print(f"Completed {version}\n")


def main() -> int:
    """Run YOLO11 conversion for all configured variants."""
    convert_yolo_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
