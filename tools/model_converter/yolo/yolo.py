import os
import shutil
import xml.etree.ElementTree as ET

from ultralytics import YOLO


def update_model_type_in_xml(xml_path, model_type="YOLO11"):
    """Update the model_type value in the OpenVINO XML file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find and update model_type in rt_info
        for rt_info in root.findall(".//rt_info"):
            for model_info in rt_info.findall(".//model_info"):
                for model_type_elem in model_info.findall(".//model_type"):
                    model_type_elem.set("value", model_type)

        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        print(f"Updated model_type to {model_type} in {xml_path}")
    except Exception as e:
        print(f"Failed to update {xml_path}: {e}")


# YOLO11 model versions
model_versions = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]

for version in model_versions:
    print(f"Processing {version}...")

    # Load model
    model = YOLO(f"{version}.pt")

    # Export regular OpenVINO model
    print(f"Exporting {version} to OpenVINO format...")
    model.export(format="openvino", half=True)

    # Rename output folder for regular model
    old_name = f"{version}_openvino_model"
    new_name = f"YOLO{version[4:]}-fp16-ov"
    if os.path.exists(old_name):
        if os.path.exists(new_name):
            shutil.rmtree(new_name)
        shutil.move(old_name, new_name)
        print(f"Renamed {old_name} to {new_name}")

        # Update model_type in XML metadata
        xml_file = os.path.join(new_name, f"{version}.xml")
        if os.path.exists(xml_file):
            update_model_type_in_xml(xml_file, "YOLO11")

    # Export INT8 quantized OpenVINO model
    print(f"Exporting {version} to OpenVINO INT8 format...")
    model.export(format="openvino", int8=True, data="coco128.yaml")

    # Rename output folder for INT8 model
    old_name_int8 = f"{version}_int8_openvino_model"
    new_name_int8 = f"YOLO{version[4:]}-int8-ov"
    if os.path.exists(old_name_int8):
        if os.path.exists(new_name_int8):
            shutil.rmtree(new_name_int8)
        shutil.move(old_name_int8, new_name_int8)
        print(f"Renamed {old_name_int8} to {new_name_int8}")

        # Update model_type in XML metadata
        xml_file = os.path.join(new_name_int8, f"{version}.xml")
        if os.path.exists(xml_file):
            update_model_type_in_xml(xml_file, "YOLO11")

    print(f"Completed {version}\n")
