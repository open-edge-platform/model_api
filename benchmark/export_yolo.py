from __future__ import annotations
from pathlib import Path
from ultralytics import YOLO


def export_openvino_int8(*, weights_path: Path, output_dir: Path, overwrite: bool = True) -> None:
	"""Export a YOLO .pt checkpoint to OpenVINO INT8 into a specific directory."""
	import shutil

	if not weights_path.exists():
		raise FileNotFoundError(f"Weights not found: {weights_path}")

	model = YOLO(str(weights_path))
	export_path = model.export(
		format="openvino",
		int8=True,
	)

	default_export_dir = Path(export_path)
	if default_export_dir.is_file():
		default_export_dir = default_export_dir.parent
	
	if default_export_dir.resolve() != output_dir.resolve():
		if output_dir.exists() and overwrite:
			shutil.rmtree(output_dir)
		shutil.move(str(default_export_dir), str(output_dir))


def main() -> None:
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    output_base = Path.cwd()

    exports: dict[str, str] = {
		"yolo11n.pt": "YOLOv11-Detection-n-int8-ov",
		"yolo11s.pt": "YOLOv11-Detection-s-int8-ov",
		"yolo11m.pt": "YOLOv11-Detection-m-int8-ov",
	}

    for weights_name, out_name in exports.items():
        weights_path = weights_dir / weights_name

        if not weights_path.exists():
            print(f"Downloading {weights_name}...")
            model = YOLO(str(weights_name))
            downloaded_path = Path(weights_name)
            if downloaded_path.exists():
                downloaded_path.rename(weights_path)

        print(f"Exporting {weights_path.name} -> {out_name} (OpenVINO INT8)")
        export_openvino_int8(weights_path=weights_path, output_dir=output_base / out_name, overwrite=True)


if __name__ == "__main__":
	main()

