#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for model_converter.yolo module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestCopyReadmeTemplate:
    """Tests for copy_readme_template function."""

    def test_copies_and_replaces_placeholder(self, tmp_path):
        """copy_readme_template replaces <<yolo_size>> in template."""
        from model_converter.yolo.yolo import copy_readme_template

        # Create a template in the actual templates dir (we need to mock it)
        template_content = "# YOLO11<<yolo_size>> Model\nSize: <<yolo_size>>"
        dest_dir = tmp_path / "output"
        dest_dir.mkdir()

        with patch.object(Path, "read_text", return_value=template_content):
            copy_readme_template("README-yolo-fp16.md", dest_dir, "n")

        readme = dest_dir / "README.md"
        assert readme.exists()
        content = readme.read_text()
        assert "YOLO11n Model" in content
        assert "Size: n" in content
        assert "<<yolo_size>>" not in content


class TestUpdateModelTypeInXml:
    """Tests for update_model_type_in_xml function."""

    def test_updates_model_type(self, tmp_path):
        """update_model_type_in_xml updates model_type value in XML."""
        from model_converter.yolo.yolo import update_model_type_in_xml

        # Create a valid OpenVINO XML structure
        xml_content = """<?xml version='1.0' encoding='utf-8'?>
<net name="test" version="11">
    <rt_info>
        <model_info>
            <model_type value="OldType"/>
            <labels value="person car"/>
        </model_info>
    </rt_info>
</net>"""
        xml_path = tmp_path / "model.xml"
        xml_path.write_text(xml_content)

        update_model_type_in_xml(xml_path, "YOLO11")

        # Verify XML was updated
        from defusedxml.ElementTree import parse

        tree = parse(xml_path)
        root = tree.getroot()
        model_type_elem = root.find(".//rt_info/model_info/model_type")
        assert model_type_elem is not None
        assert model_type_elem.get("value") == "YOLO11"

        # Verify config.json was created
        config_json = tmp_path / "config.json"
        assert config_json.exists()
        config = json.loads(config_json.read_text())
        assert config["model_type"] == "YOLO11"
        assert config["labels"] == "person car"

    def test_handles_parse_error(self, tmp_path, capsys):
        """update_model_type_in_xml handles invalid XML gracefully."""
        from model_converter.yolo.yolo import update_model_type_in_xml

        xml_path = tmp_path / "bad.xml"
        xml_path.write_text("not valid xml <<<>>>")

        # Should not raise
        update_model_type_in_xml(xml_path, "YOLO11")

        captured = capsys.readouterr()
        assert "Failed to update" in captured.out

    def test_handles_file_not_found(self, tmp_path, capsys):
        """update_model_type_in_xml handles missing file gracefully."""
        from model_converter.yolo.yolo import update_model_type_in_xml

        xml_path = tmp_path / "nonexistent.xml"

        update_model_type_in_xml(xml_path, "YOLO11")

        captured = capsys.readouterr()
        assert "Failed to update" in captured.out


class TestConvertYoloModels:
    """Tests for convert_yolo_models function."""

    @patch("model_converter.yolo.yolo.copy_readme_template")
    @patch("model_converter.yolo.yolo.update_model_type_in_xml")
    @patch("model_converter.yolo.yolo.YOLO")
    @patch("shutil.rmtree")
    def test_convert_single_model(self, mock_rmtree, mock_yolo_class, mock_update_xml, mock_copy_readme, tmp_path):
        """convert_yolo_models converts a single YOLO variant."""
        from model_converter.yolo.yolo import convert_yolo_models

        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Mock that the output folders exist after export
        with (
            patch("model_converter.yolo.yolo.Path"),
            patch.object(Path, "exists", return_value=False),
        ):
            convert_yolo_models(["yolo11n"])

        mock_yolo_class.assert_called_once_with("yolo11n.pt")
        assert mock_model.export.call_count == 2

    @patch("model_converter.yolo.yolo.copy_readme_template")
    @patch("model_converter.yolo.yolo.update_model_type_in_xml")
    @patch("model_converter.yolo.yolo.YOLO")
    def test_convert_with_existing_output(
        self,
        mock_yolo_class,
        mock_update_xml,
        mock_copy_readme,
        tmp_path,
        monkeypatch,
    ):
        """convert_yolo_models handles existing output directories."""
        from model_converter.yolo.yolo import convert_yolo_models

        monkeypatch.chdir(tmp_path)

        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Create the output dirs that YOLO export would create
        fp16_dir = tmp_path / "yolo11n_openvino_model"
        fp16_dir.mkdir()
        (fp16_dir / "yolo11n.xml").write_text("<net/>")

        int8_dir = tmp_path / "yolo11n_int8_openvino_model"
        int8_dir.mkdir()
        (int8_dir / "yolo11n.xml").write_text("<net/>")

        # Create target dirs that already exist (to test rmtree path)
        (tmp_path / "YOLO11n-fp16-ov").mkdir()
        (tmp_path / "YOLO11n-int8-ov").mkdir()

        with (
            patch("model_converter.yolo.yolo.update_model_type_in_xml"),
            patch("model_converter.yolo.yolo.copy_readme_template"),
        ):
            convert_yolo_models(["yolo11n"])


class TestYoloMain:
    """Tests for yolo main function."""

    @patch("model_converter.yolo.yolo.convert_yolo_models")
    def test_main_returns_zero(self, mock_convert):
        """main() calls convert_yolo_models and returns 0."""
        from model_converter.yolo.yolo import main

        result = main()
        assert result == 0
        mock_convert.assert_called_once_with()
