# OpenVINO Model API and Model Converter

## Repository layout

This repository contains two independently installable Python subprojects:

- [model_api](model_api) — the inference library published as [openvino-model-api](https://pypi.org/project/openvino-model-api/) with a minimal runtime dependency set.
- [model_converter](model_converter) — conversion tooling published as `openvino-model-converter`, with conversion-time dependencies such as PyTorch, TorchVision, OpenVINO, ONNX, and NNCF.

Each subproject owns its own `pyproject.toml` and `uv.lock`.
Shared repository files, including this `README.md`, `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, and CI workflows, remain at the repository root.
