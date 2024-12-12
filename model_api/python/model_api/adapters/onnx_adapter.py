#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations  # TODO: remove when Python3.9 support is dropped

import sys
from functools import partial, reduce
from typing import Any, Callable

import numpy as np

from .utils import INTERPOLATION_TYPES, RESIZE_TYPES, InputTransform

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    onnxrt_absent = False
except ImportError:
    onnxrt_absent = True

from .inference_adapter import InferenceAdapter, Metadata
from .utils import Layout, get_rt_info_from_dict, load_parameters_from_onnx


class ONNXRuntimeAdapter(InferenceAdapter):
    """This inference adapter allows running ONNX models via ONNXRuntime.
    The adapter has limited functionality: it supports only image models
    generated by OpenVINO training extensions (OTX: https://github.com/openvinotoolkit/training_extensions/).
    Each onnx file generated by OTX contains ModelAPI-style metadata, which is used for
    configuring a particular model acting on top of it.
    Models scope is limited to `SSD`, `MaskRCNNModel`, `SegmentationModel`,
    and `ClassificationModel` wrappers.
    Also, this adapter doesn't provide asynchronous inference functionality and model reshaping.
    """

    def __init__(self, model: str, ort_options: dict = {}):
        """Args:
        model (str): Filename or serialized ONNX model in a byte string.
        ort_options (dict): parameters that will be forwarded to onnxruntime.InferenceSession
        """
        loaded_model = onnx.load(model)

        inferred_model = SymbolicShapeInference.infer_shapes(
            in_mp=loaded_model,
            int_max=int(sys.maxsize / 2),
            auto_merge=False,
            guess_output_rank=False,
            verbose=False,
        )

        self.session = ort.InferenceSession(
            inferred_model.SerializeToString(),
            **ort_options,
        )
        self.model = inferred_model
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.onnx_metadata = load_parameters_from_onnx(inferred_model)
        self.preprocessor = lambda arg: arg

    def get_input_layers(self):
        inputs = {}

        for input in self.session.get_inputs():
            shape = get_shape_from_onnx(input.shape)
            inputs[input.name] = Metadata(
                {input.name},
                shape,
                Layout.from_shape(shape),
                _onnx2ov_precision.get(input.type, input.type),
            )

        return inputs

    def get_output_layers(self):
        outputs = {}
        for output in self.session.get_outputs():
            shape = get_shape_from_onnx(output.shape)
            outputs[output.name] = Metadata(
                {output.name},
                shape=shape,
                precision=_onnx2ov_precision.get(output.type, output.type),
            )

        return outputs

    def infer_sync(self, dict_data):
        inputs = {}
        for input in self.session.get_inputs():
            if len(input.shape) == 4:
                preprocessed_input = self.preprocessor(dict_data[input.name])
            if dict_data[input.name].dtype != _onnx2np_precision[input.type]:
                inputs[input.name] = ort.OrtValue.ortvalue_from_numpy(
                    preprocessed_input.astype(_onnx2np_precision[input.type]),
                )
            else:
                inputs[input.name] = ort.OrtValue.ortvalue_from_numpy(
                    preprocessed_input,
                )
        raw_result = self.session.run(self.output_names, inputs)

        named_raw_result = {}
        for i, data in enumerate(raw_result):
            named_raw_result[self.output_names[i]] = data

        return named_raw_result

    def infer_async(self, dict_data, callback_data):
        raise NotImplementedError

    def set_callback(self, callback_fn: Callable):
        self.callback_fn = callback_fn

    def is_ready(self):
        return True

    def load_model(self):
        pass

    def await_all(self):
        pass

    def await_any(self):
        pass

    def get_raw_result(self, infer_result: dict):
        pass

    def embed_preprocessing(
        self,
        layout: str,
        resize_mode: str,
        interpolation_mode: str,
        target_shape: tuple[int, ...],
        pad_value: int,
        dtype: type = int,
        brg2rgb: bool = False,
        mean: list[Any] | None = None,
        scale: list[Any] | None = None,
        input_idx: int = 0,
    ):
        """
        Adds external preprocessing steps done before ONNX model execution.
        """
        preproc_funcs = [np.squeeze]
        if resize_mode != "crop":
            if resize_mode == "fit_to_window_letterbox":
                resize_fn = partial(
                    RESIZE_TYPES[resize_mode],
                    size=target_shape,
                    interpolation=INTERPOLATION_TYPES[interpolation_mode],
                    pad_value=pad_value,
                )
            else:
                resize_fn = partial(
                    RESIZE_TYPES[resize_mode],
                    size=target_shape,
                    interpolation=INTERPOLATION_TYPES[interpolation_mode],
                )
        else:
            resize_fn = partial(RESIZE_TYPES[resize_mode], size=target_shape)
        preproc_funcs.append(resize_fn)
        input_transform = InputTransform(brg2rgb, mean, scale)
        preproc_funcs.append(input_transform.__call__)
        preproc_funcs.append(partial(change_layout, layout=layout))

        self.preprocessor = reduce(
            lambda f, g: lambda x: f(g(x)),
            reversed(preproc_funcs),
        )

    def get_model(self):
        """Return a reference to the ONNXRuntime session."""
        return self.model

    def reshape_model(self, new_shape):
        """ "Not supported by ONNX adapter."""
        raise NotImplementedError

    def get_rt_info(self, path):
        """
        Returns an attribute stored in model info.

        Args:
            path (list[str]): a sequence of tag names leading to the attribute.

        Returns:
            Any: a value stored under corresponding tag sequence.
        """
        return get_rt_info_from_dict(self.onnx_metadata, path)

    def update_model_info(self, model_info: dict[str, Any]):
        for item in model_info:
            meta = self.model.metadata_props.add()
            attr_path = "model_info " + item
            meta.key = attr_path.strip()
            if isinstance(model_info[item], list):
                meta.value = " ".join(str(x) for x in model_info[item])
            else:
                meta.value = str(model_info[item])

    def save_model(self, path: str, weights_path: str | None = None, version: str | None = None):
        """
        Serializes model to the filesystem.

        Args:
            path (str): paths to save .onnx file.
            weights_path (str | None): not used by ONNX adapter.
            version (str | None): not used by ONNX adapter.
        """
        onnx.save(self.model, path)


_onnx2ov_precision = {
    "tensor(float)": "f32",
}

_onnx2np_precision = {
    "tensor(float)": np.float32,
}


def get_shape_from_onnx(onnx_shape):
    for i, item in enumerate(onnx_shape):
        if isinstance(item, str):
            onnx_shape[i] = -1
    return tuple(onnx_shape)


def change_layout(image, layout):
    """Changes the input image layout to fit the layout of the model input layer.

    Args:
        inputs (ndarray): a single image as 3D array in HWC layout

    Returns:
        - the image with layout aligned with the model layout
    """
    if "CHW" in layout:
        image = image.transpose((2, 0, 1))  # HWC->CHW
        image = image.reshape((1, *image.shape))
    return image
