#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import asyncio
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import httpx


async def download_images(data_dir):
    async with httpx.AsyncClient(timeout=20.0) as client:
        COCO128_URL = "https://ultralytics.com/assets/coco128.zip"
        archive = await client.get(COCO128_URL, follow_redirects=True)
        with ZipFile(BytesIO(archive.content)) as zfile:
            zfile.extractall(data_dir)
        image = await client.get(
            "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00007.jpg",
        )
        with Path(data_dir / "BloodImage_00007.jpg").open("wb") as im:
            im.write(image.content)


async def stream_file(client, url, filename):
    async with client.stream("GET", url) as stream:
        with Path(filename).open("wb") as file:
            async for data in stream.aiter_bytes():
                file.write(data)


async def download_otx_model(client, otx_models_dir, model_name, format="xml"):
    if format == "onnx":
        await stream_file(
            client,
            f"https://storage.geti.intel.com/geti_predict/test/otx_models/{model_name}/model.onnx",
            f"{otx_models_dir}/{model_name}.onnx",
        )
    else:
        await asyncio.gather(
            stream_file(
                client,
                f"https://storage.geti.intel.com/geti_predict/test/otx_models/{model_name}/openvino.xml",
                f"{otx_models_dir}/{model_name}.xml",
            ),
            stream_file(
                client,
                f"https://storage.geti.intel.com/geti_predict/test/otx_models/{model_name}/openvino.bin",
                f"{otx_models_dir}/{model_name}.bin",
            ),
        )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dir",
        type=Path,
        required=True,
        help="Directory to store downloaded models and datasets",
    )
    args = parser.parse_args()

    otx_models_dir = args.data_dir / "otx_models"
    otx_models_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=20.0) as client:
        await asyncio.gather(
            download_images(args.data_dir),
            download_otx_model(client, otx_models_dir, "mlc_mobilenetv3_large_voc"),
            download_otx_model(client, otx_models_dir, "mlc_efficient_b0_voc"),
            download_otx_model(client, otx_models_dir, "mlc_efficient_v2s_voc"),
            download_otx_model(client, otx_models_dir, "det_mobilenetv2_atss_bccd"),
            download_otx_model(
                client,
                otx_models_dir,
                "det_mobilenetv2_atss_bccd_onnx",
                "onnx",
            ),
            download_otx_model(client, otx_models_dir, "cls_mobilenetv3_large_cars"),
            download_otx_model(
                client,
                otx_models_dir,
                "cls_mobilenetv3_large_cars",
                "onnx",
            ),
            download_otx_model(client, otx_models_dir, "cls_efficient_b0_cars"),
            download_otx_model(client, otx_models_dir, "cls_efficient_v2s_cars"),
            download_otx_model(client, otx_models_dir, "Lite-hrnet-18"),
            download_otx_model(client, otx_models_dir, "Lite-hrnet-18_mod2"),
            download_otx_model(client, otx_models_dir, "Lite-hrnet-s_mod2"),
            download_otx_model(client, otx_models_dir, "Lite-hrnet-s_mod2", "onnx"),
            download_otx_model(client, otx_models_dir, "Lite-hrnet-x-mod3"),
            download_otx_model(
                client,
                otx_models_dir,
                "is_efficientnetb2b_maskrcnn_coco_reduced",
            ),
            download_otx_model(
                client,
                otx_models_dir,
                "is_efficientnetb2b_maskrcnn_coco_reduced_onnx",
                "onnx",
            ),
            download_otx_model(
                client,
                otx_models_dir,
                "is_resnet50_maskrcnn_coco_reduced",
            ),
            download_otx_model(client, otx_models_dir, "mobilenet_v3_large_hc_cf"),
            download_otx_model(
                client,
                otx_models_dir,
                "classification_model_with_xai_head",
            ),
            download_otx_model(client, otx_models_dir, "detection_model_with_xai_head"),
            download_otx_model(
                client,
                otx_models_dir,
                "segmentation_model_with_xai_head",
            ),
            download_otx_model(client, otx_models_dir, "maskrcnn_model_with_xai_head"),
            download_otx_model(client, otx_models_dir, "maskrcnn_xai_tiling"),
            download_otx_model(client, otx_models_dir, "tile_classifier"),
            download_otx_model(client, otx_models_dir, "anomaly_padim_bottle_mvtec"),
            download_otx_model(client, otx_models_dir, "anomaly_stfpm_bottle_mvtec"),
            download_otx_model(client, otx_models_dir, "deit-tiny"),
            download_otx_model(
                client,
                otx_models_dir,
                "cls_efficient_b0_shuffled_outputs",
            ),
            download_otx_model(client, otx_models_dir, "action_cls_xd3_kinetic"),
            download_otx_model(client, otx_models_dir, "sam_vit_b_zsl_encoder"),
            download_otx_model(client, otx_models_dir, "sam_vit_b_zsl_decoder"),
            download_otx_model(client, otx_models_dir, "rtmpose_tiny"),
            download_otx_model(client, otx_models_dir, "segnext_t_tiling"),
        )


if __name__ == "__main__":
    asyncio.run(main())
