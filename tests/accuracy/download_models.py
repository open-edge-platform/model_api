#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


async def stream_file(client, url, filename, semaphore):
    async with semaphore:
        start_time = time.time()
        total_bytes = 0
        async with client.stream("GET", url) as stream:
            with Path(filename).open("wb") as file:
                async for data in stream.aiter_bytes():
                    file.write(data)
                    total_bytes += len(data)
        end_time = time.time()
        download_time = end_time - start_time
        total_bytes /= 1024 * 1024

        speed_mbps = total_bytes / download_time if download_time > 0 else 0
        print(f"Downloaded {url} - {total_bytes:.2f} MB in {download_time:.2f}s ({speed_mbps:.2f} MB/s)")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dir",
        type=Path,
        required=True,
        help="Directory to store downloaded models and datasets",
    )
    parser.add_argument(
        "-j",
        "--json_path",
        type=Path,
        required=True,
        help="Path to the JSON file with model information",
    )
    args = parser.parse_args()

    with args.json_path.open("r") as f:
        models_data = json.load(f)

    base_path = "https://storage.geti.intel.com/geti_predict/test/"
    semaphore = asyncio.Semaphore(10)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = []
        for model_entry in models_data:
            model_name = model_entry["name"]
            download_url = base_path + model_name
            save_path = args.data_dir / model_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append(stream_file(client, download_url, save_path, semaphore))

            if model_name.endswith(".xml"):
                tasks.append(
                    stream_file(client, download_url.replace(".xml", ".bin"), save_path.with_suffix(".bin"), semaphore),
                )

        print(f"Starting download of {len(tasks)} files with max 10 concurrent downloads...")
        await asyncio.gather(*tasks)
        print(f"All {len(tasks)} files downloaded successfully!")


if __name__ == "__main__":
    asyncio.run(main())
