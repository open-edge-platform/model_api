#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urljoin

import httpx


async def stream_file(client, url, filename):
    async with client.stream("GET", url) as stream:
        with Path(filename).open("wb") as file:
            async for data in stream.aiter_bytes():
                file.write(data)


async def download_model_from_json_entry(client, base_url, data_dir, model_entry):
    """Download a model based on a JSON entry."""
    model_path = model_entry["name"]
    model_url = urljoin(base_url, model_path)
    local_path = data_dir / model_path
    
    await stream_file(client, model_url, local_path)


async def download_models_batch(client, base_url, data_dir, model_entries, batch_size=10):
    """Download models in batches to avoid overwhelming the server."""
    for i in range(0, len(model_entries), batch_size):
        batch = model_entries[i:i + batch_size]
        tasks = [
            download_model_from_json_entry(client, base_url, data_dir, entry)
            for entry in batch
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        print(f"Completed batch {i // batch_size + 1}/{(len(model_entries) + batch_size - 1) // batch_size}")


async def main():
    parser = argparse.ArgumentParser(
        description="Download models specified in a JSON configuration file"
    )
    parser.add_argument(
        "-j",
        "--json_file",
        type=Path,
        required=True,
        help="Path to JSON file containing model configurations",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=Path,
        required=True,
        help="Directory to store downloaded models",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://storage.geti.intel.com/geti_predict/test/",
        help="Base URL for model downloads (default: %(default)s)",
    )
    
    args = parser.parse_args()

    # Ensure base URL ends with /
    base_url = args.base_url.rstrip("/") + "/"
    
    # Load JSON configuration
    try:
        with args.json_file.open("r") as f:
            model_configs = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading JSON file {args.json_file}: {e}")
        return

    print(f"Will download {len(model_configs)} models to {args.data_dir}")
    
    # Create data directory
    args.data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download models
    async with httpx.AsyncClient(timeout=60.0) as client:
        await download_models_batch(
            client, base_url, args.data_dir, model_configs, 10
        )
    
    print("Download completed!")


if __name__ == "__main__":
    asyncio.run(main())
