#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "earthengine-api==1.4.0",
#     "omegaconf==2.3.0",
#     "Pillow",
#     "requests==2.32.5",
#     "tqdm",
#     "transformers",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Extract GRAFT satellite embeddings for camera-trap locations."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import submitit
import torch
from PIL import Image
from tqdm import tqdm

from mmocc.config import cache_path
from mmocc.graft_utils import (
    GRAFT_DEFAULT_IMAGE_SIZE,
    GRAFT_EMBED_DIM,
    GraftConfig,
    build_graft_transform,
    load_graft_model,
)
from mmocc.rs_graft import (
    DEFAULT_CLOUD_PERCENT,
    DEFAULT_TIME_WINDOW_DAYS,
    SENTINEL_IMAGE_SIZE,
    SENTINEL_PIXEL_SIZE_METERS,
    fetch_sentinel_patch,
)
from mmocc.utils import get_submitit_executor


@dataclass(frozen=True)
class ExtractSettings:
    project: str
    window_days: int
    cloud_percent: float
    image_size: int
    pixel_size_meters: float


def _fetch_patch(
    idx: int, row: pd.Series, settings: ExtractSettings
) -> tuple[int, np.ndarray | None]:
    timestamp = row["Date_Time"]
    if hasattr(timestamp, "to_pydatetime"):
        ts = timestamp.to_pydatetime().isoformat()
    else:
        ts = datetime.fromisoformat(str(timestamp)).isoformat()
    try:
        patch = fetch_sentinel_patch(
            latitude=float(row["Latitude"]),
            longitude=float(row["Longitude"]),
            timestamp=ts,
            window_days=settings.window_days,
            size=settings.image_size,
            pixel_size_meters=settings.pixel_size_meters,
            cloud_percent=settings.cloud_percent,
            project=settings.project,
        )
        return idx, patch
    except Exception as exc:
        print(f"Failed to fetch Sentinel patch for index {idx}: {exc}")
        return idx, None


def extract_graft_features(
    max_sites: int | None = None,
    batch_size: int = 32,
    download_workers: int = 4,
    project: str = "zorrilla",
    window_days: int = DEFAULT_TIME_WINDOW_DAYS,
    cloud_percent: float = DEFAULT_CLOUD_PERCENT,
    image_size: int = SENTINEL_IMAGE_SIZE,
    pixel_size_meters: float = SENTINEL_PIXEL_SIZE_METERS,
    skip_existing: bool = True,
):
    df = pd.read_pickle(cache_path / "wi_blank_images.pkl").reset_index(drop=True)
    if max_sites is not None:
        df = df.head(max_sites).copy()

    output_path = cache_path / "features"
    output_path.mkdir(parents=True, exist_ok=True)
    feature_file = output_path / "wi_blank_sat_features_graft.npy"
    id_file = output_path / "wi_blank_sat_features_graft_ids.npy"

    if skip_existing and feature_file.exists() and id_file.exists():
        print(f"GRAFT sat features already exist at {feature_file}, skipping.")
        return

    loc_ids = df["loc_id"].to_numpy()
    features = np.full((len(df), GRAFT_EMBED_DIM), np.nan, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_graft_model(GraftConfig(image_size=image_size), device=device)
    transform = build_graft_transform(image_size=image_size)

    settings = ExtractSettings(
        project=project,
        window_days=window_days,
        cloud_percent=cloud_percent,
        image_size=image_size,
        pixel_size_meters=pixel_size_meters,
    )

    num_rows = len(df)
    for start in tqdm(range(0, num_rows, batch_size), desc="Batches"):
        batch_df = df.iloc[start : start + batch_size]
        patches: list[np.ndarray] = []
        indices: list[int] = []
        missing = 0

        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            futures = [
                executor.submit(_fetch_patch, idx, row, settings)
                for idx, row in batch_df.iterrows()
            ]
            for future in as_completed(futures):
                idx, patch = future.result()
                if patch is None:
                    missing += 1
                    continue
                patches.append(patch)
                indices.append(idx)

        if patches:
            images = [
                transform(Image.fromarray(patch.astype(np.uint8)))
                for patch in patches
            ]
            batch_tensor = torch.stack(images, dim=0).to(device)
            with torch.inference_mode():
                embeds = model.forward_features(batch_tensor).detach().cpu().numpy()
            for idx, emb in zip(indices, embeds):
                features[idx] = emb.astype(np.float32, copy=False)

        if missing:
            print(f"Warning: {missing} Sentinel patches missing in this batch.")

    np.save(feature_file, features)
    np.save(id_file, loc_ids)
    print(f"Wrote GRAFT sat features to {feature_file}")


def main(
    max_sites: int | None = None,
    batch_size: int = 32,
    download_workers: int = 4,
    project: str = "zorrilla",
    window_days: int = DEFAULT_TIME_WINDOW_DAYS,
    cloud_percent: float = DEFAULT_CLOUD_PERCENT,
    image_size: int = GRAFT_DEFAULT_IMAGE_SIZE,
    pixel_size_meters: float = SENTINEL_PIXEL_SIZE_METERS,
    skip_existing: bool = True,
):
    executor = get_submitit_executor("extract_graft_features")
    executor.update_parameters(
        slurm_mem="256G",
        slurm_additional_parameters=dict(gpus=1),
    )

    job = executor.submit(
        extract_graft_features,
        max_sites=max_sites,
        batch_size=batch_size,
        download_workers=download_workers,
        project=project,
        window_days=window_days,
        cloud_percent=cloud_percent,
        image_size=image_size,
        pixel_size_meters=pixel_size_meters,
        skip_existing=skip_existing,
    )

    for completed in tqdm(
        submitit.helpers.as_completed([job]), total=1, desc="Jobs", leave=False
    ):
        completed.result()


if __name__ == "__main__":
    fire.Fire(main)
