#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "Pillow",
#     "tqdm",
#     "transformers",
#     "setuptools<81"
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Extract GRAFT satellite embeddings from pre-downloaded imagery.

Uses the same pre-downloaded PNG images (keyed by loc_id) that step 08
(VisDiff) consumes, rather than fetching patches from Earth Engine
on-the-fly.  Supports both Sentinel-2 and NAIP imagery sources.

GRAFT preprocessing follows the original paper (Mall et al., 2023):
  1. Resize to 224×224
  2. Normalize with CLIP ViT-B/16 statistics
  3. Extract features using the GRAFT vision encoder
  4. L2-normalize the embeddings
"""

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
from mmocc.utils import get_submitit_executor

# Maps imagery_source → PNG directory (mirrors step 08)
IMAGERY_SOURCE_PNG_DIRS: dict[str, Path] = {
    "sentinel": cache_path / "sat_wi_rgb_images_png",
    "naip": cache_path / "naip_wi_images_png",
}

# Maps imagery_source → backbone name used in feature filenames.
# Must match what load_data(sat_backbone_name=...) expects.
IMAGERY_SOURCE_BACKBONE: dict[str, str] = {
    "sentinel": "graft",
    "naip": "graft_naip",
}


def extract_graft_features(
    imagery_source: str = "sentinel",
    image_size: int = GRAFT_DEFAULT_IMAGE_SIZE,
    batch_size: int = 32,
    skip_existing: bool = True,
):
    """Extract GRAFT embeddings from pre-downloaded PNG images.

    Features are saved aligned with ``wi_blank_images.pkl`` row order —
    matching the convention used by step 05 — so that ``load_data()``
    can load them with the appropriate ``sat_backbone_name``.
    """
    png_dir = IMAGERY_SOURCE_PNG_DIRS.get(imagery_source)
    if png_dir is None:
        raise ValueError(
            f"Unknown imagery_source '{imagery_source}'. "
            f"Choose from: {sorted(IMAGERY_SOURCE_PNG_DIRS)}"
        )
    backbone_name = IMAGERY_SOURCE_BACKBONE[imagery_source]

    # Load the same dataframe that step 05 uses, preserving row order
    df = pd.read_pickle(cache_path / "wi_blank_images.pkl")

    output_path = cache_path / "features"
    output_path.mkdir(parents=True, exist_ok=True)
    feature_file = output_path / f"wi_blank_sat_features_{backbone_name}.npy"
    id_file = output_path / f"wi_blank_sat_features_{backbone_name}_ids.npy"

    if skip_existing and feature_file.exists() and id_file.exists():
        print(f"GRAFT {imagery_source} features already exist at {feature_file}, skipping.")
        return

    loc_ids = df["loc_id"].to_numpy()
    features = np.full((len(df), GRAFT_EMBED_DIM), np.nan, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_graft_model(GraftConfig(image_size=image_size), device=device)
    transform = build_graft_transform(image_size=image_size)

    # Build a lookup of loc_id → PNG path, checking existence once
    unique_locs = df["loc_id"].unique()
    png_paths: dict[str, Path] = {}
    missing_count = 0
    for lid in unique_locs:
        p = png_dir / f"{lid}.png"
        if p.exists():
            png_paths[lid] = p
        else:
            missing_count += 1

    if missing_count:
        print(
            f"Warning: {missing_count}/{len(unique_locs)} loc_ids have no "
            f"{imagery_source} PNG in {png_dir}"
        )

    # Encode each unique loc_id once, then broadcast to all rows
    locs_to_encode = [lid for lid in unique_locs if lid in png_paths]
    loc_id_embeddings: dict[str, np.ndarray] = {}

    for start in tqdm(
        range(0, len(locs_to_encode), batch_size),
        desc=f"GRAFT ({imagery_source})",
    ):
        batch_lids = locs_to_encode[start : start + batch_size]
        images: list[torch.Tensor] = []
        valid_lids: list[str] = []

        for lid in batch_lids:
            try:
                img = Image.open(png_paths[lid]).convert("RGB")
                images.append(transform(img))
                valid_lids.append(lid)
            except Exception as exc:
                print(f"Failed to load {png_paths[lid]}: {exc}")

        if images:
            batch_tensor = torch.stack(images, dim=0).to(device)
            with torch.inference_mode():
                embeds = model.forward_features(batch_tensor).detach().cpu().numpy()
            for lid, emb in zip(valid_lids, embeds):
                loc_id_embeddings[lid] = emb.astype(np.float32, copy=False)

    # Map embeddings back to all rows (same ordering as wi_blank_images.pkl)
    for i, lid in enumerate(loc_ids):
        if lid in loc_id_embeddings:
            features[i] = loc_id_embeddings[lid]

    valid_count = int(np.isfinite(features[:, 0]).sum())
    print(
        f"Extracted GRAFT features for {valid_count}/{len(df)} rows "
        f"({len(loc_id_embeddings)} unique locs)"
    )

    np.save(feature_file, features)
    np.save(id_file, loc_ids)
    print(f"Wrote GRAFT {imagery_source} features to {feature_file}")


def main(
    imagery_source: str = "sentinel",
    image_size: int = GRAFT_DEFAULT_IMAGE_SIZE,
    batch_size: int = 32,
    skip_existing: bool = False,
):
    executor = get_submitit_executor("extract_graft_features")
    executor.update_parameters(
        slurm_mem="256G",
        slurm_additional_parameters=dict(gpus=1),
    )

    job = executor.submit(
        extract_graft_features,
        imagery_source=imagery_source,
        image_size=image_size,
        batch_size=batch_size,
        skip_existing=skip_existing,
    )

    for completed in tqdm(
        submitit.helpers.as_completed([job]), total=1, desc="Jobs", leave=False
    ):
        completed.result()


if __name__ == "__main__":
    fire.Fire(main)
