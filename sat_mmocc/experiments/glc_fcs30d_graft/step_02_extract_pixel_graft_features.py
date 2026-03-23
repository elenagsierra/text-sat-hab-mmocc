#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "Pillow",
#     "fire",
#     "numpy",
#     "pandas",
#     "torch",
#     "tqdm",
#     "transformers",
#     "setuptools<81"
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Extract pixel-level GRAFT patch-token embeddings for cached sat_mmocc images."""

from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from mmocc.graft_utils import GraftConfig, build_graft_transform, load_graft_model
from sat_mmocc.experiments.glc_fcs30d_graft.utils import (
    DEFAULT_GLC_YEAR,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IMAGERY_SOURCE,
    get_run_dir,
    reshape_patch_tokens,
)


def extract_pixel_features(
    imagery_source: str = DEFAULT_IMAGERY_SOURCE,
    year: int = DEFAULT_GLC_YEAR,
    checkpoint_level: str = "pixel",
    batch_size: int = 32,
    image_size: int = DEFAULT_IMAGE_SIZE,
    max_sites: int | None = None,
    skip_existing: bool = True,
) -> Path:
    if checkpoint_level.strip().lower() != "pixel":
        raise ValueError(
            "This experiment is intended for pixel-level GRAFT checkpoints. "
            "Use checkpoint_level='pixel'."
        )

    run_dir = get_run_dir(
        imagery_source=imagery_source,
        year=year,
        checkpoint_level=checkpoint_level,
    )
    label_manifest_path = run_dir / "glc_label_manifest.csv"
    if not label_manifest_path.exists():
        raise FileNotFoundError(
            f"Label manifest not found at {label_manifest_path}. "
            "Run step_01_download_glc_fcs30d_labels.py first."
        )

    manifest_df = pd.read_csv(label_manifest_path)
    if max_sites is not None:
        manifest_df = manifest_df.head(max_sites).copy()

    feature_dir = run_dir / "pixel_token_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    transform = build_graft_transform(image_size=image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_graft_model(
        GraftConfig(
            imagery_family=imagery_source,
            checkpoint_level="pixel",
            image_size=image_size,
        ),
        device=device,
    )

    rows: list[dict[str, object]] = []
    pending_rows: list[pd.Series] = []
    for _, row in manifest_df.iterrows():
        loc_id = str(row["loc_id"])
        feature_path = feature_dir / f"{loc_id}.npy"
        if skip_existing and feature_path.exists():
            feature_grid = np.load(feature_path, allow_pickle=False)
            rows.append(
                {
                    "loc_id": loc_id,
                    "feature_path": str(feature_path),
                    "grid_size": int(feature_grid.shape[0]),
                    "embed_dim": int(feature_grid.shape[-1]),
                }
            )
        else:
            pending_rows.append(row)

    for start in tqdm(
        range(0, len(pending_rows), batch_size),
        total=(len(pending_rows) + batch_size - 1) // batch_size,
        desc="Extracting pixel-level GRAFT features",
    ):
        batch_rows = pending_rows[start : start + batch_size]
        images: list[torch.Tensor] = []
        metadata: list[tuple[str, Path]] = []
        for row in batch_rows:
            loc_id = str(row["loc_id"])
            image_path = Path(row["image_path"])
            try:
                image = Image.open(image_path).convert("RGB")
                images.append(transform(image))
                metadata.append((loc_id, feature_dir / f"{loc_id}.npy"))
            except Exception as exc:
                print(f"Failed to load {image_path}: {exc}")

        if not images:
            continue

        batch_tensor = torch.stack(images, dim=0).to(device)
        with torch.inference_mode():
            token_embeddings = model.forward(batch_tensor).detach().cpu().numpy()

        for (loc_id, feature_path), tokens in zip(metadata, token_embeddings, strict=True):
            feature_grid = reshape_patch_tokens(tokens).astype(np.float16, copy=False)
            np.save(feature_path, feature_grid)
            rows.append(
                {
                    "loc_id": loc_id,
                    "feature_path": str(feature_path),
                    "grid_size": int(feature_grid.shape[0]),
                    "embed_dim": int(feature_grid.shape[-1]),
                }
            )

    features_manifest = pd.DataFrame(rows).sort_values("loc_id").reset_index(drop=True)
    manifest_path = run_dir / "pixel_feature_manifest.csv"
    features_manifest.to_csv(manifest_path, index=False)
    print(f"Wrote feature manifest to {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    fire.Fire(extract_pixel_features)
