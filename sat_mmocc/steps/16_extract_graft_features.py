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

Uses pre-downloaded PNG images rather than fetching patches from Earth
Engine on-the-fly. Standard Sentinel-2 / NAIP sources are keyed by
``loc_id`` while the ``*_v_graft`` sources are paired per camera-trap row
via ``camera_satellite_pairings_v_graft.csv``.

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
    resolve_graft_checkpoint,
)
from mmocc.utils import get_submitit_executor

# Maps imagery_source → PNG directory (mirrors step 08)
IMAGERY_SOURCE_PNG_DIRS: dict[str, Path] = {
    "sentinel": cache_path / "sat_wi_rgb_images_png",
    "naip": cache_path / "naip_wi_images_png",
    "sentinel_v_graft": cache_path / "sentinel_v_graft_images_png",
    "naip_v_graft": cache_path / "naip_v_graft_images_png",
}

# Base backbone names used in feature filenames. The checkpoint level may add
# a suffix so image-level and pixel-level features can coexist.
IMAGERY_SOURCE_BACKBONE_BASE: dict[str, str] = {
    "sentinel": "graft",
    "naip": "graft_naip",
    "sentinel_v_graft": "graft_sentinel_v_graft",
    "naip_v_graft": "graft_naip_v_graft",
}

PAIRING_CSV = cache_path / "camera_satellite_pairings_v_graft.csv"
V_GRAFT_PATH_COLUMNS: dict[str, str] = {
    "sentinel_v_graft": "sentinel_image_path",
    "naip_v_graft": "naip_image_path",
}
V_GRAFT_EXISTS_COLUMNS: dict[str, str] = {
    "sentinel_v_graft": "sentinel_exists",
    "naip_v_graft": "naip_exists",
}


def _build_loc_id_png_lookup(df: pd.DataFrame, png_dir: Path) -> dict[str, Path]:
    """Return available ``loc_id``-keyed PNGs for standard imagery sources."""

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
            f"Warning: {missing_count}/{len(unique_locs)} loc_ids have no PNG in {png_dir}"
        )
    return png_paths


def _build_v_graft_row_paths(df: pd.DataFrame, imagery_source: str) -> list[Path | None]:
    """Return per-row PNG paths for ``*_v_graft`` imagery sources."""

    if not PAIRING_CSV.exists():
        raise FileNotFoundError(
            f"Pairing manifest not found at {PAIRING_CSV}. "
            "Run the v_graft download script first."
        )

    path_column = V_GRAFT_PATH_COLUMNS[imagery_source]
    exists_column = V_GRAFT_EXISTS_COLUMNS[imagery_source]
    pairing_df = pd.read_csv(PAIRING_CSV)
    required_columns = {"FilePath", path_column}
    missing_columns = required_columns.difference(pairing_df.columns)
    if missing_columns:
        raise ValueError(
            f"Pairing manifest {PAIRING_CSV} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    pairing_df["FilePath"] = pairing_df["FilePath"].astype(str)
    pairing_df = pairing_df.drop_duplicates(subset="FilePath", keep="last")
    pairing_lookup = pairing_df.set_index("FilePath")

    row_paths: list[Path | None] = []
    missing_count = 0
    for filepath in df["FilePath"].astype(str):
        if filepath not in pairing_lookup.index:
            row_paths.append(None)
            missing_count += 1
            continue

        record = pairing_lookup.loc[filepath]
        image_path = record.get(path_column)
        exists_flag = record.get(exists_column, True)
        if pd.isna(image_path) or str(image_path).strip() == "":
            row_paths.append(None)
            missing_count += 1
            continue

        path_obj = Path(str(image_path))
        if exists_flag is False or not path_obj.exists():
            row_paths.append(None)
            missing_count += 1
            continue
        row_paths.append(path_obj)

    if missing_count:
        print(
            f"Warning: {missing_count}/{len(df)} rows have no {imagery_source} PNG "
            f"via {PAIRING_CSV}"
        )
    return row_paths


def _get_imagery_family(imagery_source: str) -> str:
    if imagery_source.startswith("sentinel"):
        return "sentinel"
    if imagery_source.startswith("naip"):
        return "naip"
    raise ValueError(f"Unknown imagery_source '{imagery_source}'")


def _normalize_checkpoint_level(checkpoint_level: str) -> str:
    normalized = str(checkpoint_level).strip().lower()
    aliases = {
        "image": "image",
        "image_level": "image",
        "pixel": "pixel",
        "pixel_level": "pixel",
        "patch": "pixel",
        "patch_level": "pixel",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown checkpoint_level '{checkpoint_level}'. Choose from: ['image', 'pixel']"
        )
    return aliases[normalized]


def _get_backbone_name(imagery_source: str, checkpoint_level: str) -> str:
    base = IMAGERY_SOURCE_BACKBONE_BASE.get(imagery_source)
    if base is None:
        raise ValueError(
            f"Unknown imagery_source '{imagery_source}'. "
            f"Choose from: {sorted(IMAGERY_SOURCE_BACKBONE_BASE)}"
        )
    normalized_level = _normalize_checkpoint_level(checkpoint_level)
    if normalized_level == "image":
        return base
    return f"{base}_{normalized_level}"


def extract_graft_features(
    imagery_source: str = "sentinel",
    checkpoint_level: str = "image",
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
    checkpoint_level = _normalize_checkpoint_level(checkpoint_level)
    backbone_name = _get_backbone_name(imagery_source, checkpoint_level)

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
    imagery_family = _get_imagery_family(imagery_source)
    checkpoint_path = resolve_graft_checkpoint(imagery_family, checkpoint_level)
    print(
        f"Using GRAFT checkpoint: {checkpoint_path} "
        f"(family={imagery_family}, level={checkpoint_level})"
    )
    model = load_graft_model(
        GraftConfig(
            checkpoint_path=checkpoint_path,
            imagery_family=imagery_family,
            checkpoint_level=checkpoint_level,
            image_size=image_size,
        ),
        device=device,
    )
    # The v_graft download scripts already apply source-specific scaling so the
    # PNGs stay viewable; step 16 keeps CLIP normalization here for all sources.
    transform = build_graft_transform(image_size=image_size)

    if imagery_source in V_GRAFT_PATH_COLUMNS:
        row_paths = _build_v_graft_row_paths(df, imagery_source)
        rows_to_encode = [i for i, path_obj in enumerate(row_paths) if path_obj is not None]

        for start in tqdm(
            range(0, len(rows_to_encode), batch_size),
            desc=f"GRAFT ({imagery_source})",
        ):
            batch_rows = rows_to_encode[start : start + batch_size]
            images: list[torch.Tensor] = []
            valid_rows: list[int] = []

            for row_idx in batch_rows:
                image_path = row_paths[row_idx]
                if image_path is None:
                    continue
                try:
                    img = Image.open(image_path).convert("RGB")
                    images.append(transform(img))
                    valid_rows.append(row_idx)
                except Exception as exc:
                    print(f"Failed to load {image_path}: {exc}")

            if images:
                batch_tensor = torch.stack(images, dim=0).to(device)
                with torch.inference_mode():
                    embeds = model.forward_features(batch_tensor).detach().cpu().numpy()
                for row_idx, emb in zip(valid_rows, embeds):
                    features[row_idx] = emb.astype(np.float32, copy=False)
    else:
        png_paths = _build_loc_id_png_lookup(df, png_dir)
        unique_locs = df["loc_id"].unique()
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

        for i, lid in enumerate(loc_ids):
            if lid in loc_id_embeddings:
                features[i] = loc_id_embeddings[lid]

    valid_count = int(np.isfinite(features[:, 0]).sum())
    print(f"Extracted GRAFT features for {valid_count}/{len(df)} rows")

    np.save(feature_file, features)
    np.save(id_file, loc_ids)
    print(f"Wrote GRAFT {imagery_source} features to {feature_file}")


def main(
    imagery_source: str = "sentinel",
    checkpoint_level: str = "image",
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
        checkpoint_level=checkpoint_level,
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
