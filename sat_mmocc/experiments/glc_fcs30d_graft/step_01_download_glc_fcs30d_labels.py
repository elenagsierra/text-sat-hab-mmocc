#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "earthengine-api==1.4.0",
#     "fire",
#     "numpy",
#     "pandas",
#     "Pillow",
#     "tqdm",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Export GLC_FCS30D label patches aligned to cached sat_mmocc satellite PNGs.

This step assumes the experiment uses the existing location-keyed satellite
imagery already cached for sat_mmocc, rather than building a continuous raster
map. It writes:

  - a dense per-site label patch (`.npy`) aligned to the cached image footprint
  - a token-grid label patch (`.npy`) aligned to the GRAFT patch-token grid
  - a small color preview (`.png`) for qualitative inspection
  - a manifest CSV describing the exported artifacts
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

from sat_mmocc.experiments.glc_fcs30d_graft.utils import (
    DEFAULT_GLC_YEAR,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IMAGERY_SOURCE,
    DEFAULT_PIXEL_SIZE_METERS,
    DEFAULT_TOKEN_GRID_SIZE,
    class_id_to_name,
    dominant_non_fill_class,
    downsample_labels_to_token_grid,
    fetch_glc_label_patch,
    get_run_dir,
    list_image_site_rows,
    normalize_glc_year,
    save_color_preview,
    write_site_manifest,
)


def _artifact_paths(run_dir: Path, loc_id: str) -> dict[str, Path]:
    return {
        "label_patch_path": run_dir / "label_patches_224" / f"{loc_id}.npy",
        "token_label_path": run_dir / "token_labels_14" / f"{loc_id}.npy",
        "preview_path": run_dir / "label_previews" / f"{loc_id}.png",
    }


def export_labels(
    imagery_source: str = DEFAULT_IMAGERY_SOURCE,
    year: int = DEFAULT_GLC_YEAR,
    max_sites: int | None = None,
    workers: int = 8,
    skip_existing: bool = True,
    size: int = DEFAULT_IMAGE_SIZE,
    token_grid_size: int = DEFAULT_TOKEN_GRID_SIZE,
    pixel_size_meters: float = DEFAULT_PIXEL_SIZE_METERS,
    project: str | None = None,
) -> Path:
    year = normalize_glc_year(year)
    run_dir = get_run_dir(
        imagery_source=imagery_source,
        year=year,
        checkpoint_level="pixel",
    )
    site_df = list_image_site_rows(imagery_source=imagery_source, max_sites=max_sites)

    for subdir in ["label_patches_224", "token_labels_14", "label_previews"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    def _process_site(row: pd.Series) -> dict[str, object]:
        loc_id = str(row["loc_id"])
        paths = _artifact_paths(run_dir, loc_id)
        if skip_existing and all(path.exists() for path in paths.values()):
            token_labels = np.load(paths["token_label_path"], allow_pickle=False)
        else:
            label_patch = fetch_glc_label_patch(
                latitude=float(row["Latitude"]),
                longitude=float(row["Longitude"]),
                year=year,
                size=size,
                pixel_size_meters=pixel_size_meters,
                project=project,
            )
            token_labels = downsample_labels_to_token_grid(
                label_patch,
                token_grid_size=token_grid_size,
            )
            np.save(paths["label_patch_path"], label_patch.astype(np.int16, copy=False))
            np.save(paths["token_label_path"], token_labels.astype(np.int16, copy=False))
            save_color_preview(label_patch, paths["preview_path"])

        dominant_class = dominant_non_fill_class(token_labels)
        return {
            "loc_id": loc_id,
            "Latitude": float(row["Latitude"]),
            "Longitude": float(row["Longitude"]),
            "image_path": str(row["image_path"]),
            "glc_year": year,
            "label_patch_path": str(paths["label_patch_path"]),
            "token_label_path": str(paths["token_label_path"]),
            "preview_path": str(paths["preview_path"]),
            "dominant_class_id": dominant_class,
            "dominant_class_name": class_id_to_name(dominant_class),
            "num_valid_tokens": int(np.sum(token_labels != 0)),
        }

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [executor.submit(_process_site, row) for _, row in site_df.iterrows()]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Exporting GLC labels",
        ):
            rows.append(future.result())

    manifest_df = pd.DataFrame(rows).sort_values("loc_id").reset_index(drop=True)
    manifest_path = run_dir / "glc_label_manifest.csv"
    write_site_manifest(manifest_df, manifest_path)
    print(f"Wrote manifest to {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    fire.Fire(export_labels)
