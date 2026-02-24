#!/usr/bin/env -S uv run --python 3.11 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "pyvisdiff",
#     "earthengine-api==1.4.0",
#     "Pillow",
#     "setuptools",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# pyvisdiff = { git = "https://github.com/timmh/pyvisdiff.git" }
# ///
"""Run step 08 VisDiff with on-demand satellite image pulling via rs_graft.

This entrypoint first materializes:
  CACHE_PATH/sat_images/<loc_id>.png
using mmocc_sat.rs_graft.fetch_sentinel_patch, then runs the same VisDiff flow as
08_visdiff_data.py.
"""

from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
import logging
import os
from pathlib import Path
from typing import Sequence

import fire
import pandas as pd
from PIL import Image

from mmocc_sat.config import cache_path
from mmocc_sat.rs_graft import (
    DEFAULT_CLOUD_PERCENT,
    DEFAULT_TIME_WINDOW_DAYS,
    SENTINEL_IMAGE_SIZE,
    SENTINEL_PIXEL_SIZE_METERS,
    fetch_sentinel_patch,
)

LOGGER = logging.getLogger(__name__)


def _load_data_main():
    script_path = Path(__file__).with_name("08_visdiff_data.py")
    spec = spec_from_file_location("mmocc_sat_steps_08_visdiff_data", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load data VisDiff script at {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise RuntimeError(f"Data VisDiff script missing main(): {script_path}")
    return module.main


def _to_iso_timestamp(value) -> str:
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().isoformat()
    return datetime.fromisoformat(str(value)).isoformat()


def pull_sat_images(
    max_sites: int | None = None,
    skip_existing: bool = True,
    project: str | None = os.getenv("EE_PROJECT"),
    window_days: int = DEFAULT_TIME_WINDOW_DAYS,
    cloud_percent: float = DEFAULT_CLOUD_PERCENT,
    image_size: int = SENTINEL_IMAGE_SIZE,
    pixel_size_meters: float = SENTINEL_PIXEL_SIZE_METERS,
) -> dict:
    sat_dir = cache_path / "sat_images"
    sat_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(cache_path / "wi_blank_images.pkl").reset_index(drop=True)
    df = df.drop_duplicates(subset="loc_id").reset_index(drop=True)
    if max_sites is not None:
        df = df.head(max_sites).copy()

    counts = dict(total=int(len(df)), existing=0, written=0, missing=0, failed=0)

    for _, row in df.iterrows():
        loc_id = str(row["loc_id"])
        out_path = sat_dir / f"{loc_id}.png"
        if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            counts["existing"] += 1
            continue

        try:
            patch = fetch_sentinel_patch(
                latitude=float(row["Latitude"]),
                longitude=float(row["Longitude"]),
                timestamp=_to_iso_timestamp(row["Date_Time"]),
                window_days=window_days,
                size=image_size,
                pixel_size_meters=pixel_size_meters,
                cloud_percent=cloud_percent,
                project=project,
            )
        except Exception as exc:
            LOGGER.warning("Failed to fetch patch for %s: %s", loc_id, exc)
            counts["failed"] += 1
            continue

        if patch is None:
            counts["missing"] += 1
            continue

        try:
            Image.fromarray(patch.astype("uint8"), mode="RGB").save(
                out_path, format="PNG"
            )
            counts["written"] += 1
        except Exception as exc:
            LOGGER.warning("Failed to write PNG for %s to %s: %s", loc_id, out_path, exc)
            counts["failed"] += 1

    LOGGER.info(
        "Satellite pull summary: total=%d existing=%d written=%d missing=%d failed=%d",
        counts["total"],
        counts["existing"],
        counts["written"],
        counts["missing"],
        counts["failed"],
    )
    return counts


def main(
    modalities: Sequence[str] | str | None = None,
    image_backbones: Sequence[str] | str | None = None,
    sat_backbones: Sequence[str] | str | None = None,
    species_ids: Sequence[str] | str | None = None,
    top_k: int = 50,
    modes: Sequence[str] | str | None = None,
    unique_weight: float = 2.0,
    wandb_entity: str | None = os.getenv("VISDIFF_WANDB_ENTITY"),
    wandb_project: str | None = os.getenv("VISDIFF_WANDB_PROJECT"),
    cache_dir: str | None = None,
    output_file: str | Path = cache_path / "visdiff_sat_descriptions.csv",
    pull_max_sites: int | None = None,
    pull_skip_existing: bool = True,
    pull_project: str | None = os.getenv("EE_PROJECT"),
    pull_window_days: int = DEFAULT_TIME_WINDOW_DAYS,
    pull_cloud_percent: float = DEFAULT_CLOUD_PERCENT,
    pull_image_size: int = SENTINEL_IMAGE_SIZE,
    pull_pixel_size_meters: float = SENTINEL_PIXEL_SIZE_METERS,
):
    pull_sat_images(
        max_sites=pull_max_sites,
        skip_existing=pull_skip_existing,
        project=pull_project,
        window_days=pull_window_days,
        cloud_percent=pull_cloud_percent,
        image_size=pull_image_size,
        pixel_size_meters=pull_pixel_size_meters,
    )

    data_main = _load_data_main()
    return data_main(
        modalities=modalities,
        image_backbones=image_backbones,
        sat_backbones=sat_backbones,
        species_ids=species_ids,
        top_k=top_k,
        modes=modes,
        unique_weight=unique_weight,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        cache_dir=cache_dir,
        output_file=output_file,
    )


if __name__ == "__main__":
    fire.Fire(main)
