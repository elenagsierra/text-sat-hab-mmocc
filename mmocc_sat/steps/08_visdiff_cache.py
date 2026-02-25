#!/usr/bin/env -S uv run --python 3.11 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "Pillow",
#     "pyvisdiff",
#     "setuptools",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# pyvisdiff = { git = "https://github.com/timmh/pyvisdiff.git" }
# ///
"""Run step 08 VisDiff using pre-existing rs_graft joblib cache entries.

This entrypoint reads cached fetch_sentinel_patch outputs from:
  CACHE_PATH/joblib_graft/mmocc/rs_graft/fetch_sentinel_patch/<hash>/
and materializes:
  CACHE_PATH/sat_images/<loc_id>.png
before delegating to 08_visdiff_data.py.
"""

from __future__ import annotations

import ast
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
import json
import logging
import pickle
from pathlib import Path
from typing import Sequence

import fire
import numpy as np
import pandas as pd
from PIL import Image

from mmocc_sat.config import cache_path

LOGGER = logging.getLogger(__name__)
DEFAULT_DATASET = "COPERNICUS/S2_SR_HARMONIZED"
DEFAULT_PROJECT = "zorrilla"


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


def _parse_arg(value: str):
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _normalize_timestamp(value) -> str:
    if isinstance(value, datetime):
        return pd.Timestamp(value, tz="UTC").isoformat()
    return pd.to_datetime(value, utc=True).isoformat()


def _cache_key(
    latitude: float,
    longitude: float,
    timestamp: str,
    *,
    window_days: int,
    size: int,
    pixel_size_meters: float,
    cloud_percent: float,
    project: str | None,
    dataset: str,
    latlon_round: int,
) -> tuple:
    return (
        round(float(latitude), latlon_round),
        round(float(longitude), latlon_round),
        _normalize_timestamp(timestamp),
        int(window_days),
        int(size),
        float(pixel_size_meters),
        float(cloud_percent),
        None if project in {None, ""} else str(project),
        str(dataset),
    )


def _build_joblib_index(
    joblib_fetch_dir: Path, *, latlon_round: int
) -> dict[tuple, Path]:
    index: dict[tuple, Path] = {}
    if not joblib_fetch_dir.exists():
        LOGGER.warning("Joblib cache directory not found: %s", joblib_fetch_dir)
        return index

    for hash_dir in sorted(joblib_fetch_dir.iterdir()):
        if not hash_dir.is_dir():
            continue
        meta_path = hash_dir / "metadata.json"
        out_path = hash_dir / "output.pkl"
        if not meta_path.exists() or not out_path.exists():
            continue
        try:
            metadata = json.loads(meta_path.read_text())
            args_raw = metadata.get("input_args", {})
            args = {k: _parse_arg(v) for k, v in args_raw.items()}
            key = _cache_key(
                args["latitude"],
                args["longitude"],
                args["timestamp"],
                window_days=args.get("window_days", 60),
                size=args.get("size", 224),
                pixel_size_meters=args.get("pixel_size_meters", 10.0),
                cloud_percent=args.get("cloud_percent", 20.0),
                project=args.get("project"),
                dataset=args.get("dataset", DEFAULT_DATASET),
                latlon_round=latlon_round,
            )
        except Exception as exc:
            LOGGER.warning("Skipping malformed cache entry at %s: %s", hash_dir, exc)
            continue
        index[key] = out_path
    return index


def _to_rgb_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def materialize_sat_images_from_joblib(
    *,
    max_sites: int | None = None,
    skip_existing: bool = True,
    window_days: int = 60,
    size: int = 224,
    pixel_size_meters: float = 10.0,
    cloud_percent: float = 20.0,
    project: str | None = DEFAULT_PROJECT,
    dataset: str = DEFAULT_DATASET,
    latlon_round: int = 6,
    joblib_fetch_dir: str | Path | None = None,
) -> dict:
    joblib_fetch_dir = Path(joblib_fetch_dir) if joblib_fetch_dir else (
        cache_path / "joblib_graft" / "mmocc" / "rs_graft" / "fetch_sentinel_patch"
    )
    sat_dir = cache_path / "sat_images"
    sat_dir.mkdir(parents=True, exist_ok=True)

    index = _build_joblib_index(joblib_fetch_dir, latlon_round=latlon_round)
    LOGGER.info("Loaded %d cache entries from %s", len(index), joblib_fetch_dir)

    df = pd.read_pickle(cache_path / "wi_blank_images.pkl").reset_index(drop=True)
    df = df.drop_duplicates(subset="loc_id").reset_index(drop=True)
    if max_sites is not None:
        df = df.head(max_sites).copy()

    counts = dict(total=int(len(df)), existing=0, written=0, cache_miss=0, failed=0)

    for _, row in df.iterrows():
        loc_id = str(row["loc_id"])
        out_path = sat_dir / f"{loc_id}.png"
        if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            counts["existing"] += 1
            continue

        key = _cache_key(
            row["Latitude"],
            row["Longitude"],
            row["Date_Time"],
            window_days=window_days,
            size=size,
            pixel_size_meters=pixel_size_meters,
            cloud_percent=cloud_percent,
            project=project,
            dataset=dataset,
            latlon_round=latlon_round,
        )
        out_pkl = index.get(key)
        if out_pkl is None:
            counts["cache_miss"] += 1
            continue

        try:
            with out_pkl.open("rb") as handle:
                patch = pickle.load(handle)
            if patch is None:
                counts["cache_miss"] += 1
                continue
            patch_rgb = _to_rgb_uint8(np.asarray(patch))
            Image.fromarray(patch_rgb, mode="RGB").save(out_path, format="PNG")
            counts["written"] += 1
        except Exception as exc:
            LOGGER.warning("Failed to materialize %s from %s: %s", loc_id, out_pkl, exc)
            counts["failed"] += 1

    LOGGER.info(
        "Materialization summary: total=%d existing=%d written=%d cache_miss=%d failed=%d",
        counts["total"],
        counts["existing"],
        counts["written"],
        counts["cache_miss"],
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
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    cache_dir: str | None = None,
    output_file: str | Path = cache_path / "visdiff_sat_descriptions.csv",
    preload_max_sites: int | None = None,
    preload_skip_existing: bool = True,
    preload_window_days: int = 60,
    preload_size: int = 224,
    preload_pixel_size_meters: float = 10.0,
    preload_cloud_percent: float = 20.0,
    preload_project: str | None = DEFAULT_PROJECT,
    preload_dataset: str = DEFAULT_DATASET,
    preload_latlon_round: int = 6,
    preload_joblib_fetch_dir: str | Path | None = None,
):
    materialize_sat_images_from_joblib(
        max_sites=preload_max_sites,
        skip_existing=preload_skip_existing,
        window_days=preload_window_days,
        size=preload_size,
        pixel_size_meters=preload_pixel_size_meters,
        cloud_percent=preload_cloud_percent,
        project=preload_project,
        dataset=preload_dataset,
        latlon_round=preload_latlon_round,
        joblib_fetch_dir=preload_joblib_fetch_dir,
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
