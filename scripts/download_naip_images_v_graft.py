#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "earthengine-api==1.4.0",
#     "fire",
#     "joblib",
#     "numpy",
#     "pandas",
#     "Pillow",
#     "requests==2.32.5",
#     "tqdm",
# ]
#
# [tool.uv.sources]
# mmocc = { path = ".." }
# ///
"""Download NAIP RGB patches prepared for GRAFT, while keeping PNGs viewable.

This variant now pairs one NAIP image to each camera-trap row and records the
association in a shared manifest for step 08c.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from hashlib import sha1
from io import BytesIO
from pathlib import Path
from typing import Any

import ee
import fire
import numpy as np
import pandas as pd
import requests
from joblib import Memory
from PIL import Image
from tqdm import tqdm

from mmocc.config import cache_path, wi_image_path

NAIP_DATASET = "USDA/NAIP/DOQQ"
NAIP_RGB_BANDS = ("R", "G", "B")
NAIP_PIXEL_SIZE_METERS = 1.0
NAIP_IMAGE_SIZE = 224
DEFAULT_ARCHIVE_START = "2003-01-01"
PAIRING_CSV = cache_path / "camera_satellite_pairings_v_graft.csv"

MANIFEST_COLUMNS = [
    "pair_id",
    "FilePath",
    "loc_id",
    "ground_date_time",
    "Latitude",
    "Longitude",
    "ground_image_path",
    "ground_image_exists",
    "sentinel_image_path",
    "sentinel_exists",
    "sentinel_source_datetime",
    "sentinel_days_offset",
    "sentinel_scene_cloud_percent",
    "sentinel_patch_cloud_fraction",
    "naip_image_path",
    "naip_exists",
    "naip_source_datetime",
    "naip_days_offset",
]

_MEMORY = Memory(cache_path / "naip_wi_rgb_v_graft_cache", verbose=0)
_EE_PROJECT: str | None = None


def _initialize_ee(project: str | None = None) -> None:
    global _EE_PROJECT
    if _EE_PROJECT == project:
        return
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    _EE_PROJECT = project


def _make_pair_id(filepath: str) -> str:
    return sha1(filepath.encode("utf-8")).hexdigest()[:16]


def _load_ground_rows(max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_pickle(cache_path / "wi_blank_images.pkl").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No cached Wildlife Insights blank images found.")
    if max_rows is not None:
        df = df.head(max_rows).copy()
    df["FilePath"] = df["FilePath"].astype(str)
    df["pair_id"] = df["FilePath"].apply(_make_pair_id)
    df["ground_date_time"] = pd.to_datetime(df["Date_Time"], errors="coerce")
    df["ground_image_path"] = df["FilePath"].str.replace(
        "gs://", f"{wi_image_path}/", n=1
    )
    df["ground_image_exists"] = df["ground_image_path"].apply(lambda p: Path(p).exists())
    return df


def _parse_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


def _millis_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _build_temporally_ranked_collection(
    point: ee.Geometry,
    target_date: date,
    dataset: str,
    archive_start: str,
) -> ee.ImageCollection:
    archive_end = (date.today() + timedelta(days=1)).isoformat()
    target_ee_date = ee.Date(target_date.isoformat())

    def _annotate(image):
        days_from_target = ee.Number(
            ee.Date(image.get("system:time_start")).difference(target_ee_date, "day").abs()
        )
        return image.set({"days_from_target": days_from_target})

    return (
        ee.ImageCollection(dataset)
        .filterBounds(point)
        .filterDate(archive_start, archive_end)
        .map(_annotate)
        .sort("days_from_target")
    )


def _download_thumb(
    image: ee.Image,
    region: ee.Geometry,
    size: int,
) -> np.ndarray | None:
    url = image.getThumbURL(
        dict(
            region=region,
            dimensions=[size, size],
            format="png",
            min=0,
            max=255,
        )
    )
    try:
        response = requests.get(url, timeout=90)
        response.raise_for_status()
    except requests.HTTPError as exc:
        print(f"  thumbnail HTTP error ({exc.response.status_code}), skipping.")
        return None
    with Image.open(BytesIO(response.content)) as img:
        img = img.convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


@_MEMORY.cache
def _fetch_patch_cached(
    latitude: float,
    longitude: float,
    timestamp: str,
    size: int,
    pixel_size_meters: float,
    project: str | None,
    dataset: str,
    archive_start: str,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    _initialize_ee(project=project)
    target_date = _parse_date(timestamp)
    point = ee.Geometry.Point([float(longitude), float(latitude)])

    collection = _build_temporally_ranked_collection(
        point=point,
        target_date=target_date,
        dataset=dataset,
        archive_start=archive_start,
    )
    if collection.size().getInfo() == 0:
        return None, None

    first = ee.Image(collection.first())
    props = first.toDictionary(["system:time_start", "days_from_target"]).getInfo()
    image = first.select(list(NAIP_RGB_BANDS)).unmask(0)
    half_extent = (size * pixel_size_meters) / 2.0
    region = point.buffer(half_extent).bounds()
    arr = _download_thumb(image, region, size)
    if arr is None:
        return None, None
    payload = {
        "naip_source_datetime": _millis_to_iso(props.get("system:time_start")),
        "naip_days_offset": float(props.get("days_from_target")) if props.get("days_from_target") is not None else None,
    }
    return arr, payload


def _process_row(
    row_idx: int,
    row: pd.Series,
    dest: Path,
    size: int,
    pixel_size_meters: float,
    project: str | None,
    dataset: str,
    archive_start: str,
) -> tuple[int, str, dict[str, Any]]:
    if dest.exists():
        return row_idx, "skipped", {"pair_id": row["pair_id"]}

    timestamp = row["Date_Time"]
    if hasattr(timestamp, "to_pydatetime"):
        ts = timestamp.to_pydatetime().isoformat()
    else:
        ts = datetime.fromisoformat(str(timestamp)).isoformat()

    try:
        arr, payload = _fetch_patch_cached(
            latitude=float(row["Latitude"]),
            longitude=float(row["Longitude"]),
            timestamp=ts,
            size=size,
            pixel_size_meters=pixel_size_meters,
            project=project,
            dataset=dataset,
            archive_start=archive_start,
        )
    except Exception as exc:
        print(f"\n[row {row_idx}] fetch failed: {exc}")
        return row_idx, "failed", {"pair_id": row["pair_id"]}

    if arr is None or payload is None:
        print(f"\n[row {row_idx}] no usable NAIP imagery found, skipping.")
        return row_idx, "failed", {"pair_id": row["pair_id"]}

    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(dest)
    payload["pair_id"] = row["pair_id"]
    return row_idx, "saved", payload


def _merge_manifest(existing: pd.DataFrame | None, updated: pd.DataFrame) -> pd.DataFrame:
    updated = updated.set_index("pair_id")
    if existing is not None and not existing.empty:
        existing = existing.copy()
        existing["pair_id"] = existing["pair_id"].astype(str)
        merged = updated.combine_first(existing.set_index("pair_id"))
    else:
        merged = updated
    merged = merged.reset_index()
    for col in MANIFEST_COLUMNS:
        if col not in merged.columns:
            merged[col] = None
    return merged[MANIFEST_COLUMNS].sort_values(["ground_date_time", "pair_id"], na_position="last")


def _write_manifest(
    df: pd.DataFrame,
    out_root: Path,
    updates: list[dict[str, Any]],
    manifest_path: Path,
) -> None:
    base = pd.DataFrame(
        {
            "pair_id": df["pair_id"],
            "FilePath": df["FilePath"],
            "loc_id": df["loc_id"],
            "ground_date_time": df["ground_date_time"],
            "Latitude": df["Latitude"],
            "Longitude": df["Longitude"],
            "ground_image_path": df["ground_image_path"],
            "ground_image_exists": df["ground_image_exists"],
            "sentinel_image_path": None,
            "sentinel_exists": None,
            "sentinel_source_datetime": None,
            "sentinel_days_offset": None,
            "sentinel_scene_cloud_percent": None,
            "sentinel_patch_cloud_fraction": None,
            "naip_image_path": df["pair_id"].apply(lambda pid: str(out_root / f"{pid}.png")),
            "naip_exists": df["pair_id"].apply(lambda pid: (out_root / f"{pid}.png").exists()),
            "naip_source_datetime": None,
            "naip_days_offset": None,
        }
    )

    if updates:
        updates_df = pd.DataFrame(updates).drop_duplicates(subset="pair_id", keep="last")
        base = base.merge(updates_df, how="left", on="pair_id", suffixes=("", "_new"))
        for col in ["naip_source_datetime", "naip_days_offset"]:
            new_col = f"{col}_new"
            if new_col in base.columns:
                base[col] = base[new_col].combine_first(base[col])
                base = base.drop(columns=[new_col])

    existing = pd.read_csv(manifest_path) if manifest_path.exists() else None
    merged = _merge_manifest(existing, base)
    merged.to_csv(manifest_path, index=False)
    print(f"Wrote pairing manifest to {manifest_path}")


def main(
    project: str | None = None,
    max_rows: int | None = None,
    workers: int = 8,
    output_dir: str | None = None,
    image_size: int = NAIP_IMAGE_SIZE,
    pixel_size_meters: float = NAIP_PIXEL_SIZE_METERS,
    dataset: str = NAIP_DATASET,
    archive_start: str = DEFAULT_ARCHIVE_START,
    manifest_path: str | None = None,
):
    """Download one NAIP PNG per camera-trap row and update the pairing manifest."""
    out_root = Path(output_dir) if output_dir else cache_path / "naip_v_graft_images_png"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest_path) if manifest_path else PAIRING_CSV

    df = _load_ground_rows(max_rows=max_rows)
    print(f"Processing {len(df)} camera-trap rows  ->  saving to {out_root}")

    counts = {"saved": 0, "skipped": 0, "failed": 0}
    updates: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_row,
                idx,
                row,
                out_root / f"{row['pair_id']}.png",
                image_size,
                pixel_size_meters,
                project,
                dataset,
                archive_start,
            ): idx
            for idx, row in df.iterrows()
        }

        with tqdm(total=len(futures), desc="Downloading NAIP v_graft", unit="img") as pbar:
            for future in as_completed(futures):
                _, status, payload = future.result()
                counts[status] += 1
                if payload:
                    updates.append(payload)
                pbar.set_postfix(counts, refresh=False)
                pbar.update(1)

    _write_manifest(df, out_root, updates, manifest)

    total = sum(counts.values())
    print(
        f"\nDone.  {total} rows processed: "
        f"{counts['saved']} saved, "
        f"{counts['skipped']} skipped, "
        f"{counts['failed']} failed."
    )
    print(f"Images in: {out_root}")


if __name__ == "__main__":
    fire.Fire(main)
