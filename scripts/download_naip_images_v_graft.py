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

This variant saves 224x224 RGB PNGs without CLIP normalization so the images
remain visually reasonable for step 08 / VisDiff. CLIP normalization is still
applied later in sat_mmocc/steps/16_extract_graft_features.py.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path

import ee
import fire
import numpy as np
import pandas as pd
import requests
from joblib import Memory
from PIL import Image
from tqdm import tqdm

from mmocc.config import cache_path

NAIP_DATASET = "USDA/NAIP/DOQQ"
NAIP_RGB_BANDS = ("R", "G", "B")
NAIP_PIXEL_SIZE_METERS = 1.0
NAIP_IMAGE_SIZE = 224
DEFAULT_YEAR_RADIUS = 5
DEFAULT_TIME_WINDOW_DAYS = 60

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


def _parse_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


def _build_collection(
    point: ee.Geometry,
    start: date,
    end: date,
    dataset: str,
) -> ee.ImageCollection:
    return (
        ee.ImageCollection(dataset)
        .filterBounds(point)
        .filterDate(start.isoformat(), end.isoformat())
    )


def _build_seasonal_collection(
    point: ee.Geometry,
    target_date: date,
    window_days: int,
    year_radius: int,
    dataset: str,
) -> ee.ImageCollection:
    start_year = max(2003, target_date.year - year_radius)
    end_year = target_date.year + year_radius
    collection = (
        ee.ImageCollection(dataset)
        .filterBounds(point)
        .filterDate(f"{start_year}-01-01", f"{end_year + 1}-01-01")
    )

    start_window = target_date - timedelta(days=window_days)
    end_window = target_date + timedelta(days=window_days)
    start_doy = start_window.timetuple().tm_yday
    end_doy = end_window.timetuple().tm_yday

    if start_doy <= end_doy:
        doy_filter = ee.Filter.dayOfYear(start_doy, end_doy)
    else:
        doy_filter = ee.Filter.Or(
            ee.Filter.dayOfYear(start_doy, 366),
            ee.Filter.dayOfYear(1, end_doy),
        )
    return collection.filter(doy_filter)


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
    window_days: int,
    size: int,
    pixel_size_meters: float,
    project: str | None,
    dataset: str,
    year_radius: int,
) -> np.ndarray | None:
    _initialize_ee(project=project)
    target_date = _parse_date(timestamp)
    point = ee.Geometry.Point([float(longitude), float(latitude)])

    start = target_date - timedelta(days=window_days)
    end = target_date + timedelta(days=window_days)
    collection = _build_collection(point, start, end, dataset)

    if collection.size().getInfo() == 0:
        collection = _build_seasonal_collection(
            point=point,
            target_date=target_date,
            window_days=window_days,
            year_radius=year_radius,
            dataset=dataset,
        )

    if collection.size().getInfo() == 0:
        start_year = max(2003, target_date.year - year_radius)
        end_year = target_date.year + year_radius
        collection = (
            ee.ImageCollection(dataset)
            .filterBounds(point)
            .filterDate(f"{start_year}-01-01", f"{end_year + 1}-01-01")
        )

    if collection.size().getInfo() == 0:
        return None

    image = collection.median().select(list(NAIP_RGB_BANDS)).unmask(0)
    half_extent = (size * pixel_size_meters) / 2.0
    region = point.buffer(half_extent).bounds()
    return _download_thumb(image, region, size)


def _process_row(
    row_idx: int,
    row: pd.Series,
    dest: Path,
    window_days: int,
    size: int,
    pixel_size_meters: float,
    project: str | None,
    dataset: str,
    year_radius: int,
) -> tuple[int, str]:
    if dest.exists():
        return row_idx, "skipped"

    timestamp = row["Date_Time"]
    if hasattr(timestamp, "to_pydatetime"):
        ts = timestamp.to_pydatetime().isoformat()
    else:
        ts = datetime.fromisoformat(str(timestamp)).isoformat()

    try:
        arr = _fetch_patch_cached(
            latitude=float(row["Latitude"]),
            longitude=float(row["Longitude"]),
            timestamp=ts,
            window_days=window_days,
            size=size,
            pixel_size_meters=pixel_size_meters,
            project=project,
            dataset=dataset,
            year_radius=year_radius,
        )
    except Exception as exc:
        print(f"\n[row {row_idx}] fetch failed: {exc}")
        return row_idx, "failed"

    if arr is None:
        print(f"\n[row {row_idx}] no usable NAIP imagery found, skipping.")
        return row_idx, "failed"

    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(dest)
    return row_idx, "saved"


def main(
    project: str | None = None,
    max_rows: int | None = None,
    workers: int = 8,
    output_dir: str | None = None,
    window_days: int = DEFAULT_TIME_WINDOW_DAYS,
    image_size: int = NAIP_IMAGE_SIZE,
    pixel_size_meters: float = NAIP_PIXEL_SIZE_METERS,
    year_radius: int = DEFAULT_YEAR_RADIUS,
    dataset: str = NAIP_DATASET,
):
    """Download NAIP v_graft PNGs keyed by loc_id."""
    out_root = Path(output_dir) if output_dir else cache_path / "naip_v_graft_images_png"
    out_root.mkdir(parents=True, exist_ok=True)

    pkl_path = cache_path / "wi_blank_images.pkl"
    print(f"Loading {pkl_path} ...")
    df = pd.read_pickle(pkl_path)

    valid_path = cache_path / "wi_blank_images_valid.txt"
    df_valid = pd.read_csv(valid_path, header=None, names=["FilePath"])
    df = pd.merge(df, df_valid, how="inner", on="FilePath")
    print(f"  {len(df)} rows after filtering to wi_blank_images_valid.txt")

    df = (
        df.sort_values("Date_Time")
        .drop_duplicates(subset="loc_id", keep="first")
        .reset_index(drop=True)
    )

    if max_rows is not None:
        df = df.head(max_rows).copy()
    print(f"Processing {len(df)} unique loc_ids  ->  saving to {out_root}")

    def _dest(row: pd.Series) -> Path:
        return out_root / f"{row['loc_id']}.png"

    counts = {"saved": 0, "skipped": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_row,
                idx,
                row,
                _dest(row),
                window_days,
                image_size,
                pixel_size_meters,
                project,
                dataset,
                year_radius,
            ): idx
            for idx, row in df.iterrows()
        }

        with tqdm(total=len(futures), desc="Downloading NAIP v_graft", unit="img") as pbar:
            for future in as_completed(futures):
                _, status = future.result()
                counts[status] += 1
                pbar.set_postfix(counts, refresh=False)
                pbar.update(1)

    total = sum(counts.values())
    print(
        f"\nDone.  {total} loc_ids processed: "
        f"{counts['saved']} saved, "
        f"{counts['skipped']} skipped, "
        f"{counts['failed']} failed."
    )
    print(f"Images in: {out_root}")


if __name__ == "__main__":
    fire.Fire(main)
