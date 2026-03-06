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
"""Download NAIP RGB images for every row in wi_blank_images.pkl.

NAIP (National Agriculture Imagery Program) provides high-resolution (~1 m/px)
aerial orthophotography for the continental United States (CONUS).  Rows whose
coordinates fall outside CONUS will fail to find any imagery and are skipped.

Key differences from download_sentinel_images.py
-------------------------------------------------
* Dataset : ``USDA/NAIP/DOQQ`` instead of ``COPERNICUS/S2_SR_HARMONIZED``
* Bands   : R, G, B (uint8, 0-255) — no cloud-masking step required
* Pixel   : ~1 m/pixel (NAIP_PIXEL_SIZE_METERS = 1.0)
* Coverage: CONUS only; ~every 2-3 years per location
* Year radius default set to 5 to account for sparser revisit frequency

Usage examples
--------------
# dry-run / sanity-check:
python scripts/download_naip_images.py --help

# download first 20 rows with 8 parallel workers:
python scripts/download_naip_images.py \
    --max_rows 20 --workers 8 --project my-gee-project

# full run:
python scripts/download_naip_images.py --project my-gee-project
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

# ---------------------------------------------------------------------------
# NAIP constants
# ---------------------------------------------------------------------------
NAIP_DATASET = "USDA/NAIP/DOQQ"
NAIP_RGB_BANDS = ("R", "G", "B")
# NAIP nominal ground sample distance is 0.6–1 m; use 1 m as a safe default
NAIP_PIXEL_SIZE_METERS = 1.0
NAIP_IMAGE_SIZE = 224
# Sparser revisit frequency than Sentinel — look ±5 years by default
DEFAULT_YEAR_RADIUS = 5
DEFAULT_TIME_WINDOW_DAYS = 60

# joblib cache so interrupted runs resume without re-fetching
_MEMORY = Memory(cache_path / "naip_rgb_cache", verbose=0)
_EE_PROJECT: str | None = None


# ---------------------------------------------------------------------------
# Earth Engine helpers
# ---------------------------------------------------------------------------

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
    """Return NAIP images within a date window.

    NAIP is aerial photography acquired under clear conditions, so no
    cloud-masking step is needed.
    """
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
    """Collect NAIP images from the same seasonal window across multiple years."""
    start_year = max(2003, target_date.year - year_radius)  # NAIP starts ~2003
    end_year = target_date.year + year_radius
    start_str = f"{start_year}-01-01"
    end_str = f"{end_year + 1}-01-01"

    collection = (
        ee.ImageCollection(dataset)
        .filterBounds(point)
        .filterDate(start_str, end_str)
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
) -> np.ndarray:
    """Download a thumbnail PNG and return as uint8 RGB array.

    NAIP R/G/B bands are uint8 (0-255), so min=0, max=255 maps the full
    dynamic range naturally without stretching or gamma correction.
    """
    url = image.getThumbURL(
        dict(
            region=region,
            dimensions=[size, size],
            format="png",
            min=0,
            max=255,
        )
    )
    response = requests.get(url, timeout=90)
    response.raise_for_status()
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
    """Cached fetch of a single NAIP patch (no save side-effects here)."""
    _initialize_ee(project=project)
    target_date = _parse_date(timestamp)
    point = ee.Geometry.Point([float(longitude), float(latitude)])

    start = target_date - timedelta(days=window_days)
    end = target_date + timedelta(days=window_days)

    # Attempt 1: direct time window
    collection = _build_collection(point, start, end, dataset)

    # Attempt 2: same season, multiple years (NAIP revisit is every 2-3 yr)
    if collection.size().getInfo() == 0:
        collection = _build_seasonal_collection(
            point=point,
            target_date=target_date,
            window_days=window_days,
            year_radius=year_radius,
            dataset=dataset,
        )

    # Attempt 3: whole multi-year range (drop seasonal constraint entirely)
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

    image = (
        collection.median()
        .select(list(NAIP_RGB_BANDS))
        .unmask(0)
    )

    half_extent = (size * pixel_size_meters) / 2.0
    region = point.buffer(half_extent).bounds()
    return _download_thumb(image, region, size)


# ---------------------------------------------------------------------------
# Worker function (run in thread pool)
# ---------------------------------------------------------------------------

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
    """Fetch one patch and save it.  Returns (row_idx, status)."""
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
        print(f"\n[row {row_idx}] no usable NAIP imagery found (may be outside CONUS), skipping.")
        return row_idx, "failed"

    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(dest)
    return row_idx, "saved"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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
    """Download NAIP RGB patches keyed by loc_id for use with step 8 (VisDiff).

    One PNG is saved per unique loc_id as:
        $CACHE_PATH/naip_images_png/{loc_id}.png

    This matches the path expected by sat_mmocc/steps/08_visdiff.py when
    ``--imagery_source naip`` is specified.

    Note: NAIP covers CONUS only.  Rows outside the continental United States
    will not find imagery and are silently skipped (status: failed).
    """
    out_root = Path(output_dir) if output_dir else cache_path / "naip_images_png"
    out_root.mkdir(parents=True, exist_ok=True)

    pkl_path = cache_path / "wi_blank_images.pkl"
    print(f"Loading {pkl_path} ...")
    df = pd.read_pickle(pkl_path).reset_index(drop=True)

    # Deduplicate: one representative row per loc_id (first occurrence)
    df = df.drop_duplicates(subset="loc_id", keep="first").reset_index(drop=True)

    if max_rows is not None:
        df = df.head(max_rows).copy()
    print(f"Processing {len(df)} unique loc_ids  →  saving to {out_root}")

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

        with tqdm(total=len(futures), desc="Downloading NAIP", unit="img") as pbar:
            for future in as_completed(futures):
                _, status = future.result()
                counts[status] += 1
                pbar.set_postfix(counts, refresh=False)
                pbar.update(1)

    total = sum(counts.values())
    print(
        f"\nDone.  {total} loc_ids processed: "
        f"{counts['saved']} saved, "
        f"{counts['skipped']} skipped (already exist), "
        f"{counts['failed']} failed (no NAIP coverage or outside CONUS)."
    )
    print(f"Images in: {out_root}")


if __name__ == "__main__":
    fire.Fire(main)
