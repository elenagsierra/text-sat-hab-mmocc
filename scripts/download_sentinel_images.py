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
"""Download Sentinel-2 RGB images for every row in wi_blank_images.pkl.

This script is self-contained (no GRAFT / embedding logic) and can be run
independently of any other pipeline step.  It replicates the full
seasonal-fallback fetch logic used by sat_mmocc/rs_graft.py.

Usage examples
--------------
# dry-run / sanity-check (first 5 rows, no EE calls):
python scripts/download_sentinel_images.py --max_rows 0 --help

# download first 20 rows with 8 parallel workers:
python scripts/download_sentinel_images.py \
    --max_rows 20 --workers 8 --project my-gee-project

# full run:
python scripts/download_sentinel_images.py --project my-gee-project
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
# Sentinel-2 constants (mirrors sat_mmocc/rs_graft.py)
# ---------------------------------------------------------------------------
SENTINEL_DATASET = "COPERNICUS/S2_SR_HARMONIZED"
SENTINEL_RGB_BANDS = ("B4", "B3", "B2")
SENTINEL_PIXEL_SIZE_METERS = 10.0
SENTINEL_IMAGE_SIZE = 224
DEFAULT_CLOUD_PERCENT = 20.0
DEFAULT_TIME_WINDOW_DAYS = 60
DEFAULT_YEAR_RADIUS = 2

# joblib cache so interrupted runs resume without re-fetching
_MEMORY = Memory(cache_path / "sentinel_wi_rgb_cache", verbose=0)
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


def _apply_s2cloudless(
    collection: ee.ImageCollection,
    point: ee.Geometry,
    start_date: str,
    end_date: str,
) -> ee.ImageCollection:
    """Mask clouds using QA60 bitmask + s2cloudless probability band."""
    s2_cloudless = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(point)
        .filterDate(start_date, end_date)
    )
    inner_join = ee.Join.inner()
    join_filter = ee.Filter.equals(
        leftField="system:index", rightField="system:index"
    )
    joined = ee.ImageCollection(
        inner_join.apply(collection, s2_cloudless, join_filter)
    )

    def _mask(feature):
        img = ee.Image(feature.get("primary"))
        # QA60 bitmask (clouds + cirrus)
        qa = img.select("QA60")
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        qa_clear = (
            qa.bitwiseAnd(cloud_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )
        # s2cloudless probability
        cloud_prob = ee.Image(feature.get("secondary"))
        prob_clear = cloud_prob.select("probability").lt(50)
        return img.updateMask(qa_clear.And(prob_clear))

    return joined.map(_mask)


def _build_collection(
    point: ee.Geometry,
    start: date,
    end: date,
    dataset: str,
    cloud_percent: float,
) -> ee.ImageCollection:
    start_str = start.isoformat()
    end_str = end.isoformat()
    collection = (
        ee.ImageCollection(dataset)
        .filterBounds(point)
        .filterDate(start_str, end_str)
    )
    if cloud_percent is not None:
        collection = collection.filter(
            ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_percent)
        )
    return _apply_s2cloudless(collection, point, start_str, end_str)


def _build_seasonal_collection(
    point: ee.Geometry,
    target_date: date,
    window_days: int,
    year_radius: int,
    dataset: str,
    cloud_percent: float,
) -> ee.ImageCollection:
    """Collect images from the same seasonal window across multiple years."""
    start_year = max(2015, target_date.year - year_radius)
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
    collection = collection.filter(doy_filter)

    if cloud_percent is not None:
        collection = collection.filter(
            ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_percent)
        )
    return _apply_s2cloudless(collection, point, start_str, end_str)


def _download_thumb(
    image: ee.Image,
    region: ee.Geometry,
    size: int,
    pixel_size_meters: float = SENTINEL_PIXEL_SIZE_METERS,
) -> np.ndarray:
    """Download a thumbnail PNG from Earth Engine and return as uint8 RGB array.

    Uses a per-scene 2nd/98th percentile stretch with a shared scalar
    min/max across all three RGB bands.  A single shared stretch preserves
    colour balance (snow stays white, canopy stays green) so that images
    from different biomes and seasons remain reliably comparable by eye,
    while still adapting to each scene's actual dynamic range.
    """
    bands = list(SENTINEL_RGB_BANDS)
    # Default true-color stretch for Sentinel-2 surface reflectance (/10000, clamped 0-1).
    # 0-0.3 gives vivid colours for typical vegetation/land scenes.
    # For bright scenes (snow, sand, clouds) where p98 > 0.4, fall back to the
    # per-scene percentile stretch so highlights are not clipped to white.
    stretch_min, stretch_max = 0.0, 0.3
    try:
        stats = image.reduceRegion(
            reducer=ee.Reducer.percentile([2, 98]),
            geometry=region,
            scale=pixel_size_meters,
            maxPixels=1e6,
            bestEffort=True,
        ).getInfo()
        # dict.get only substitutes the default when the key is absent;
        # guard explicitly against None values returned for empty/masked regions.
        p2_vals  = [v for b in bands if (v := stats.get(f"{b}_p2"))  is not None]
        p98_vals = [v for b in bands if (v := stats.get(f"{b}_p98")) is not None]
        if p2_vals and p98_vals:
            p2  = min(p2_vals)
            p98 = max(p98_vals)
            p98 = max(p98, p2 + 1e-4)  # guard against flat/missing image
            if p98 > 0.4:
                # Bright scene (snow, sand, etc.) — use per-scene percentile stretch
                stretch_min, stretch_max = p2, p98
    except Exception:
        pass  # fall back to default fixed stretch

    try:
        url = image.getThumbURL(
            dict(
                region=region,
                dimensions=[size, size],
                format="png",
                min=stretch_min,
                max=stretch_max,
                gamma=1.0,
            )
        )
    except Exception as exc:
        # getThumbURL makes a server-side call in EE API ≥1.x and can raise
        # EEException with messages like "Collection.reduceColumns: Empty date
        # ranges not supported" when the composite is empty or fully masked.
        print(f"  getThumbURL failed ({exc}), skipping.")
        return None

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
    cloud_percent: float,
    project: str | None,
    dataset: str,
    year_radius: int,
) -> np.ndarray | None:
    """Cached fetch of a single Sentinel-2 patch (no save side-effects here)."""
    _initialize_ee(project=project)
    target_date = _parse_date(timestamp)
    point = ee.Geometry.Point([float(longitude), float(latitude)])

    start = target_date - timedelta(days=window_days)
    end = target_date + timedelta(days=window_days)

    # Attempt 1: direct time window
    collection = _build_collection(point, start, end, dataset, cloud_percent)

    # Attempt 2: same season, multiple years
    if collection.size().getInfo() == 0:
        collection = _build_seasonal_collection(
            point=point,
            target_date=target_date,
            window_days=window_days,
            year_radius=year_radius,
            dataset=dataset,
            cloud_percent=cloud_percent,
        )

    # Attempt 3: relax cloud filter entirely
    if collection.size().getInfo() == 0:
        collection = _build_seasonal_collection(
            point=point,
            target_date=target_date,
            window_days=window_days,
            year_radius=year_radius,
            dataset=dataset,
            cloud_percent=100.0,
        )

    if collection.size().getInfo() == 0:
        return None

    image = (
        collection.median()
        .select(list(SENTINEL_RGB_BANDS))
        .divide(10000)
        .clamp(0, 1)
        .unmask(0)
    )

    half_extent = (size * pixel_size_meters) / 2.0
    region = point.buffer(half_extent).bounds()
    return _download_thumb(image, region, size, pixel_size_meters)


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
    cloud_percent: float,
    project: str | None,
    dataset: str,
    year_radius: int,
) -> tuple[int, str]:
    """Fetch one patch and save it.  Returns (row_idx, status)."""
    # if dest.exists():
    #     return row_idx, "skipped"

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
            cloud_percent=cloud_percent,
            project=project,
            dataset=dataset,
            year_radius=year_radius,
        )
    except Exception as exc:
        print(f"\n[row {row_idx}] fetch failed: {exc}")
        return row_idx, "failed"

    if arr is None:
        print(f"\n[row {row_idx}] no usable imagery found, skipping.")
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
    cloud_percent: float = DEFAULT_CLOUD_PERCENT,
    image_size: int = SENTINEL_IMAGE_SIZE,
    pixel_size_meters: float = SENTINEL_PIXEL_SIZE_METERS,
    year_radius: int = DEFAULT_YEAR_RADIUS,
    dataset: str = SENTINEL_DATASET,
):
    """Download Sentinel-2 RGB patches keyed by loc_id for use with step 8.

    One PNG is saved per unique loc_id as:
        $CACHE_PATH/sat_wi_rgb_images_png/{loc_id}.png

    This matches the path expected by sat_mmocc/steps/08_visdiff.py.
    When a loc_id has multiple observation rows the first row's timestamp
    is used to anchor the temporal search window (same location, same
    seasonal fallback logic applies regardless of which row is chosen).
    """
    # Output dir must match what step 8 hard-codes
    out_root = Path(output_dir) if output_dir else cache_path / "sat_wi_rgb_images_png"
    out_root.mkdir(parents=True, exist_ok=True)

    pkl_path = cache_path / "wi_blank_images.pkl"
    print(f"Loading {pkl_path} ...")
    df = pd.read_pickle(pkl_path)

    # Mirror load_image_lookup() exactly so downloaded images correspond to the
    # same loc_ids and timestamps that 08_visdiff.py uses for ground-level images.
    #
    # Step 1: filter to validated blank images only
    valid_path = cache_path / "wi_blank_images_valid.txt"
    df_valid = pd.read_csv(valid_path, header=None, names=["FilePath"])
    df = pd.merge(df, df_valid, how="inner", on="FilePath")
    print(f"  {len(df)} rows after filtering to wi_blank_images_valid.txt")

    # Step 2: sort by Date_Time then deduplicate — keeps earliest timestamp per
    # loc_id, matching the drop_duplicates behaviour in load_image_lookup()
    df = (
        df.sort_values("Date_Time")
        .drop_duplicates(subset="loc_id", keep="first")
        .reset_index(drop=True)
    )

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
                cloud_percent,
                project,
                dataset,
                year_radius,
            ): idx
            for idx, row in df.iterrows()
        }

        with tqdm(total=len(futures), desc="Downloading", unit="img") as pbar:
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
        f"{counts['failed']} failed."
    )
    print(f"Images in: {out_root}")


if __name__ == "__main__":
    fire.Fire(main)
