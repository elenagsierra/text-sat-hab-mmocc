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
"""Download Sentinel-2 patches prepared for GRAFT, while keeping PNGs viewable.

This variant:
  1. selects the temporally closest Sentinel-2 image for each location,
  2. requires <=1% scene cloud and <=1% cloud over the requested patch,
  3. saves B4/B3/B2 RGB PNGs after divide-by-3000 and clip-to-[0, 1],
  4. leaves CLIP normalization to sat_mmocc/steps/16_extract_graft_features.py.
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

SENTINEL_DATASET = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_PROBABILITY_DATASET = "COPERNICUS/S2_CLOUD_PROBABILITY"
SENTINEL_RGB_BANDS = ("B4", "B3", "B2")
SENTINEL_PIXEL_SIZE_METERS = 10.0
SENTINEL_IMAGE_SIZE = 224
DEFAULT_CLOUD_PERCENT = 1.0
DEFAULT_MAX_PATCH_CLOUD_FRACTION = 0.01
DEFAULT_ARCHIVE_START = "2015-01-01"
S2_CLOUD_PROBABILITY_THRESHOLD = 50

_MEMORY = Memory(cache_path / "sentinel_wi_rgb_v_graft_cache", verbose=0)
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


def _download_thumb(
    image: ee.Image,
    region: ee.Geometry,
    size: int,
) -> np.ndarray | None:
    try:
        url = image.getThumbURL(
            dict(
                region=region,
                dimensions=[size, size],
                format="png",
                min=0,
                max=1,
                gamma=1.0,
            )
        )
    except Exception as exc:
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


def _build_temporally_ranked_collection(
    point: ee.Geometry,
    region: ee.Geometry,
    target_date: date,
    dataset: str,
    cloud_percent: float,
    max_patch_cloud_fraction: float,
    pixel_size_meters: float,
    archive_start: str,
) -> ee.ImageCollection:
    archive_end = (date.today() + timedelta(days=1)).isoformat()
    collection = (
        ee.ImageCollection(dataset)
        .filterBounds(point)
        .filterDate(archive_start, archive_end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_percent))
    )
    s2_cloudless = (
        ee.ImageCollection(S2_CLOUD_PROBABILITY_DATASET)
        .filterBounds(point)
        .filterDate(archive_start, archive_end)
    )
    joined = ee.ImageCollection(
        ee.Join.inner().apply(
            collection,
            s2_cloudless,
            ee.Filter.equals(leftField="system:index", rightField="system:index"),
        )
    )
    target_ee_date = ee.Date(target_date.isoformat())

    def _annotate(feature):
        image = ee.Image(feature.get("primary"))
        cloud_prob = ee.Image(feature.get("secondary")).select("probability")
        qa = image.select("QA60")
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        qa_cloud = qa.bitwiseAnd(cloud_bit_mask).neq(0).Or(
            qa.bitwiseAnd(cirrus_bit_mask).neq(0)
        )
        prob_cloud = cloud_prob.gte(S2_CLOUD_PROBABILITY_THRESHOLD)
        cloud_mask = qa_cloud.Or(prob_cloud).rename("cloud_mask")
        cloud_stats = ee.Dictionary(
            cloud_mask.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=pixel_size_meters,
                maxPixels=1e6,
                bestEffort=True,
            )
        )
        patch_cloud_fraction = ee.Number(
            ee.Algorithms.If(
                cloud_stats.contains("cloud_mask"),
                cloud_stats.get("cloud_mask"),
                1.0,
            )
        )
        days_from_target = ee.Number(
            ee.Date(image.get("system:time_start"))
            .difference(target_ee_date, "day")
            .abs()
        )
        return image.set(
            {
                "days_from_target": days_from_target,
                "patch_cloud_fraction": patch_cloud_fraction,
            }
        )

    return (
        joined.map(_annotate)
        .filter(ee.Filter.lte("patch_cloud_fraction", max_patch_cloud_fraction))
        .sort("days_from_target")
    )


@_MEMORY.cache
def _fetch_patch_cached(
    latitude: float,
    longitude: float,
    timestamp: str,
    size: int,
    pixel_size_meters: float,
    cloud_percent: float,
    max_patch_cloud_fraction: float,
    project: str | None,
    dataset: str,
    archive_start: str,
) -> np.ndarray | None:
    _initialize_ee(project=project)
    target_date = _parse_date(timestamp)
    point = ee.Geometry.Point([float(longitude), float(latitude)])
    half_extent = (size * pixel_size_meters) / 2.0
    region = point.buffer(half_extent).bounds()

    collection = _build_temporally_ranked_collection(
        point=point,
        region=region,
        target_date=target_date,
        dataset=dataset,
        cloud_percent=cloud_percent,
        max_patch_cloud_fraction=max_patch_cloud_fraction,
        pixel_size_meters=pixel_size_meters,
        archive_start=archive_start,
    )
    if collection.size().getInfo() == 0:
        return None

    image = (
        ee.Image(collection.first())
        .select(list(SENTINEL_RGB_BANDS))
        .divide(3000)
        .clamp(0, 1)
        .unmask(0)
    )
    return _download_thumb(image, region, size)


def _process_row(
    row_idx: int,
    row: pd.Series,
    dest: Path,
    size: int,
    pixel_size_meters: float,
    cloud_percent: float,
    max_patch_cloud_fraction: float,
    project: str | None,
    dataset: str,
    archive_start: str,
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
            size=size,
            pixel_size_meters=pixel_size_meters,
            cloud_percent=cloud_percent,
            max_patch_cloud_fraction=max_patch_cloud_fraction,
            project=project,
            dataset=dataset,
            archive_start=archive_start,
        )
    except Exception as exc:
        print(f"\n[row {row_idx}] fetch failed: {exc}")
        return row_idx, "failed"

    if arr is None:
        print(f"\n[row {row_idx}] no qualifying Sentinel imagery found, skipping.")
        return row_idx, "failed"

    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(dest)
    return row_idx, "saved"


def main(
    project: str | None = None,
    max_rows: int | None = None,
    workers: int = 8,
    output_dir: str | None = None,
    cloud_percent: float = DEFAULT_CLOUD_PERCENT,
    max_patch_cloud_fraction: float = DEFAULT_MAX_PATCH_CLOUD_FRACTION,
    image_size: int = SENTINEL_IMAGE_SIZE,
    pixel_size_meters: float = SENTINEL_PIXEL_SIZE_METERS,
    dataset: str = SENTINEL_DATASET,
    archive_start: str = DEFAULT_ARCHIVE_START,
):
    """Download Sentinel v_graft PNGs keyed by loc_id."""
    out_root = (
        Path(output_dir)
        if output_dir
        else cache_path / "sentinel_v_graft_images_png"
    )
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
                image_size,
                pixel_size_meters,
                cloud_percent,
                max_patch_cloud_fraction,
                project,
                dataset,
                archive_start,
            ): idx
            for idx, row in df.iterrows()
        }

        with tqdm(total=len(futures), desc="Downloading Sentinel v_graft", unit="img") as pbar:
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
