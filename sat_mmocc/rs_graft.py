"""Remote-sensing utilities for fetching Sentinel-2 patches for GRAFT."""

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Optional

import ee
import numpy as np
import requests
from joblib import Memory
from PIL import Image

from mmocc.config import cache_path

SENTINEL_DATASET = "COPERNICUS/S2_SR_HARMONIZED"
SENTINEL_RGB_BANDS = ("B4", "B3", "B2")
SENTINEL_PIXEL_SIZE_METERS = 10.0
SENTINEL_IMAGE_SIZE = 224
DEFAULT_CLOUD_PERCENT = 20.0
DEFAULT_TIME_WINDOW_DAYS = 60

_MEMORY = Memory(cache_path / "sat_and_graft", verbose=0)
_EE_PROJECT: str | None = None


@dataclass(frozen=True)
class SentinelPatchRequest:
    """Parameters used to fetch a Sentinel-2 RGB patch."""

    latitude: float
    longitude: float
    timestamp: str
    window_days: int = DEFAULT_TIME_WINDOW_DAYS
    size: int = SENTINEL_IMAGE_SIZE
    pixel_size_meters: float = SENTINEL_PIXEL_SIZE_METERS
    cloud_percent: float = DEFAULT_CLOUD_PERCENT
    project: str | None = None
    dataset: str = SENTINEL_DATASET


def _parse_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


def initialize_ee(project: str | None = None) -> None:
    """Initialize Earth Engine, optionally with a project."""

    global _EE_PROJECT
    if _EE_PROJECT == project:
        return
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    _EE_PROJECT = project


def _apply_s2cloudless(
    collection: ee.ImageCollection,  # type: ignore[type-arg]
    point: ee.Geometry,  # type: ignore[type-arg]
    start_date: str,
    end_date: str,
) -> ee.ImageCollection:  # type: ignore[type-arg]
    """Helper to apply the s2cloudless probability mask to a given collection."""
    s2_cloudless = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(point)
        .filterDate(start_date, end_date)
    )
    inner_join = ee.Join.inner()
    join_filter = ee.Filter.equals(leftField="system:index", rightField="system:index")
    joined = ee.ImageCollection(inner_join.apply(collection, s2_cloudless, join_filter))

    def mask_clouds(feature):  # type: ignore
        img = ee.Image(feature.get("primary"))
        cloud_prob = ee.Image(feature.get("secondary"))
        is_clear = cloud_prob.select("probability").lt(50)
        return img.updateMask(is_clear)

    return joined.map(mask_clouds)


def _build_collection(
    point: ee.Geometry,  # type: ignore[type-arg]
    start: date,
    end: date,
    dataset: str,
    cloud_percent: float,
) -> ee.ImageCollection:  # type: ignore[type-arg]
    
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
    point: ee.Geometry,  # type: ignore[type-arg]
    target_date: date,
    window_days: int,
    year_radius: int,
    dataset: str,
    cloud_percent: float,
) -> ee.ImageCollection:  # type: ignore[type-arg]
    """Builds a collection across multiple years, strictly filtered to the target season."""
    
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
            ee.Filter.dayOfYear(1, end_doy)
        )
        
    collection = collection.filter(doy_filter)

    if cloud_percent is not None:
        collection = collection.filter(
            ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_percent)
        )
        
    return _apply_s2cloudless(collection, point, start_str, end_str)


def _download_thumb(
    image: ee.Image,  # type: ignore[type-arg]
    region: ee.Geometry,  # type: ignore[type-arg]
    size: int,
) -> np.ndarray | None:
    thumb = image.getThumbURL(
        dict(
            region=region,
            dimensions=[size, size],
            format="png",
            min=0,
            max=1,
        )
    )
    response = requests.get(thumb, timeout=90)
    response.raise_for_status()
    with Image.open(BytesIO(response.content)) as img:
        img = img.convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


@_MEMORY.cache
def fetch_sentinel_patch(
    latitude: float,
    longitude: float,
    timestamp: str,
    window_days: int = DEFAULT_TIME_WINDOW_DAYS,
    size: int = SENTINEL_IMAGE_SIZE,
    pixel_size_meters: float = SENTINEL_PIXEL_SIZE_METERS,
    cloud_percent: float = DEFAULT_CLOUD_PERCENT,
    project: str | None = None,
    dataset: str = SENTINEL_DATASET,
    year_radius: int = 2,
    output_dir: str | Path | None = None, # NEW PARAMETER FOR SAVING
) -> np.ndarray | None:
    """Fetch a Sentinel-2 RGB patch centered at the given point."""

    initialize_ee(project=project)
    target_date = _parse_date(timestamp)
    point = ee.Geometry.Point([float(longitude), float(latitude)])  # type: ignore

    start = target_date - timedelta(days=window_days)
    end = target_date + timedelta(days=window_days)
    collection = _build_collection(point, start, end, dataset, cloud_percent)
    
    if collection.size().getInfo() == 0:
        collection = _build_seasonal_collection(
            point=point,
            target_date=target_date,
            window_days=window_days,
            year_radius=year_radius, 
            dataset=dataset,
            cloud_percent=cloud_percent
        )
        
        if collection.size().getInfo() == 0:
             collection = _build_seasonal_collection(
                point=point,
                target_date=target_date,
                window_days=window_days,
                year_radius=year_radius, 
                dataset=dataset,
                cloud_percent=100
            )
             if collection.size().getInfo() == 0:
                 return None

    image = collection.median()
    image = (
        image.select(list(SENTINEL_RGB_BANDS))
        .divide(10000)
        .clamp(0, 1)
        .unmask(0)
    )

    half_extent = (size * pixel_size_meters) / 2.0
    region = point.buffer(half_extent).bounds()  # type: ignore
    img_array = _download_thumb(image, region, size)

    # SAVING LOGIC HERE
    if img_array is not None and output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        safe_ts = timestamp.replace(":", "-").replace("+", "_plus_")
        filename = f"sentinel_{latitude:.4f}_{longitude:.4f}_{safe_ts}.png"
        
        Image.fromarray(img_array).save(out_path / filename)

    return img_array