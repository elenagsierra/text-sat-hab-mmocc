"""Remote-sensing utilities for fetching Sentinel-2 patches for GRAFT."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Optional

import ee
import numpy as np
import requests
from joblib import Memory
from PIL import Image

from mmocc_sat.config import cache_path

SENTINEL_DATASET = "COPERNICUS/S2_SR_HARMONIZED"
SENTINEL_RGB_BANDS = ("B4", "B3", "B2")
SENTINEL_PIXEL_SIZE_METERS = 10.0
SENTINEL_IMAGE_SIZE = 224
DEFAULT_CLOUD_PERCENT = 20.0
DEFAULT_TIME_WINDOW_DAYS = 60
DEFAULT_EE_PROJECT = "multimodal-sdm-473820"

_MEMORY = Memory(cache_path / "joblib_graft", verbose=0)
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


def initialize_ee(project: str | None = DEFAULT_EE_PROJECT) -> None:
    """Initialize Earth Engine, optionally with a project."""

    global _EE_PROJECT
    if _EE_PROJECT == project:
        return
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    _EE_PROJECT = project


def _mask_s2_sr(image: ee.Image) -> ee.Image:  # type: ignore[type-arg]
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = (
        qa.bitwiseAnd(cloud_bit_mask)
        .eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )
    return image.updateMask(mask)


def _build_collection(
    point: ee.Geometry,  # type: ignore[type-arg]
    start: date,
    end: date,
    dataset: str,
    cloud_percent: float,
) -> ee.ImageCollection:  # type: ignore[type-arg]
    collection = (
        ee.ImageCollection(dataset)
        .filterBounds(point)
        .filterDate(start.isoformat(), end.isoformat())
    )
    if cloud_percent is not None:
        collection = collection.filter(
            ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_percent)
        )
    return collection.map(_mask_s2_sr)


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
    project: str | None = DEFAULT_EE_PROJECT,
    dataset: str = SENTINEL_DATASET,
) -> np.ndarray | None:
    """Fetch a Sentinel-2 RGB patch centered at the given point."""

    initialize_ee(project=project)
    target_date = _parse_date(timestamp)
    start = target_date - timedelta(days=window_days)
    end = target_date + timedelta(days=window_days)
    point = ee.Geometry.Point([float(longitude), float(latitude)])  # type: ignore

    collection = _build_collection(point, start, end, dataset, cloud_percent)
    if collection.size().getInfo() == 0:
        # Fallback: expand temporal window and drop cloud filtering
        fallback_start = target_date - timedelta(days=365)
        fallback_end = target_date + timedelta(days=365)
        collection = _build_collection(point, fallback_start, fallback_end, dataset, 100)
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
    return _download_thumb(image, region, size)
