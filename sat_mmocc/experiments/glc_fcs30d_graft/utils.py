"""Shared utilities for the GLC_FCS30D pixel-level GRAFT experiment.

This experiment is intentionally separate from the main occupancy pipeline.
It reuses the cached satellite PNGs already prepared for sat_mmocc and pairs
them with GLC_FCS30D land-cover labels for token-level zero-shot analysis.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from mmocc.config import cache_path

try:
    import ee
except ImportError:  # pragma: no cover - imported only in the EE download step
    ee = None

GLC_FCS30D_ANNUAL_DATASET = "projects/sat-io/open-datasets/GLC-FCS30D/annual"
GLC_FCS30D_FIVE_YEAR_DATASET = (
    "projects/sat-io/open-datasets/GLC-FCS30D/five-years-map"
)
DEFAULT_GLC_YEAR = 2022
DEFAULT_IMAGERY_SOURCE = "sentinel"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_PIXEL_SIZE_METERS = 10.0
DEFAULT_TOKEN_GRID_SIZE = 14
GLC_FILL_VALUE = 0

IMAGERY_SOURCE_PNG_DIRS: dict[str, Path] = {
    "sentinel": cache_path / "sat_wi_rgb_images_png",
    "naip": cache_path / "naip_wi_images_png",
}

GLC_CLASS_VALUES = [
    10,
    11,
    12,
    20,
    51,
    52,
    61,
    62,
    71,
    72,
    81,
    82,
    91,
    92,
    120,
    121,
    122,
    130,
    140,
    150,
    152,
    153,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    190,
    200,
    201,
    202,
    210,
    220,
    0,
]

GLC_CLASS_NAMES = [
    "Rainfed cropland",
    "Herbaceous cover cropland",
    "Tree or shrub cover cropland",
    "Irrigated cropland",
    "Open evergreen broadleaved forest",
    "Closed evergreen broadleaved forest",
    "Open deciduous broadleaved forest",
    "Closed deciduous broadleaved forest",
    "Open evergreen needle-leaved forest",
    "Closed evergreen needle-leaved forest",
    "Open deciduous needle-leaved forest",
    "Closed deciduous needle-leaved forest",
    "Open mixed leaf forest",
    "Closed mixed leaf forest",
    "Shrubland",
    "Evergreen shrubland",
    "Deciduous shrubland",
    "Grassland",
    "Lichens and mosses",
    "Sparse vegetation",
    "Sparse shrubland",
    "Sparse herbaceous",
    "Swamp",
    "Marsh",
    "Flooded flat",
    "Saline",
    "Mangrove",
    "Salt marsh",
    "Tidal flat",
    "Impervious surfaces",
    "Bare areas",
    "Consolidated bare areas",
    "Unconsolidated bare areas",
    "Water body",
    "Permanent ice and snow",
    "Filled value",
]

GLC_CLASS_COLORS = [
    "#ffff64",
    "#ffff64",
    "#ffff00",
    "#aaf0f0",
    "#4c7300",
    "#006400",
    "#a8c800",
    "#00a000",
    "#005000",
    "#003c00",
    "#286400",
    "#285000",
    "#a0b432",
    "#788200",
    "#966400",
    "#964b00",
    "#966400",
    "#ffb432",
    "#ffdcd2",
    "#ffebaf",
    "#ffd278",
    "#ffebaf",
    "#00a884",
    "#73ffdf",
    "#9ebb3b",
    "#828282",
    "#f57ab6",
    "#66cdab",
    "#444f89",
    "#c31400",
    "#fff5d7",
    "#dcdcdc",
    "#fff5d7",
    "#0046c8",
    "#ffffff",
    "#ffffff",
]

GLC_CLASS_TABLE = pd.DataFrame(
    {
        "class_id": GLC_CLASS_VALUES,
        "class_name": GLC_CLASS_NAMES,
        "color": GLC_CLASS_COLORS,
    }
)

AVAILABLE_GLC_YEARS = [1985, 1990, 1995] + list(range(2000, 2023))
CLASS_ID_TO_NAME = dict(zip(GLC_CLASS_VALUES, GLC_CLASS_NAMES, strict=True))
CLASS_ID_TO_COLOR = dict(zip(GLC_CLASS_VALUES, GLC_CLASS_COLORS, strict=True))

_EE_PROJECT: str | None = None
_GLC_IMAGE_CACHE: dict[int, Any] = {}


def experiment_cache_dir() -> Path:
    path = cache_path / "experiments" / "glc_fcs30d_graft"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_name(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def get_run_dir(
    imagery_source: str = DEFAULT_IMAGERY_SOURCE,
    year: int = DEFAULT_GLC_YEAR,
    checkpoint_level: str = "pixel",
) -> Path:
    path = (
        experiment_cache_dir()
        / sanitize_name(imagery_source)
        / f"glc_year_{normalize_glc_year(year)}"
        / f"graft_{sanitize_name(checkpoint_level)}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_image_dir(imagery_source: str = DEFAULT_IMAGERY_SOURCE) -> Path:
    image_dir = IMAGERY_SOURCE_PNG_DIRS.get(imagery_source)
    if image_dir is None:
        raise ValueError(
            f"Unknown imagery_source '{imagery_source}'. "
            f"Choose from: {sorted(IMAGERY_SOURCE_PNG_DIRS)}"
        )
    return image_dir


def normalize_glc_year(year: int) -> int:
    year = int(year)
    if year not in AVAILABLE_GLC_YEARS:
        raise ValueError(
            f"Unsupported GLC_FCS30D year '{year}'. "
            f"Choose from {AVAILABLE_GLC_YEARS[:3]} and 2000-2022."
        )
    return year


def get_glc_class_table(include_fill: bool = False) -> pd.DataFrame:
    if include_fill:
        return GLC_CLASS_TABLE.copy()
    return GLC_CLASS_TABLE[GLC_CLASS_TABLE["class_id"] != GLC_FILL_VALUE].reset_index(
        drop=True
    )


def class_id_to_name(class_id: int) -> str:
    return CLASS_ID_TO_NAME.get(int(class_id), f"Unknown ({class_id})")


def build_label_texts(include_fill: bool = False) -> list[str]:
    table = get_glc_class_table(include_fill=include_fill)
    return table["class_name"].tolist()


def colorize_label_grid(label_grid: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*label_grid.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_ID_TO_COLOR.items():
        mask = label_grid == class_id
        if not np.any(mask):
            continue
        hex_color = color.lstrip("#")
        rgb_value = np.array(
            [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8
        )
        rgb[mask] = rgb_value
    return rgb


def save_color_preview(label_grid: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colorize_label_grid(label_grid)).save(output_path)


def resize_nearest_2d(
    array: np.ndarray,
    out_height: int,
    out_width: int,
) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}")
    in_height, in_width = array.shape
    if in_height == out_height and in_width == out_width:
        return array.copy()
    row_idx = np.floor(np.arange(out_height) * (in_height / out_height)).astype(int)
    col_idx = np.floor(np.arange(out_width) * (in_width / out_width)).astype(int)
    row_idx = np.clip(row_idx, 0, in_height - 1)
    col_idx = np.clip(col_idx, 0, in_width - 1)
    return array[row_idx][:, col_idx]


def downsample_labels_to_token_grid(
    label_grid: np.ndarray,
    token_grid_size: int = DEFAULT_TOKEN_GRID_SIZE,
    fill_value: int = GLC_FILL_VALUE,
) -> np.ndarray:
    if label_grid.ndim != 2:
        raise ValueError(f"Expected a 2D label grid, got {label_grid.shape}")
    if label_grid.shape[0] != label_grid.shape[1]:
        raise ValueError("Label grid must be square for token alignment.")
    if label_grid.shape[0] % token_grid_size != 0:
        raise ValueError(
            f"Label grid size {label_grid.shape[0]} is not divisible by "
            f"token grid size {token_grid_size}."
        )

    block = label_grid.shape[0] // token_grid_size
    token_labels = np.full((token_grid_size, token_grid_size), fill_value, dtype=np.int16)

    for row in range(token_grid_size):
        for col in range(token_grid_size):
            patch = label_grid[
                row * block : (row + 1) * block,
                col * block : (col + 1) * block,
            ]
            values, counts = np.unique(patch, return_counts=True)
            if len(values) == 0:
                continue
            valid_mask = values != fill_value
            if np.any(valid_mask):
                values = values[valid_mask]
                counts = counts[valid_mask]
            token_labels[row, col] = int(values[np.argmax(counts)])
    return token_labels


def dominant_non_fill_class(
    label_grid: np.ndarray,
    fill_value: int = GLC_FILL_VALUE,
) -> int:
    values, counts = np.unique(label_grid, return_counts=True)
    valid_mask = values != fill_value
    if np.any(valid_mask):
        values = values[valid_mask]
        counts = counts[valid_mask]
    if len(values) == 0:
        return fill_value
    return int(values[np.argmax(counts)])


def reshape_patch_tokens(
    token_embeddings: np.ndarray,
    remove_cls_token: bool = True,
) -> np.ndarray:
    if token_embeddings.ndim != 2:
        raise ValueError(
            f"Expected token embeddings with shape [tokens, dim], got {token_embeddings.shape}"
        )
    tokens = token_embeddings
    if remove_cls_token:
        tokens = tokens[1:]
    grid_size = int(round(np.sqrt(tokens.shape[0])))
    if grid_size * grid_size != tokens.shape[0]:
        raise ValueError(
            f"Cannot reshape {tokens.shape[0]} tokens into a square grid."
        )
    return tokens.reshape(grid_size, grid_size, tokens.shape[-1])


def list_image_site_rows(
    imagery_source: str = DEFAULT_IMAGERY_SOURCE,
    max_sites: int | None = None,
) -> pd.DataFrame:
    image_dir = get_image_dir(imagery_source)
    df = pd.read_pickle(cache_path / "wi_blank_images.pkl")
    required_columns = {"loc_id", "Latitude", "Longitude"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"wi_blank_images.pkl is missing required columns: {sorted(missing)}"
        )

    site_df = (
        df[["loc_id", "Latitude", "Longitude"]]
        .dropna(subset=["loc_id", "Latitude", "Longitude"])
        .drop_duplicates(subset="loc_id", keep="first")
        .copy()
    )
    site_df["loc_id"] = site_df["loc_id"].astype(str)
    site_df["image_path"] = site_df["loc_id"].map(
        lambda loc_id: image_dir / f"{loc_id}.png"
    )
    site_df["image_exists"] = site_df["image_path"].map(Path.exists)
    site_df = site_df[site_df["image_exists"]].reset_index(drop=True)
    if max_sites is not None:
        site_df = site_df.head(max_sites).copy()
    return site_df


def write_site_manifest(site_df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = site_df.copy()
    for column in manifest.columns:
        if manifest[column].map(lambda value: isinstance(value, Path)).any():
            manifest[column] = manifest[column].map(str)
    manifest.to_csv(output_path, index=False)
    return output_path


def initialize_ee(project: str | None = None) -> None:
    global _EE_PROJECT
    if ee is None:
        raise ImportError(
            "earthengine-api is required for GLC_FCS30D label export. "
            "Install it in the calling script environment."
        )
    if _EE_PROJECT == project:
        return
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    _EE_PROJECT = project


@cache
def _five_year_band_names() -> list[str]:
    return [str(year) for year in [1985, 1990, 1995]]


@cache
def _annual_band_names() -> list[str]:
    return [str(year) for year in range(2000, 2023)]


def build_glc_image(year: int):
    year = normalize_glc_year(year)
    if year in _GLC_IMAGE_CACHE:
        return _GLC_IMAGE_CACHE[year]
    if ee is None:
        raise ImportError(
            "earthengine-api is required for GLC_FCS30D label export. "
            "Install it in the calling script environment."
        )

    annual_mosaic = ee.ImageCollection(GLC_FCS30D_ANNUAL_DATASET).mosaic()
    five_year_mosaic = ee.ImageCollection(GLC_FCS30D_FIVE_YEAR_DATASET).mosaic()

    if year in {1985, 1990, 1995}:
        image = five_year_mosaic.rename(_five_year_band_names()).select(str(year))
    else:
        image = annual_mosaic.rename(_annual_band_names()).select(str(year))

    image = image.rename("classification").toInt16()
    _GLC_IMAGE_CACHE[year] = image
    return image


def fetch_glc_label_patch(
    latitude: float,
    longitude: float,
    year: int = DEFAULT_GLC_YEAR,
    size: int = DEFAULT_IMAGE_SIZE,
    pixel_size_meters: float = DEFAULT_PIXEL_SIZE_METERS,
    project: str | None = None,
) -> np.ndarray:
    year = normalize_glc_year(year)
    initialize_ee(project=project)
    image = (
        build_glc_image(year)
        .unmask(GLC_FILL_VALUE)
        .resample("nearest")
        .reproject(crs="EPSG:3857", scale=pixel_size_meters)
    )
    point = ee.Geometry.Point([float(longitude), float(latitude)])
    half_extent = (size * pixel_size_meters) / 2.0
    region = point.buffer(half_extent).bounds()
    sampled = image.sampleRectangle(region=region, defaultValue=GLC_FILL_VALUE)
    values = sampled.get("classification").getInfo()
    if values is None:
        raise RuntimeError(
            f"Earth Engine returned no GLC_FCS30D values for point ({latitude}, {longitude})."
        )
    array = np.asarray(values, dtype=np.int16)
    if array.ndim != 2:
        raise RuntimeError(
            f"Expected a 2D GLC_FCS30D patch, received array with shape {array.shape}."
        )
    if array.shape != (size, size):
        array = resize_nearest_2d(array, size, size)
    return array
