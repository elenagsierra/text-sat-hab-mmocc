from __future__ import annotations

from pathlib import Path

import pandas as pd

from sat_mmocc.config import cache_path

DEFAULT_PAIRING_CSV = cache_path / "camera_satellite_pairings_v_graft.csv"

IMAGERY_SOURCE_PNG_DIRS: dict[str, Path] = {
    "sentinel": cache_path / "sat_wi_rgb_images_png",
    "naip": cache_path / "naip_wi_images_png",
    "sentinel_v_graft": cache_path / "sentinel_v_graft_images_png",
    "naip_v_graft": cache_path / "naip_v_graft_images_png",
}

IMAGERY_SOURCE_LABELS: dict[str, str] = {
    "sentinel": "Sentinel",
    "naip": "NAIP",
    "sentinel_v_graft": "Sentinel v_graft",
    "naip_v_graft": "NAIP v_graft",
}

PAIRING_PATH_COLUMNS: dict[str, str] = {
    "ground": "ground_image_path",
    "sentinel_v_graft": "sentinel_image_path",
    "naip_v_graft": "naip_image_path",
}

PAIRING_EXISTS_COLUMNS: dict[str, str] = {
    "ground": "ground_image_exists",
    "sentinel_v_graft": "sentinel_exists",
    "naip_v_graft": "naip_exists",
}


def _coerce_exists(series: pd.Series, paths: pd.Series) -> pd.Series:
    normalized = series.fillna(False)
    if normalized.dtype == bool:
        return normalized

    text = normalized.astype(str).str.strip().str.lower()
    mapped = text.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "": False,
            "nan": False,
            "none": False,
        }
    )
    fallback = paths.fillna("").astype(str).str.strip().ne("")
    return mapped.fillna(fallback).astype(bool)


def _path_exists(path: str) -> bool:
    text = str(path).strip()
    return bool(text) and Path(text).exists()


def get_default_imagery_png_dir(imagery_source: str) -> Path:
    try:
        return IMAGERY_SOURCE_PNG_DIRS[imagery_source]
    except KeyError as exc:
        raise ValueError(
            f"Unknown imagery_source '{imagery_source}'. "
            f"Choose from: {sorted(IMAGERY_SOURCE_PNG_DIRS)}"
        ) from exc


def get_imagery_source_label(imagery_source: str) -> str:
    return IMAGERY_SOURCE_LABELS.get(imagery_source, imagery_source.replace("_", " "))


def load_pairing_manifest(path: Path | str = DEFAULT_PAIRING_CSV) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pairing manifest not found: {path}")

    df = pd.read_csv(path)
    required = {
        "FilePath",
        "loc_id",
        "ground_date_time",
        "Latitude",
        "Longitude",
        "ground_image_path",
        "ground_image_exists",
        "sentinel_image_path",
        "sentinel_exists",
        "naip_image_path",
        "naip_exists",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Pairing manifest {path} missing required columns: {missing}")

    df = df.copy()
    df["FilePath"] = df["FilePath"].astype(str)
    df["loc_id"] = df["loc_id"].astype(str)
    df["ground_date_time"] = pd.to_datetime(df["ground_date_time"], errors="coerce")

    for path_col, exists_col in [
        ("ground_image_path", "ground_image_exists"),
        ("sentinel_image_path", "sentinel_exists"),
        ("naip_image_path", "naip_exists"),
    ]:
        df[path_col] = df[path_col].fillna("").astype(str)
        df[exists_col] = _coerce_exists(df[exists_col], df[path_col])

    return df


def build_pairing_filepath_lookup(pairing_df: pd.DataFrame) -> pd.DataFrame:
    lookup = pairing_df.copy()
    lookup["FilePath"] = lookup["FilePath"].astype(str)
    lookup = lookup.drop_duplicates(subset="FilePath", keep="last")
    return lookup.set_index("FilePath")


def build_pairing_image_lookup(
    pairing_df: pd.DataFrame,
    path_column: str,
    exists_column: str,
    require_complete_triplet: bool = False,
) -> pd.DataFrame:
    subset = pairing_df.copy()
    if require_complete_triplet:
        subset = subset[
            subset["ground_image_exists"]
            & subset["sentinel_exists"]
            & subset["naip_exists"]
        ].copy()
        subset = subset[
            subset["ground_image_path"].map(_path_exists)
            & subset["sentinel_image_path"].map(_path_exists)
            & subset["naip_image_path"].map(_path_exists)
        ].copy()

    subset = subset[subset[path_column].astype(str).str.strip() != ""].copy()
    subset = subset.sort_values("ground_date_time", na_position="last")
    subset = subset.drop_duplicates(subset="loc_id", keep="first")

    if subset.empty:
        return pd.DataFrame(columns=["image_path", "image_exists", "Latitude", "Longitude"])

    lookup = subset.set_index("loc_id")[[path_column, exists_column, "Latitude", "Longitude"]].rename(
        columns={path_column: "image_path", exists_column: "image_exists"}
    )
    lookup["image_path"] = lookup["image_path"].fillna("").astype(str)
    lookup["image_exists"] = _coerce_exists(
        lookup["image_exists"], lookup["image_path"]
    ) & lookup["image_path"].map(_path_exists)
    return lookup[["image_path", "image_exists", "Latitude", "Longitude"]]


def load_imagery_lookup(
    imagery_source: str,
    png_dir: Path | str | None = None,
    pairing_csv: Path | str = DEFAULT_PAIRING_CSV,
) -> pd.DataFrame:
    if imagery_source in PAIRING_PATH_COLUMNS:
        pairing_df = load_pairing_manifest(pairing_csv)
        return build_pairing_image_lookup(
            pairing_df,
            PAIRING_PATH_COLUMNS[imagery_source],
            PAIRING_EXISTS_COLUMNS[imagery_source],
        )

    png_dir = Path(png_dir) if png_dir is not None else get_default_imagery_png_dir(imagery_source)
    df = pd.read_pickle(cache_path / "wi_blank_images.pkl")
    if df.empty:
        raise RuntimeError("No cached Wildlife Insights blank images found.")

    required = {"loc_id", "Date_Time", "Latitude", "Longitude"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"wi_blank_images.pkl missing required columns: {missing}")

    lookup = df.copy()
    lookup["loc_id"] = lookup["loc_id"].astype(str)
    lookup["Date_Time"] = pd.to_datetime(lookup["Date_Time"], errors="coerce")
    lookup = lookup.sort_values("Date_Time", na_position="last")
    lookup = lookup.drop_duplicates(subset="loc_id", keep="first")
    lookup["image_path"] = lookup["loc_id"].map(lambda lid: str(png_dir / f"{lid}.png"))
    lookup["image_exists"] = lookup["image_path"].map(lambda p: Path(p).exists())
    return lookup.set_index("loc_id")[["image_path", "image_exists", "Latitude", "Longitude"]]


def lookup_to_path_map(image_lookup: pd.DataFrame) -> dict[str, str]:
    if image_lookup.empty or "image_path" not in image_lookup.columns:
        return {}
    return {str(loc_id): str(row["image_path"]) for loc_id, row in image_lookup.iterrows()}
