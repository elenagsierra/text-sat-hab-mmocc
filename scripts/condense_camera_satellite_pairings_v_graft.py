#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "fire",
#     "pandas",
# ]
#
# [tool.uv.sources]
# mmocc = { path = ".." }
# ///
"""Condense the GRAFT camera/satellite pairing manifest to rows used downstream.

This helper mirrors the row-selection behavior in the current codebase:

- ``sat_mmocc/steps/16_extract_graft_features.py``
  uses the latest manifest record per ``FilePath`` when a source-specific PNG is
  available for that blank-image row.
- ``sat_mmocc/steps/08c_visdiff_compare_sources.py``
  uses the earliest complete image triplet per ``loc_id``.

The output CSV keeps only rows used by at least one of those paths and adds
boolean flags explaining why each retained row is present.
"""

from __future__ import annotations

from pathlib import Path

import fire
import pandas as pd

from mmocc.config import cache_path, wildlife_insights_test_project_ids

DEFAULT_INPUT_CSV = cache_path / "camera_satellite_pairings_v_graft.csv"
DEFAULT_OUTPUT_CSV = cache_path / "camera_satellite_pairings_v_graft_used_by_steps.csv"
DEFAULT_BLANK_IMAGES_PKL = cache_path / "wi_blank_images.pkl"

OUTPUT_COLUMNS = [
    "FilePath",
    "loc_id",
    "project_id",
    "project_split",
    "is_train_project",
    "is_test_project",
    "ground_date_time",
    "Latitude",
    "Longitude",
    "ground_image_path",
    "ground_image_exists",
    "sentinel_image_path",
    "sentinel_exists",
    "naip_image_path",
    "naip_exists",
    "is_latest_filepath_record",
    "used_by_step16_sentinel_v_graft",
    "used_by_step16_naip_v_graft",
    "used_by_step08c_complete_triplet",
    "used_by_any_step",
]

VALID_SPLITS = {"all", "train", "test"}


def _require_columns(df: pd.DataFrame, required: set[str], label: Path) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _as_path_exists(path_value: object) -> bool:
    if pd.isna(path_value):
        return False
    text = str(path_value).strip()
    return bool(text) and Path(text).exists()


def _load_pairing_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Pairing manifest not found: {path}")

    df = pd.read_csv(path).copy()
    _require_columns(
        df,
        {
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
        },
        path,
    )

    df["FilePath"] = df["FilePath"].astype(str)
    df["loc_id"] = df["loc_id"].astype(str)
    df["ground_date_time"] = pd.to_datetime(df["ground_date_time"], errors="coerce")
    df["__row_idx"] = range(len(df))
    df["is_latest_filepath_record"] = ~df.duplicated(subset="FilePath", keep="last")
    return df


def _load_blank_filepaths(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Blank-image dataframe not found: {path}")

    blank_df = pd.read_pickle(path)
    _require_columns(blank_df, {"FilePath"}, path)
    return set(blank_df["FilePath"].astype(str))


def _add_project_split_columns(pairing_df: pd.DataFrame) -> pd.DataFrame:
    df = pairing_df.copy()
    df["project_id"] = df["loc_id"].astype(str).str.split("___").str[0]
    df["is_test_project"] = df["project_id"].isin(wildlife_insights_test_project_ids)
    df["is_train_project"] = ~df["is_test_project"]
    df["project_split"] = df["is_test_project"].map({True: "test", False: "train"})
    return df


def _mark_step16_rows(pairing_df: pd.DataFrame, blank_filepaths: set[str]) -> pd.DataFrame:
    df = pairing_df.copy()
    df["used_by_step16_sentinel_v_graft"] = False
    df["used_by_step16_naip_v_graft"] = False

    eligible = df["is_latest_filepath_record"] & df["FilePath"].isin(blank_filepaths)

    sentinel_ok = (
        eligible
        & df["sentinel_image_path"].fillna("").astype(str).str.strip().ne("")
        & df["sentinel_exists"].fillna(False).astype(bool)
        & df["sentinel_image_path"].apply(_as_path_exists)
    )
    naip_ok = (
        eligible
        & df["naip_image_path"].fillna("").astype(str).str.strip().ne("")
        & df["naip_exists"].fillna(False).astype(bool)
        & df["naip_image_path"].apply(_as_path_exists)
    )

    df.loc[sentinel_ok, "used_by_step16_sentinel_v_graft"] = True
    df.loc[naip_ok, "used_by_step16_naip_v_graft"] = True
    return df


def _mark_step08c_rows(pairing_df: pd.DataFrame) -> pd.DataFrame:
    df = pairing_df.copy()
    df["used_by_step08c_complete_triplet"] = False

    complete_triplets = df[
        df["ground_image_exists"].fillna(False).astype(bool)
        & df["sentinel_exists"].fillna(False).astype(bool)
        & df["naip_exists"].fillna(False).astype(bool)
        & df["ground_image_path"].fillna("").astype(str).str.strip().ne("")
        & df["sentinel_image_path"].fillna("").astype(str).str.strip().ne("")
        & df["naip_image_path"].fillna("").astype(str).str.strip().ne("")
    ].copy()

    if complete_triplets.empty:
        return df

    complete_triplets = complete_triplets.sort_values(
        ["ground_date_time", "__row_idx"], na_position="last"
    )
    selected_rows = complete_triplets.drop_duplicates(subset="loc_id", keep="first")[
        "__row_idx"
    ]
    df.loc[df["__row_idx"].isin(selected_rows), "used_by_step08c_complete_triplet"] = True
    return df


def main(
    input_csv: str | Path = DEFAULT_INPUT_CSV,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    blank_images_pkl: str | Path = DEFAULT_BLANK_IMAGES_PKL,
    split: str = "all",
    drop_unused: bool = True,
) -> None:
    """Write a condensed pairing CSV keyed to current downstream usage."""

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    blank_images_pkl = Path(blank_images_pkl)
    split = str(split).strip().lower()
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")

    pairing_df = _load_pairing_manifest(input_csv)
    pairing_df = _add_project_split_columns(pairing_df)
    blank_filepaths = _load_blank_filepaths(blank_images_pkl)
    pairing_df = _mark_step16_rows(pairing_df, blank_filepaths)
    pairing_df = _mark_step08c_rows(pairing_df)
    pairing_df["used_by_any_step"] = (
        pairing_df["used_by_step16_sentinel_v_graft"]
        | pairing_df["used_by_step16_naip_v_graft"]
        | pairing_df["used_by_step08c_complete_triplet"]
    )

    if drop_unused:
        output_df = pairing_df[pairing_df["used_by_any_step"]].copy()
    else:
        output_df = pairing_df.copy()

    if split != "all":
        output_df = output_df[output_df["project_split"] == split].copy()

    output_df = output_df[OUTPUT_COLUMNS].sort_values(
        ["ground_date_time", "loc_id", "FilePath"], na_position="last"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    print(f"Read {len(pairing_df):,} pairing rows from {input_csv}")
    print(f"Train-project rows: {int(pairing_df['is_train_project'].sum()):,}")
    print(f"Test-project rows: {int(pairing_df['is_test_project'].sum()):,}")
    print(f"Step 16 sentinel_v_graft rows: {int(pairing_df['used_by_step16_sentinel_v_graft'].sum()):,}")
    print(f"Step 16 naip_v_graft rows: {int(pairing_df['used_by_step16_naip_v_graft'].sum()):,}")
    print(f"Step 08c complete-triplet rows: {int(pairing_df['used_by_step08c_complete_triplet'].sum()):,}")
    print(f"Applied split filter: {split}")
    print(f"Wrote {len(output_df):,} rows to {output_csv}")


if __name__ == "__main__":
    fire.Fire(main)
