#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "numpy",
#     "pandas",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Compare baseline sat+env performance against Sentinel and NAIP GRAFT refits.

This script loads cached fit result pickles for three default experiment variants:
  - original sat+env baseline
  - Sentinel GRAFT refit
  - NAIP GRAFT refit

It writes:
  - a long-form CSV with one row per species/model pair
  - a wide-form CSV with one row per species
  - a summary CSV over the species with all requested models available
"""

from __future__ import annotations

import argparse
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from sat_mmocc.config import cache_path, default_image_backbone, default_sat_backbone
from sat_mmocc.utils import (
    experiment_to_filename,
    get_focal_species_ids,
    get_scientific_taxon_map,
    get_taxon_map,
)

DEFAULT_MODALITIES = ("covariates", "sat")
DEFAULT_OUTPUT_PREFIX = (
    Path(__file__).resolve().parent / "outputs" / "p4" / "p4"
)

PRIMARY_METRICS = [
    "lppd_test",
    "lppd_null_test",
    "lppd_oracle_test",
    "lppd_test_norm",
    "biolith_ap_test",
    "biolith_roc_auc_test",
    "lr_map_test",
    "lr_precision_test",
    "lr_recall_test",
    "lr_f1_test",
    "lr_mcc_test",
]

VALID_IMAGERY_SOURCES = frozenset({"sentinel", "naip", "sentinel_v_graft", "naip_v_graft"})

IMAGERY_SOURCE_VISDIFF_FILES: dict[str, Path] = {
    "sentinel_v_graft": cache_path / "visdiff_sentinel_v_graft_descriptions_p4.csv",
    "naip_v_graft": cache_path / "visdiff_naip_v_graft_descriptions_p4.csv",
}


def _get_descriptor_output_tag(imagery_source: str) -> str:
    """Return the compact tag embedded in the VisDiff CSV filename (e.g. '_prompt2')."""
    path = IMAGERY_SOURCE_VISDIFF_FILES.get(imagery_source)
    if path is None:
        return ""
    match = re.search(r"_(p\d+|prompt\d+)$", path.stem, flags=re.IGNORECASE)
    if match is not None:
        return f"_{match.group(1).lower()}"
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", path.stem).strip("_").lower()
    prefix = f"visdiff_{imagery_source}".lower()
    if sanitized.startswith(prefix):
        sanitized = sanitized[len(prefix):].strip("_")
    return f"_{sanitized}" if sanitized else ""


@dataclass(frozen=True)
class ComparisonSpec:
    label: str
    display_name: str
    modalities: tuple[str, ...]
    image_backbone: str | None
    sat_backbone: str | None
    imagery_source: str | None = None
    descriptor_source: str | None = None


def parse_csv_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def load_species_ids_from_path(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Species IDs file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "taxon_id" not in df.columns:
            raise ValueError(
                f"Provided species CSV {path} is missing required column 'taxon_id'."
            )
        return df["taxon_id"].dropna().astype(str).tolist()

    text = path.read_text().strip()
    if not text:
        return []
    return parse_csv_list(text.replace("\n", ","))


def load_species_ids(
    species_ids: str | Sequence[str] | None = None,
    species_ids_file: str | Path | None = None,
) -> list[str]:
    if species_ids is not None and species_ids_file is not None:
        raise ValueError("Provide either species_ids or species_ids_file, not both.")
    if species_ids_file is not None:
        return load_species_ids_from_path(species_ids_file)
    if species_ids is not None:
        return parse_csv_list(species_ids)
    return get_focal_species_ids()


def load_final_species_ids(
    species_ids_file: str | Path | None = None,
) -> set[str]:
    final_species_path = (
        Path(species_ids_file)
        if species_ids_file is not None
        else cache_path / "final_species_ids.txt"
    )
    return set(load_species_ids_from_path(final_species_path))


def normalize_imagery_source(imagery_source: str | None) -> str:
    value = (imagery_source or "sentinel").strip().lower()
    if value not in VALID_IMAGERY_SOURCES:
        raise ValueError(
            f"Unknown imagery_source '{imagery_source}'. "
            f"Choose from: {sorted(VALID_IMAGERY_SOURCES)}"
        )
    return value


def get_graft_refit_sat_backbone(
    descriptor_source: str,
    imagery_source: str = "sentinel",
) -> str:
    source = descriptor_source.strip().lower()
    if source == "expert":
        return "graft_expert"
    if source != "visdiff":
        raise ValueError(
            f"Unknown descriptor source '{descriptor_source}'. "
            "Choose from: ['expert', 'visdiff']"
        )
    norm = normalize_imagery_source(imagery_source)
    return f"graft_visdiff_{norm}{_get_descriptor_output_tag(norm)}"



def build_fit_results_path(
    taxon_id: str,
    modalities: Sequence[str],
    image_backbone: str | None,
    sat_backbone: str | None,
) -> Path:
    filename = experiment_to_filename(
        taxon_id, set(modalities), image_backbone, sat_backbone, "pkl"
    )
    return cache_path / "fit_results" / filename


def parse_modalities(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_MODALITIES
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    if not items:
        raise ValueError("At least one modality is required.")
    return tuple(sorted(set(items)))


def build_specs(modalities: Sequence[str]) -> list[ComparisonSpec]:
    modalities = tuple(sorted(set(modalities)))
    baseline_image_backbone = default_image_backbone if "image" in modalities else None
    return [
        ComparisonSpec(
            label="original_sat_env",
            display_name="Original sat+env",
            modalities=modalities,
            image_backbone=baseline_image_backbone,
            sat_backbone=default_sat_backbone,
        ),
        ComparisonSpec(
            label="refit_sentinel",
            display_name="Refit GRAFT Sentinel",
            modalities=modalities,
            image_backbone=default_image_backbone,
            sat_backbone=get_graft_refit_sat_backbone("visdiff", "sentinel_v_graft"),
            imagery_source="sentinel_v_graft",
            descriptor_source="visdiff",
        ),
        ComparisonSpec(
            label="refit_naip",
            display_name="Refit GRAFT NAIP",
            modalities=modalities,
            image_backbone=default_image_backbone,
            sat_backbone=get_graft_refit_sat_backbone("visdiff", "naip_v_graft"),
            imagery_source="naip_v_graft",
            descriptor_source="visdiff",
        ),
    ]


def compute_lppd_test_norm(record: dict[str, Any]) -> float:
    required = ("lppd_test", "lppd_null_test", "lppd_oracle_test")
    if any(key not in record for key in required):
        return float("nan")
    lppd_test = pd.to_numeric(pd.Series([record["lppd_test"]]), errors="coerce").iloc[0]
    lppd_null = pd.to_numeric(
        pd.Series([record["lppd_null_test"]]), errors="coerce"
    ).iloc[0]
    lppd_oracle = pd.to_numeric(
        pd.Series([record["lppd_oracle_test"]]), errors="coerce"
    ).iloc[0]
    denom = lppd_oracle - lppd_null
    if not np.isfinite(denom) or np.isclose(denom, 0.0):
        return float("nan")
    return float((lppd_test - lppd_null) / denom)


def load_fit_record(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        record = pickle.load(handle)
    record.pop("mcmc_samples", None)
    record["lppd_test_norm"] = compute_lppd_test_norm(record)
    return record


def build_long_table(
    specs: Sequence[ComparisonSpec],
    species_ids: Sequence[str],
    final_species_ids: set[str] | None = None,
    error_on_missing: bool = False,
) -> pd.DataFrame:
    scientific_map = get_scientific_taxon_map()
    common_map = {str(k): str(v) for k, v in get_taxon_map().items()}
    final_species_ids = final_species_ids or set()
    rows: list[dict[str, Any]] = []

    for taxon_id in species_ids:
        taxon_id = str(taxon_id)
        for spec in specs:
            path = build_fit_results_path(
                taxon_id,
                spec.modalities,
                spec.image_backbone,
                spec.sat_backbone,
            )
            row: dict[str, Any] = {
                "taxon_id": taxon_id,
                "scientific_name": scientific_map.get(taxon_id),
                "common_name": common_map.get(taxon_id),
                "is_final_species": taxon_id in final_species_ids,
                "model_label": spec.label,
                "model_display_name": spec.display_name,
                "modalities": ",".join(spec.modalities),
                "image_backbone": spec.image_backbone,
                "sat_backbone": spec.sat_backbone,
                "imagery_source": spec.imagery_source,
                "descriptor_source": spec.descriptor_source,
                "fit_result_path": str(path),
                "fit_result_exists": path.exists(),
            }
            if not path.exists():
                if error_on_missing:
                    raise FileNotFoundError(f"Missing fit results at {path}")
                rows.append(row)
                continue

            record = load_fit_record(path)
            row["scientific_name"] = record.get(
                "scientific_name", row["scientific_name"]
            )
            row["common_name"] = record.get("common_name", row["common_name"])
            row["descriptor_source"] = record.get(
                "descriptor_source", row["descriptor_source"]
            )
            row["imagery_source"] = record.get("imagery_source", row["imagery_source"])
            row["sat_backbone_data"] = record.get("sat_backbone_data")
            for metric in PRIMARY_METRICS:
                row[metric] = record.get(metric)
            rows.append(row)

    df = pd.DataFrame(rows)
    for metric in PRIMARY_METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def flatten_wide_column_name(column: object) -> str:
    if isinstance(column, str):
        return column
    if isinstance(column, tuple):
        parts = [str(part) for part in column if part not in ("", None)]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return f"{parts[1]}__{parts[0]}"
    return str(column)


def build_wide_table(long_df: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["taxon_id", "scientific_name", "common_name", "is_final_species"]
    value_cols = ["fit_result_exists", *PRIMARY_METRICS]
    if long_df.empty:
        return pd.DataFrame(columns=index_cols)
    available_cols = [col for col in value_cols if col in long_df.columns]

    wide = (
        long_df[index_cols + ["model_label", *available_cols]]
        .pivot_table(
            index=index_cols,
            columns="model_label",
            values=available_cols,
            aggfunc="first",
        )
        .reset_index()
    )

    wide.columns = [flatten_wide_column_name(col) for col in wide.columns]

    baseline_label = "original_sat_env"
    for target_label in ("refit_sentinel", "refit_naip"):
        for metric in PRIMARY_METRICS:
            baseline_col = f"{baseline_label}__{metric}"
            target_col = f"{target_label}__{metric}"
            delta_col = f"{target_label}__delta_vs_{baseline_label}__{metric}"
            if baseline_col in wide.columns and target_col in wide.columns:
                wide[delta_col] = wide[target_col] - wide[baseline_col]

    sort_cols = [col for col in ("common_name", "taxon_id") if col in wide.columns]
    if sort_cols:
        wide = wide.sort_values(sort_cols, na_position="last")
    return wide


def build_summary_table(long_df: pd.DataFrame, specs: Sequence[ComparisonSpec]) -> pd.DataFrame:
    required_labels = [spec.label for spec in specs]
    available = long_df[long_df["fit_result_exists"].fillna(False)].copy()
    overlap_ids = (
        available.groupby("taxon_id")["model_label"]
        .nunique()
        .loc[lambda s: s == len(required_labels)]
        .index
    )
    overlap = available[available["taxon_id"].isin(overlap_ids)].copy()

    rows: list[dict[str, Any]] = []
    for spec in specs:
        subset = overlap[overlap["model_label"] == spec.label].copy()
        row: dict[str, Any] = {
            "model_label": spec.label,
            "model_display_name": spec.display_name,
            "n_species_overlap": int(subset["taxon_id"].nunique()),
        }
        for metric in PRIMARY_METRICS:
            if metric not in subset.columns:
                continue
            values = pd.to_numeric(subset[metric], errors="coerce")
            row[f"mean_{metric}"] = values.mean()
            row[f"median_{metric}"] = values.median()
        rows.append(row)

    baseline = overlap[overlap["model_label"] == "original_sat_env"][
        ["taxon_id", *[m for m in PRIMARY_METRICS if m in overlap.columns]]
    ].rename(columns={metric: f"baseline_{metric}" for metric in PRIMARY_METRICS})

    for spec in specs:
        if spec.label == "original_sat_env":
            continue
        target = overlap[overlap["model_label"] == spec.label]
        merged = baseline.merge(target, on="taxon_id", how="inner")
        row = {
            "model_label": f"{spec.label}_vs_original_sat_env",
            "model_display_name": f"{spec.display_name} vs Original sat+env",
            "n_species_overlap": int(merged["taxon_id"].nunique()),
        }
        for metric in PRIMARY_METRICS:
            baseline_col = f"baseline_{metric}"
            if baseline_col not in merged.columns or metric not in merged.columns:
                continue
            delta = pd.to_numeric(merged[metric], errors="coerce") - pd.to_numeric(
                merged[baseline_col], errors="coerce"
            )
            row[f"mean_delta_{metric}"] = delta.mean()
            row[f"median_delta_{metric}"] = delta.median()
            row[f"num_improved_{metric}"] = int((delta > 0).sum())
        rows.append(row)

    return pd.DataFrame(rows)


def write_outputs(
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_prefix: Path,
) -> tuple[Path, Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    long_path = output_prefix.with_name(f"{output_prefix.name}_long.csv")
    wide_path = output_prefix.with_name(f"{output_prefix.name}_wide.csv")
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.csv")
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return long_path, wide_path, summary_path


def main(
    species_ids: str | Sequence[str] | None = None,
    species_ids_file: str | Path | None = None,
    final_species_ids_file: str | Path | None = None,
    modalities: str | Sequence[str] | None = None,
    output_prefix: str | Path = DEFAULT_OUTPUT_PREFIX,
    error_on_missing: bool = False,
) -> None:
    selected_species = load_species_ids(species_ids, species_ids_file)
    final_species_ids = load_final_species_ids(final_species_ids_file)
    specs = build_specs(parse_modalities(modalities))
    long_df = build_long_table(
        specs,
        species_ids=selected_species,
        final_species_ids=final_species_ids,
        error_on_missing=error_on_missing,
    )
    wide_df = build_wide_table(long_df)
    summary_df = build_summary_table(long_df, specs)
    long_path, wide_path, summary_path = write_outputs(
        long_df, wide_df, summary_df, Path(output_prefix)
    )

    print("Wrote predictive performance tables:")
    print(f"  long:    {long_path}")
    print(f"  wide:    {wide_path}")
    print(f"  summary: {summary_path}")

    availability = (
        long_df.groupby("model_label", as_index=False)
        .agg(
            requested_species=("taxon_id", "size"),
            available_species=("fit_result_exists", "sum"),
        )
    )
    print("\nAvailability:")
    print(availability.to_string(index=False))

    if not summary_df.empty:
        preview_cols = [
            col
            for col in (
                "model_display_name",
                "n_species_overlap",
                "mean_lppd_test_norm",
                "mean_delta_lppd_test_norm",
            )
            if col in summary_df.columns
        ]
        print("\nSummary preview:")
        print(summary_df[preview_cols].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline sat+env fit performance with Sentinel and NAIP GRAFT refits."
        )
    )
    parser.add_argument(
        "--species-ids",
        type=str,
        default=None,
        help="Optional comma-separated taxon IDs.",
    )
    parser.add_argument(
        "--species-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional species file. Accepts either a text file with comma/newline-"
            "separated taxon IDs or a CSV with a 'taxon_id' column. If omitted, "
            "uses get_focal_species_ids()."
        ),
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="covariates,sat",
        help="Comma-separated modalities to compare. Default: covariates,sat.",
    )
    parser.add_argument(
        "--final-species-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional file containing the final analysis species IDs. Defaults to "
            "CACHE_PATH/final_species_ids.txt."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix used to write <prefix>_long.csv, <prefix>_wide.csv, and <prefix>_summary.csv.",
    )
    parser.add_argument(
        "--error-on-missing",
        action="store_true",
        help="Raise an error instead of writing missing fit results as empty rows.",
    )
    args = parser.parse_args()
    main(
        species_ids=args.species_ids,
        species_ids_file=args.species_ids_file,
        final_species_ids_file=args.final_species_ids_file,
        modalities=args.modalities,
        output_prefix=args.output_prefix,
        error_on_missing=args.error_on_missing,
    )
