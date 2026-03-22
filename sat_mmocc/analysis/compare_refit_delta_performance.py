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
"""Build per-species delta tables for satellite and camera-trap refits.

This script compares:
  - Sentinel / NAIP GRAFT satellite refits against the sat+env baseline
  - camera-trap descriptor refits against the image+sat+env baseline
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from compare_graft_sat_env_performance import (  # sibling module import
    PRIMARY_METRICS,
    build_fit_results_path,
    load_fit_record,
    load_final_species_ids,
    load_species_ids,
)
from sat_mmocc.config import default_image_backbone, default_sat_backbone
from sat_mmocc.utils import get_scientific_taxon_map, get_taxon_map

DEFAULT_OUTPUT_PREFIX = (
    Path(__file__).resolve().parent / "outputs" / "compare_refit_delta_performance"
)


@dataclass(frozen=True)
class ExperimentSpec:
    label: str
    display_name: str
    modalities: tuple[str, ...]
    image_backbone: str | None
    sat_backbone: str | None
    domain: str
    reference_label: str | None = None


def build_experiment_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            label="sat_env_baseline",
            display_name="Sat+env baseline",
            modalities=("covariates", "sat"),
            image_backbone=None,
            sat_backbone=default_sat_backbone,
            domain="satellite",
        ),
        ExperimentSpec(
            label="graft_visdiff_sentinel",
            display_name="GRAFT Sentinel refit",
            modalities=("covariates", "sat"),
            image_backbone=default_image_backbone,
            sat_backbone="graft_visdiff_sentinel",
            domain="satellite",
            reference_label="sat_env_baseline",
        ),
        ExperimentSpec(
            label="graft_visdiff_naip",
            display_name="GRAFT NAIP refit",
            modalities=("covariates", "sat"),
            image_backbone=default_image_backbone,
            sat_backbone="graft_visdiff_naip",
            domain="satellite",
            reference_label="sat_env_baseline",
        ),
        ExperimentSpec(
            label="image_sat_env_baseline",
            display_name="Image+sat+env baseline",
            modalities=("covariates", "image", "sat"),
            image_backbone=default_image_backbone,
            sat_backbone=default_sat_backbone,
            domain="camera_trap",
        ),
        ExperimentSpec(
            label="clip_visdiff",
            display_name="Camera-trap VisDiff refit",
            modalities=("covariates", "image", "sat"),
            image_backbone="visdiff_clip",
            sat_backbone=default_sat_backbone,
            domain="camera_trap",
            reference_label="image_sat_env_baseline",
        ),
        ExperimentSpec(
            label="clip_expert",
            display_name="Camera-trap expert refit",
            modalities=("covariates", "image", "sat"),
            image_backbone="expert_clip",
            sat_backbone=default_sat_backbone,
            domain="camera_trap",
            reference_label="image_sat_env_baseline",
        ),
    ]


def build_results_table(
    specs: Sequence[ExperimentSpec],
    species_ids: Sequence[str],
    final_species_ids: set[str] | None = None,
) -> pd.DataFrame:
    scientific_map = get_scientific_taxon_map()
    common_map = {str(k): str(v) for k, v in get_taxon_map().items()}
    final_species_ids = final_species_ids or set()
    rows: list[dict[str, object]] = []

    for spec in specs:
        for taxon_id in species_ids:
            taxon_id = str(taxon_id)
            path = build_fit_results_path(
                taxon_id=taxon_id,
                modalities=spec.modalities,
                image_backbone=spec.image_backbone,
                sat_backbone=spec.sat_backbone,
            )
            row: dict[str, object] = {
                "taxon_id": taxon_id,
                "scientific_name": scientific_map.get(taxon_id),
                "common_name": common_map.get(taxon_id),
                "is_final_species": taxon_id in final_species_ids,
                "experiment_label": spec.label,
                "experiment_display_name": spec.display_name,
                "domain": spec.domain,
                "reference_label": spec.reference_label,
                "modalities": ",".join(spec.modalities),
                "image_backbone": spec.image_backbone,
                "sat_backbone": spec.sat_backbone,
                "fit_result_path": str(path),
                "fit_result_exists": path.exists(),
            }
            if path.exists():
                record = load_fit_record(path)
                row["scientific_name"] = record.get(
                    "scientific_name", row["scientific_name"]
                )
                row["common_name"] = record.get("common_name", row["common_name"])
                for metric in PRIMARY_METRICS:
                    row[metric] = record.get(metric)
            rows.append(row)

    df = pd.DataFrame(rows)
    for metric in PRIMARY_METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def build_delta_table(
    results_df: pd.DataFrame,
    specs: Sequence[ExperimentSpec],
) -> pd.DataFrame:
    spec_by_label = {spec.label: spec for spec in specs}
    delta_rows: list[dict[str, object]] = []

    for spec in specs:
        if spec.reference_label is None:
            continue
        reference_spec = spec_by_label[spec.reference_label]
        reference_df = results_df[
            (results_df["experiment_label"] == reference_spec.label)
            & (results_df["fit_result_exists"])
        ].copy()
        target_df = results_df[
            (results_df["experiment_label"] == spec.label)
            & (results_df["fit_result_exists"])
        ].copy()
        overlap = sorted(set(reference_df["taxon_id"]) & set(target_df["taxon_id"]))
        if not overlap:
            continue

        merged = (
            reference_df.set_index("taxon_id").loc[overlap]
            .join(
                target_df.set_index("taxon_id").loc[overlap],
                lsuffix="_reference",
                rsuffix="_target",
            )
            .reset_index()
        )

        for _, row in merged.iterrows():
            record: dict[str, object] = {
                "taxon_id": row["taxon_id"],
                "scientific_name": row.get("scientific_name_target")
                or row.get("scientific_name_reference"),
                "common_name": row.get("common_name_target")
                or row.get("common_name_reference"),
                "is_final_species": bool(row.get("is_final_species_target")),
                "domain": spec.domain,
                "reference_label": reference_spec.label,
                "reference_display_name": reference_spec.display_name,
                "target_label": spec.label,
                "target_display_name": spec.display_name,
                "reference_modalities": ",".join(reference_spec.modalities),
                "target_modalities": ",".join(spec.modalities),
                "reference_image_backbone": reference_spec.image_backbone,
                "target_image_backbone": spec.image_backbone,
                "reference_sat_backbone": reference_spec.sat_backbone,
                "target_sat_backbone": spec.sat_backbone,
                "reference_fit_result_path": row.get("fit_result_path_reference"),
                "target_fit_result_path": row.get("fit_result_path_target"),
            }
            for metric in PRIMARY_METRICS:
                reference_value = pd.to_numeric(
                    pd.Series([row.get(f"{metric}_reference")]), errors="coerce"
                ).iloc[0]
                target_value = pd.to_numeric(
                    pd.Series([row.get(f"{metric}_target")]), errors="coerce"
                ).iloc[0]
                record[f"reference_{metric}"] = reference_value
                record[f"target_{metric}"] = target_value
                record[f"delta_{metric}"] = target_value - reference_value
            delta_rows.append(record)

    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        delta_df = delta_df.sort_values(
            ["domain", "target_label", "common_name", "taxon_id"],
            na_position="last",
        )
    return delta_df


def build_summary_table(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame(
            columns=[
                "domain",
                "target_label",
                "target_display_name",
                "n_species",
            ]
        )

    rows: list[dict[str, object]] = []
    group_cols = ["domain", "target_label", "target_display_name"]
    for keys, group in delta_df.groupby(group_cols, dropna=False):
        domain, target_label, target_display_name = keys
        row: dict[str, object] = {
            "domain": domain,
            "target_label": target_label,
            "target_display_name": target_display_name,
            "n_species": int(group["taxon_id"].nunique()),
        }
        for metric in PRIMARY_METRICS:
            delta_col = f"delta_{metric}"
            if delta_col not in group.columns:
                continue
            values = pd.to_numeric(group[delta_col], errors="coerce")
            row[f"mean_{delta_col}"] = values.mean()
            row[f"median_{delta_col}"] = values.median()
            row[f"num_improved_{metric}"] = int((values > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["domain", "target_label"])


def write_outputs(
    results_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_prefix: Path,
) -> tuple[Path, Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    results_path = output_prefix.with_name(f"{output_prefix.name}_results.csv")
    delta_path = output_prefix.with_name(f"{output_prefix.name}_delta.csv")
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.csv")
    results_df.to_csv(results_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return results_path, delta_path, summary_path


def main(
    species_ids: str | Sequence[str] | None = None,
    species_ids_file: str | Path | None = None,
    final_species_ids_file: str | Path | None = None,
    output_prefix: str | Path = DEFAULT_OUTPUT_PREFIX,
) -> None:
    selected_species = load_species_ids(species_ids, species_ids_file)
    final_species_ids = load_final_species_ids(final_species_ids_file)
    specs = build_experiment_specs()
    results_df = build_results_table(specs, selected_species, final_species_ids)
    delta_df = build_delta_table(results_df, specs)
    summary_df = build_summary_table(delta_df)
    results_path, delta_path, summary_path = write_outputs(
        results_df, delta_df, summary_df, Path(output_prefix)
    )

    print("Wrote refit comparison tables:")
    print(f"  results: {results_path}")
    print(f"  delta:   {delta_path}")
    print(f"  summary: {summary_path}")

    availability = (
        results_df.groupby(["domain", "experiment_label"], as_index=False)
        .agg(
            requested_species=("taxon_id", "size"),
            available_species=("fit_result_exists", "sum"),
        )
        .sort_values(["domain", "experiment_label"])
    )
    print("\nAvailability:")
    print(availability.to_string(index=False))

    if not summary_df.empty:
        preview_cols = [
            col
            for col in (
                "domain",
                "target_display_name",
                "n_species",
                "mean_delta_lppd_test_norm",
                "mean_delta_biolith_ap_test",
            )
            if col in summary_df.columns
        ]
        print("\nSummary preview:")
        print(summary_df[preview_cols].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-species performance deltas for satellite and camera-trap refits."
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
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help=(
            "Prefix used to write <prefix>_results.csv, <prefix>_delta.csv, "
            "and <prefix>_summary.csv."
        ),
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
    args = parser.parse_args()
    main(
        species_ids=args.species_ids,
        species_ids_file=args.species_ids_file,
        final_species_ids_file=args.final_species_ids_file,
        output_prefix=args.output_prefix,
    )
