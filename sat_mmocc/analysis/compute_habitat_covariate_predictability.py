"""Summarize habitat covariate predictability for a selectable result table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sat_mmocc.analysis.satellite_experiments import load_species_ids
from sat_mmocc.config import cache_path

DEFAULT_SUMMARY_FILE = (
    cache_path / "habitat_explainability" / "habitat_rs_explainability_summary.csv"
)
DEFAULT_SPECIES_IDS_FILE = cache_path / "final_species_ids.txt"


def main(
    summary_file: str | Path = DEFAULT_SUMMARY_FILE,
    metric: str = "mean_r2_sat",
    species_ids: str | None = None,
    species_ids_file: str | Path | None = DEFAULT_SPECIES_IDS_FILE,
    clip_lower: float | None = 0.0,
) -> None:
    taxon_ids = load_species_ids(species_ids, species_ids_file)
    df = pd.read_csv(summary_file)
    if "taxon_id" not in df:
        raise ValueError(f"{summary_file} is missing required column 'taxon_id'.")
    if metric not in df:
        raise ValueError(f"{summary_file} is missing requested metric column '{metric}'.")

    df["taxon_id"] = df["taxon_id"].astype(str)
    filtered = df[df["taxon_id"].isin(taxon_ids)].copy()
    if filtered.empty:
        raise RuntimeError("No rows matched the requested taxon IDs.")

    values = pd.to_numeric(filtered[metric], errors="coerce")
    if clip_lower is not None:
        print(f"Clipped mean ({metric}, lower={clip_lower:g}): {np.nanmean(values.clip(lower=clip_lower)):.4f}")
    print(f"Unclipped mean ({metric}): {np.nanmean(values):.4f}")
    print(f"Rows used: {values.notna().sum()} / {len(filtered)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize habitat covariate predictability for selected species."
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=DEFAULT_SUMMARY_FILE,
        help="CSV summary file to analyze.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mean_r2_sat",
        help="Metric column to summarize.",
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
        default=DEFAULT_SPECIES_IDS_FILE,
        help="Optional text file containing comma- or newline-separated taxon IDs.",
    )
    parser.add_argument(
        "--clip-lower",
        type=float,
        default=0.0,
        help="Optional lower bound for the clipped mean. Use a negative number if needed.",
    )
    args = parser.parse_args()
    main(
        summary_file=args.summary_file,
        metric=args.metric,
        species_ids=args.species_ids,
        species_ids_file=args.species_ids_file,
        clip_lower=args.clip_lower,
    )
