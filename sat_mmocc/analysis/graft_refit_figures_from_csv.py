#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "seaborn",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Reimplement the refit-figure style analysis from a wide comparison CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sat_mmocc.config import fig_page_width
from sat_mmocc.plot_utils import setup_matplotlib

COMPARISON_METRICS = [
    "lppd_test_norm",
    "biolith_ap_test",
    "biolith_roc_auc_test",
    "lr_map_test",
    "lr_mcc_test",
]

MODEL_SPECS = [
    ("original_sat_env", "Sat+env baseline"),
    ("refit_sentinel", "Sentinel GRAFT"),
    ("refit_naip", "NAIP GRAFT"),
]
TARGET_SPECS = MODEL_SPECS[1:]
DEFAULT_WIDE_CSV = (
    Path(__file__).resolve().parent / "outputs" / "p4" / "p4_wide.csv"
)
ROBUST_LOWER_QUANTILE = 0.05
ROBUST_UPPER_QUANTILE = 0.95
DEFAULT_LPPD_CLIP_LIMITS = (-1.0, 1.0)


def load_results(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"taxon_id", "scientific_name", "common_name", "is_final_species"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    df["taxon_id"] = df["taxon_id"].astype(str)
    df["is_final_species"] = (
        df["is_final_species"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    df = df[df["is_final_species"]].copy()
    if df.empty:
        raise RuntimeError(f"No final-species rows were found in {csv_path}.")
    return df


def metric_column(model_label: str, metric: str) -> str:
    return f"{model_label}__{metric}"


def available_column(model_label: str) -> str:
    return f"{model_label}__fit_result_exists"


def validate_model_columns(df: pd.DataFrame) -> None:
    required = []
    for model_label, _ in MODEL_SPECS:
        required.append(available_column(model_label))
        for metric in COMPARISON_METRICS:
            required.append(metric_column(model_label, metric))
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required comparison columns: {missing}")


def compute_axis_limits(
    values: np.ndarray,
    robust: bool,
    lower_quantile: float = ROBUST_LOWER_QUANTILE,
    upper_quantile: float = ROBUST_UPPER_QUANTILE,
) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0)
    if robust:
        lower = float(np.nanquantile(finite, lower_quantile))
        upper = float(np.nanquantile(finite, upper_quantile))
    else:
        lower = float(np.nanmin(finite))
        upper = float(np.nanmax(finite))
    if np.isclose(lower, upper):
        pad = 1.0 if np.isclose(lower, 0.0) else abs(lower) * 0.1
        return (lower - pad, upper + pad)
    pad = 0.08 * (upper - lower)
    return (lower + -pad, upper + pad)


def count_clipped(values: np.ndarray, limits: tuple[float, float]) -> int:
    lower, upper = limits
    finite = values[np.isfinite(values)]
    return int(((finite < lower) | (finite > upper)).sum())


def parse_clip_limits(values: list[float] | None) -> tuple[float, float] | None:
    if values is None:
        return DEFAULT_LPPD_CLIP_LIMITS
    if len(values) != 2:
        raise ValueError("Clip limits must have exactly two values: lower upper.")
    lower, upper = float(values[0]), float(values[1])
    if lower >= upper:
        raise ValueError("Clip limits must satisfy lower < upper.")
    return (lower, upper)


def align_model(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    cols = ["taxon_id", "scientific_name", "common_name"]
    model_cols = [metric_column(model_label, metric) for metric in COMPARISON_METRICS]
    available_col = available_column(model_label)
    subset = df.loc[:, cols + [available_col, *model_cols]].copy()
    subset = subset[subset[available_col].fillna(False)].copy()
    rename_map = {metric_column(model_label, metric): metric for metric in COMPARISON_METRICS}
    subset.rename(columns=rename_map, inplace=True)
    subset = subset.set_index("taxon_id")
    return subset.loc[:, ["scientific_name", "common_name", *COMPARISON_METRICS]]


def summarize_against_baseline(
    baseline_df: pd.DataFrame,
    target_df: pd.DataFrame,
    label: str,
    display_name: str,
) -> pd.DataFrame:
    overlap = baseline_df.index.intersection(target_df.index)
    merged = (
        baseline_df.loc[overlap, COMPARISON_METRICS]
        .add_suffix("_baseline")
        .join(target_df.loc[overlap, COMPARISON_METRICS].add_suffix(f"_{label}"))
    )
    rows = []
    for metric in COMPARISON_METRICS:
        deltas = merged[f"{metric}_{label}"] - merged[f"{metric}_baseline"]
        rows.append(
            {
                "model": label,
                "model_display_name": display_name,
                "metric": metric,
                "n_species": int(deltas.notna().sum()),
                "mean_delta": deltas.mean(),
                "median_delta": deltas.median(),
                "improved_%": 100.0 * (deltas > 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def build_delta_table(
    baseline_df: pd.DataFrame,
    target_df: pd.DataFrame,
    label: str,
    display_name: str,
) -> pd.DataFrame:
    overlap = baseline_df.index.intersection(target_df.index)
    merged = (
        baseline_df.loc[overlap, ["scientific_name", "common_name", *COMPARISON_METRICS]]
        .rename_axis("taxon_id")
        .join(
            target_df.loc[overlap, COMPARISON_METRICS],
            lsuffix="_baseline",
            rsuffix=f"_{label}",
        )
    )
    records = []
    for metric in COMPARISON_METRICS:
        deltas = merged[f"{metric}_{label}"] - merged[f"{metric}_baseline"]
        for taxon_id, delta in deltas.items():
            records.append(
                {
                    "taxon_id": taxon_id,
                    "scientific_name": merged.loc[taxon_id, "scientific_name"],
                    "common_name": merged.loc[taxon_id, "common_name"],
                    "metric": metric,
                    "model": label,
                    "model_display_name": display_name,
                    "delta": delta,
                    "baseline": merged.loc[taxon_id, f"{metric}_baseline"],
                    "target": merged.loc[taxon_id, f"{metric}_{label}"],
                }
            )
    return pd.DataFrame(records)


def build_absolute_metric_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    records = []
    for model_label, model_display_name in MODEL_SPECS:
        model_df = align_model(df, model_label).reset_index()
        if metric not in model_df.columns:
            continue
        for _, row in model_df.iterrows():
            records.append(
                {
                    "taxon_id": row["taxon_id"],
                    "scientific_name": row["scientific_name"],
                    "common_name": row["common_name"],
                    "model": model_label,
                    "model_display_name": model_display_name,
                    "metric": metric,
                    "value": row[metric],
                }
            )
    metric_df = pd.DataFrame(records)
    if metric_df.empty:
        return metric_df
    metric_df["value"] = pd.to_numeric(metric_df["value"], errors="coerce")
    species_order = (
        metric_df.groupby(["taxon_id", "common_name"], dropna=False)["value"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()["common_name"]
        .tolist()
    )
    metric_df["common_name"] = pd.Categorical(
        metric_df["common_name"], categories=species_order, ordered=True
    )
    return metric_df.sort_values(["common_name", "model_display_name"])


def plot_lppd_norm_performance_by_model(
    metric_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig_width = max(fig_page_width, 14)
    fig, ax = plt.subplots(figsize=(fig_width, fig_page_width / 1.5))
    sns.barplot(
        data=metric_df,
        x="common_name",
        y="value",
        hue="model_display_name",
        ax=ax,
    )
    ax.set_title("LPPD Norm Performance by Model")
    ax.set_ylabel(r"$\mathrm{LPPD}_{norm}^{test}$")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_lppd_norm_distribution(
    metric_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(fig_page_width / 1.2, fig_page_width / 1.8))
    sns.boxplot(
        data=metric_df,
        x="model_display_name",
        y="value",
        ax=ax,
    )
    sns.stripplot(
        data=metric_df,
        x="model_display_name",
        y="value",
        color="black",
        alpha=0.55,
        size=4,
        ax=ax,
    )
    ax.set_title("Distribution of LPPD Norm")
    ax.set_ylabel(r"$\mathrm{LPPD}_{norm}^{test}$")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_delta_lppd_norm_vs_original(
    delta_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_df = delta_df[delta_df["metric"] == "lppd_test_norm"].copy()
    if plot_df.empty:
        raise RuntimeError("No LPPD norm delta rows were available for plotting.")
    species_order = (
        plot_df.groupby("common_name", dropna=False)["delta"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    plot_df["common_name"] = pd.Categorical(
        plot_df["common_name"], categories=species_order, ordered=True
    )

    fig_width = max(fig_page_width, 14)
    fig, ax = plt.subplots(figsize=(fig_width, fig_page_width / 1.6))
    sns.barplot(
        data=plot_df,
        x="common_name",
        y="delta",
        hue="model_display_name",
        ax=ax,
    )
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_title("Delta LPPD Norm vs Original")
    ax.set_ylabel(r"$\Delta \mathrm{LPPD}_{norm}^{test}$")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_delta_barplot(
    delta_df: pd.DataFrame,
    output_path: Path,
    robust: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(fig_page_width, fig_page_width / 2.3))
    sns.barplot(
        data=delta_df,
        x="metric",
        y="delta",
        hue="model_display_name",
        estimator=np.mean,
        errorbar="sd",
        ax=ax,
    )
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_ylabel("Delta vs sat+env baseline")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="")
    limits = compute_axis_limits(delta_df["delta"].to_numpy(dtype=float), robust=robust)
    ax.set_ylim(limits)
    if robust:
        clipped = count_clipped(delta_df["delta"].to_numpy(dtype=float), limits)
        ax.text(
            0.01,
            0.99,
            (
                f"Zoomed to {int(100 * ROBUST_LOWER_QUANTILE)}th-"
                f"{int(100 * ROBUST_UPPER_QUANTILE)}th percentile range\n"
                f"Clipped points in view: {clipped}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_lppd_scatter(
    baseline_df: pd.DataFrame,
    targets: list[tuple[str, str, pd.DataFrame]],
    output_path: Path,
    robust: bool,
    clip_limits: tuple[float, float] | None = None,
) -> None:
    fig, axes = plt.subplots(1, len(targets), figsize=(fig_page_width, fig_page_width / 2.2))
    if len(targets) == 1:
        axes = [axes]

    for ax, (label, display_name, target_df) in zip(axes, targets):
        overlap = baseline_df.index.intersection(target_df.index)
        scatter_data = (
            baseline_df.loc[overlap, ["scientific_name", "lppd_test_norm"]]
            .rename(columns={"lppd_test_norm": "baseline"})
            .join(
                target_df.loc[overlap, ["lppd_test_norm"]].rename(
                    columns={"lppd_test_norm": "target"}
                )
            )
            .dropna(subset=["baseline", "target"])
        )
        if scatter_data.empty:
            ax.set_visible(False)
            continue

        ax.scatter(
            scatter_data["baseline"],
            scatter_data["target"],
            s=35,
            alpha=0.8,
            edgecolor="none",
        )
        all_values = scatter_data[["baseline", "target"]].to_numpy(dtype=float).ravel()
        lims = clip_limits if clip_limits is not None else compute_axis_limits(
            all_values, robust=robust
        )
        ax.plot(lims, lims, linestyle="--", color="gray", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(display_name)
        ax.set_xlabel(r"Baseline $\mathrm{LPPD}_{norm}^{test}$")
        ax.set_ylabel(rf"{display_name} $\mathrm{{LPPD}}_{{norm}}^{{test}}$")
        if robust or clip_limits is not None:
            clipped = int(
                (
                    (scatter_data["baseline"] < lims[0])
                    | (scatter_data["baseline"] > lims[1])
                    | (scatter_data["target"] < lims[0])
                    | (scatter_data["target"] > lims[1])
                ).sum()
            )
            if clip_limits is not None:
                note = (
                    f"Clipped view: [{lims[0]:.0f}, {lims[1]:.0f}]\n"
                    f"Clipped species in view: {clipped}"
                )
            else:
                note = (
                    f"Zoomed to {int(100 * ROBUST_LOWER_QUANTILE)}th-"
                    f"{int(100 * ROBUST_UPPER_QUANTILE)}th percentile range\n"
                    f"Clipped species in view: {clipped}"
                )
            ax.text(
                0.03,
                0.97,
                note,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main(
    wide_csv: str | Path = DEFAULT_WIDE_CSV,
    output_dir: str | Path | None = None,
    output_prefix: str | None = None,
    lppd_clip_limits: tuple[float, float] | None = DEFAULT_LPPD_CLIP_LIMITS,
) -> None:
    setup_matplotlib()
    wide_csv = Path(wide_csv)
    df = load_results(wide_csv)
    validate_model_columns(df)

    baseline_label, baseline_display_name = MODEL_SPECS[0]
    baseline_df = align_model(df, baseline_label)

    summaries = []
    delta_tables = []
    scatter_targets: list[tuple[str, str, pd.DataFrame]] = []
    for target_label, target_display_name in TARGET_SPECS:
        target_df = align_model(df, target_label)
        summaries.append(
            summarize_against_baseline(
                baseline_df, target_df, target_label, target_display_name
            )
        )
        delta_tables.append(
            build_delta_table(
                baseline_df, target_df, target_label, target_display_name
            )
        )
        scatter_targets.append((target_label, target_display_name, target_df))

    summary_df = pd.concat(summaries, ignore_index=True)
    delta_long = pd.concat(delta_tables, ignore_index=True)
    if delta_long.empty:
        raise RuntimeError(
            f"No overlapping baseline/refit rows were found in {wide_csv}."
        )
    absolute_lppd_df = build_absolute_metric_table(df, "lppd_test_norm")
    if absolute_lppd_df.empty:
        raise RuntimeError(f"No LPPD norm rows were available in {wide_csv}.")

    resolved_output_dir = Path(output_dir) if output_dir is not None else wide_csv.parent
    prefix = output_prefix or wide_csv.stem
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = resolved_output_dir / f"{prefix}_refit_summary.csv"
    delta_path = resolved_output_dir / f"{prefix}_refit_delta_long.csv"
    barplot_path = resolved_output_dir / f"{prefix}_refit_delta_barplot.pdf"
    barplot_full_path = resolved_output_dir / f"{prefix}_refit_delta_barplot_full_range.pdf"
    scatter_path = resolved_output_dir / f"{prefix}_refit_lppd_comparison.pdf"
    scatter_quantile_path = (
        resolved_output_dir / f"{prefix}_refit_lppd_comparison_quantile_zoom.pdf"
    )
    scatter_full_path = (
        resolved_output_dir / f"{prefix}_refit_lppd_comparison_full_range.pdf"
    )
    absolute_lppd_path = (
        resolved_output_dir / f"{prefix}_lppd_norm_performance_by_model.pdf"
    )
    lppd_distribution_path = (
        resolved_output_dir / f"{prefix}_lppd_norm_distribution_boxplot.pdf"
    )
    delta_lppd_path = (
        resolved_output_dir / f"{prefix}_delta_lppd_norm_vs_original.pdf"
    )

    summary_df.to_csv(summary_path, index=False)
    delta_long.to_csv(delta_path, index=False)
    plot_delta_barplot(delta_long, barplot_path, robust=True)
    plot_delta_barplot(delta_long, barplot_full_path, robust=False)
    plot_lppd_norm_performance_by_model(absolute_lppd_df, absolute_lppd_path)
    plot_lppd_norm_distribution(absolute_lppd_df, lppd_distribution_path)
    plot_delta_lppd_norm_vs_original(delta_long, delta_lppd_path)
    plot_lppd_scatter(
        baseline_df,
        scatter_targets,
        scatter_path,
        robust=False,
        clip_limits=lppd_clip_limits,
    )
    plot_lppd_scatter(
        baseline_df,
        scatter_targets,
        scatter_quantile_path,
        robust=True,
        clip_limits=None,
    )
    plot_lppd_scatter(
        baseline_df,
        scatter_targets,
        scatter_full_path,
        robust=False,
        clip_limits=None,
    )

    print("Wrote outputs:")
    print(f"  summary: {summary_path}")
    print(f"  delta:   {delta_path}")
    print(f"  barplot: {barplot_path}")
    print(f"  barplot_full: {barplot_full_path}")
    print(f"  scatter: {scatter_path}")
    print(f"  scatter_quantile: {scatter_quantile_path}")
    print(f"  scatter_full: {scatter_full_path}")
    print(f"  lppd_by_model: {absolute_lppd_path}")
    print(f"  lppd_distribution: {lppd_distribution_path}")
    print(f"  lppd_delta_vs_original: {delta_lppd_path}")

    lppd_summary = summary_df[summary_df["metric"] == "lppd_test_norm"].copy()
    if not lppd_summary.empty:
        print("\nAverage improvement in normalized LPPD:")
        for _, row in lppd_summary.iterrows():
            print(f"  {row['model_display_name']}: {row['mean_delta']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Reimplement the refit_figures-style analysis for a wide graft comparison CSV."
        )
    )
    parser.add_argument(
        "--wide-csv",
        type=Path,
        default=DEFAULT_WIDE_CSV,
        help="Wide CSV produced by compare_graft_sat_env_performance.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated CSV/PDF outputs. Defaults to the input CSV directory.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Filename prefix for generated outputs. Defaults to the input CSV stem.",
    )
    parser.add_argument(
        "--lppd-clip-limits",
        type=float,
        nargs=2,
        default=list(DEFAULT_LPPD_CLIP_LIMITS),
        metavar=("LOWER", "UPPER"),
        help=(
            "Fixed axis limits for the main LPPD scatter plot. "
            "Use '--lppd-clip-limits -1 1' for normalized LPPD, or pass "
            "'--lppd-clip-limits' with different bounds."
        ),
    )
    args = parser.parse_args()
    main(
        wide_csv=args.wide_csv,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        lppd_clip_limits=parse_clip_limits(args.lppd_clip_limits),
    )
