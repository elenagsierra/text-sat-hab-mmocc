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
"""Plot refit-delta figures from compare_refit_delta_performance CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from compare_refit_delta_performance import DEFAULT_OUTPUT_PREFIX
from sat_mmocc.config import fig_page_width
from sat_mmocc.plot_utils import setup_matplotlib

COMPARISON_METRICS = [
    "lppd_test_norm",
    "biolith_ap_test",
    "biolith_roc_auc_test",
    "lr_map_test",
    "lr_mcc_test",
]

DOMAIN_DISPLAY_NAMES = {
    "satellite": "Satellite refits",
    "camera_trap": "Camera-trap refits",
}

DEFAULT_RESULTS_CSV = DEFAULT_OUTPUT_PREFIX.with_name(
    f"{DEFAULT_OUTPUT_PREFIX.name}_results.csv"
)
DEFAULT_DELTA_CSV = DEFAULT_OUTPUT_PREFIX.with_name(
    f"{DEFAULT_OUTPUT_PREFIX.name}_delta.csv"
)

ROBUST_LOWER_QUANTILE = 0.05
ROBUST_UPPER_QUANTILE = 0.95
DEFAULT_LPPD_CLIP_LIMITS = (-1.0, 1.0)


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def load_results(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "taxon_id",
        "scientific_name",
        "common_name",
        "domain",
        "experiment_label",
        "experiment_display_name",
        "is_final_species",
        "fit_result_exists",
        "lppd_test_norm",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    df["taxon_id"] = df["taxon_id"].astype(str)
    df["is_final_species"] = normalize_bool(df["is_final_species"])
    df["fit_result_exists"] = normalize_bool(df["fit_result_exists"])
    df = df[df["is_final_species"]].copy()
    if df.empty:
        raise RuntimeError(f"No final-species rows were found in {csv_path}.")
    return df


def load_delta(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "taxon_id",
        "scientific_name",
        "common_name",
        "domain",
        "reference_label",
        "reference_display_name",
        "target_label",
        "target_display_name",
        "is_final_species",
    }
    metric_cols: set[str] = set()
    for metric in COMPARISON_METRICS:
        metric_cols.update(
            {
                f"reference_{metric}",
                f"target_{metric}",
                f"delta_{metric}",
            }
        )
    missing = (required | metric_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    df["taxon_id"] = df["taxon_id"].astype(str)
    df["is_final_species"] = normalize_bool(df["is_final_species"])
    df = df[df["is_final_species"]].copy()
    if df.empty:
        raise RuntimeError(f"No final-species rows were found in {csv_path}.")
    return df


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
    return (lower - pad, upper + pad)


def parse_clip_limits(values: list[float] | None) -> tuple[float, float] | None:
    if values is None:
        return DEFAULT_LPPD_CLIP_LIMITS
    if len(values) != 2:
        raise ValueError("Clip limits must have exactly two values: lower upper.")
    lower, upper = float(values[0]), float(values[1])
    if lower >= upper:
        raise ValueError("Clip limits must satisfy lower < upper.")
    return (lower, upper)


def count_clipped(values: np.ndarray, limits: tuple[float, float]) -> int:
    lower, upper = limits
    finite = values[np.isfinite(values)]
    return int(((finite < lower) | (finite > upper)).sum())


def infer_output_prefix_name(path: Path) -> str:
    stem = path.stem
    for suffix in ("_results", "_delta", "_summary"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def add_domain_display_names(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["domain_display_name"] = result["domain"].map(DOMAIN_DISPLAY_NAMES).fillna(
        result["domain"]
    )
    return result


def build_delta_long(delta_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    base_cols = [
        "taxon_id",
        "scientific_name",
        "common_name",
        "domain",
        "reference_label",
        "reference_display_name",
        "target_label",
        "target_display_name",
    ]
    for metric in COMPARISON_METRICS:
        subset = delta_df.loc[
            :,
            base_cols
            + [
                f"reference_{metric}",
                f"target_{metric}",
                f"delta_{metric}",
            ],
        ].copy()
        subset.rename(
            columns={
                f"reference_{metric}": "baseline",
                f"target_{metric}": "target",
                f"delta_{metric}": "delta",
                "target_label": "model",
                "target_display_name": "model_display_name",
            },
            inplace=True,
        )
        subset["metric"] = metric
        records.append(subset)
    return add_domain_display_names(pd.concat(records, ignore_index=True))


def build_summary_table(delta_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "domain",
        "domain_display_name",
        "model",
        "model_display_name",
        "metric",
    ]
    for keys, group in delta_long.groupby(group_cols, dropna=False):
        domain, domain_display_name, model, model_display_name, metric = keys
        deltas = pd.to_numeric(group["delta"], errors="coerce")
        rows.append(
            {
                "domain": domain,
                "domain_display_name": domain_display_name,
                "model": model,
                "model_display_name": model_display_name,
                "metric": metric,
                "n_species": int(deltas.notna().sum()),
                "mean_delta": deltas.mean(),
                "median_delta": deltas.median(),
                "improved_%": 100.0 * (deltas > 0).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["domain", "model_display_name", "metric"], na_position="last"
    )


def build_absolute_metric_table(results_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    available = results_df[results_df["fit_result_exists"]].copy()
    if metric not in available.columns:
        return pd.DataFrame()
    metric_df = available[
        [
            "taxon_id",
            "scientific_name",
            "common_name",
            "domain",
            "experiment_label",
            "experiment_display_name",
            metric,
        ]
    ].copy()
    metric_df.rename(
        columns={
            "experiment_label": "model",
            "experiment_display_name": "model_display_name",
            metric: "value",
        },
        inplace=True,
    )
    metric_df["value"] = pd.to_numeric(metric_df["value"], errors="coerce")
    return add_domain_display_names(metric_df)


def ordered_domains(df: pd.DataFrame) -> list[str]:
    preferred = ["satellite", "camera_trap"]
    present = df["domain"].dropna().astype(str).unique().tolist()
    ordered = [domain for domain in preferred if domain in present]
    ordered.extend(domain for domain in present if domain not in ordered)
    return ordered


def plot_lppd_norm_performance_by_model(
    metric_df: pd.DataFrame,
    output_path: Path,
) -> None:
    domains = ordered_domains(metric_df)
    fig_width = max(fig_page_width, 14)
    fig, axes = plt.subplots(
        len(domains),
        1,
        figsize=(fig_width, max(3.8, len(domains) * fig_page_width / 1.5)),
    )
    if len(domains) == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        subset = metric_df[metric_df["domain"] == domain].copy()
        species_order = (
            subset.groupby("common_name", dropna=False)["value"]
            .mean()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        subset["common_name"] = pd.Categorical(
            subset["common_name"], categories=species_order, ordered=True
        )
        sns.barplot(
            data=subset,
            x="common_name",
            y="value",
            hue="model_display_name",
            ax=ax,
        )
        ax.set_title(f"{DOMAIN_DISPLAY_NAMES.get(domain, domain)}: LPPD Norm by model")
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
    domains = ordered_domains(metric_df)
    fig, axes = plt.subplots(
        len(domains),
        1,
        figsize=(fig_page_width / 1.1, max(3.0, len(domains) * fig_page_width / 1.8)),
    )
    if len(domains) == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        subset = metric_df[metric_df["domain"] == domain].copy()
        sns.boxplot(
            data=subset,
            x="model_display_name",
            y="value",
            ax=ax,
        )
        sns.stripplot(
            data=subset,
            x="model_display_name",
            y="value",
            color="black",
            alpha=0.55,
            size=4,
            ax=ax,
        )
        ax.set_title(f"{DOMAIN_DISPLAY_NAMES.get(domain, domain)}: LPPD Norm distribution")
        ax.set_ylabel(r"$\mathrm{LPPD}_{norm}^{test}$")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_delta_lppd_norm_vs_reference(
    delta_long: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_df = delta_long[delta_long["metric"] == "lppd_test_norm"].copy()
    if plot_df.empty:
        raise RuntimeError("No LPPD norm delta rows were available for plotting.")

    domains = ordered_domains(plot_df)
    fig_width = max(fig_page_width, 14)
    fig, axes = plt.subplots(
        len(domains),
        1,
        figsize=(fig_width, max(3.8, len(domains) * fig_page_width / 1.6)),
    )
    if len(domains) == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        subset = plot_df[plot_df["domain"] == domain].copy()
        species_order = (
            subset.groupby("common_name", dropna=False)["delta"]
            .mean()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        subset["common_name"] = pd.Categorical(
            subset["common_name"], categories=species_order, ordered=True
        )
        sns.barplot(
            data=subset,
            x="common_name",
            y="delta",
            hue="model_display_name",
            ax=ax,
        )
        ax.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax.set_title(
            f"{DOMAIN_DISPLAY_NAMES.get(domain, domain)}: Delta LPPD Norm vs baseline"
        )
        ax.set_ylabel(r"$\Delta \mathrm{LPPD}_{norm}^{test}$")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_delta_barplot(
    delta_long: pd.DataFrame,
    output_path: Path,
    robust: bool,
) -> None:
    domains = ordered_domains(delta_long)
    fig, axes = plt.subplots(
        len(domains),
        1,
        figsize=(fig_page_width, max(3.2, len(domains) * fig_page_width / 2.1)),
    )
    if len(domains) == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        subset = delta_long[delta_long["domain"] == domain].copy()
        sns.barplot(
            data=subset,
            x="metric",
            y="delta",
            hue="model_display_name",
            estimator=np.mean,
            errorbar="sd",
            ax=ax,
            order=COMPARISON_METRICS,
        )
        ax.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax.set_title(f"{DOMAIN_DISPLAY_NAMES.get(domain, domain)}: Delta vs baseline")
        ax.set_ylabel("Delta vs domain baseline")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="")
        limits = compute_axis_limits(subset["delta"].to_numpy(dtype=float), robust=robust)
        ax.set_ylim(limits)
        if robust:
            clipped = count_clipped(subset["delta"].to_numpy(dtype=float), limits)
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
    delta_df: pd.DataFrame,
    output_path: Path,
    robust: bool,
    clip_limits: tuple[float, float] | None = None,
) -> None:
    domains = ordered_domains(delta_df)
    targets_by_domain = {
        domain: delta_df.loc[
            delta_df["domain"] == domain,
            ["target_label", "target_display_name"],
        ]
        .dropna(subset=["target_label"])
        .drop_duplicates()
        .sort_values("target_display_name", na_position="last")["target_label"]
        .astype(str)
        .tolist()
        for domain in domains
    }
    max_targets = max(len(targets) for targets in targets_by_domain.values())
    fig, axes = plt.subplots(
        len(domains),
        max_targets,
        figsize=(fig_page_width, max(3.4, len(domains) * fig_page_width / 2.0)),
        squeeze=False,
    )

    for row_idx, domain in enumerate(domains):
        target_labels = targets_by_domain[domain]
        for col_idx in range(max_targets):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(target_labels):
                ax.set_visible(False)
                continue

            target_label = target_labels[col_idx]
            subset = delta_df[
                (delta_df["domain"] == domain) & (delta_df["target_label"] == target_label)
            ].copy()
            scatter_data = subset[
                [
                    "scientific_name",
                    "reference_display_name",
                    "target_display_name",
                    "reference_lppd_test_norm",
                    "target_lppd_test_norm",
                ]
            ].rename(
                columns={
                    "reference_lppd_test_norm": "baseline",
                    "target_lppd_test_norm": "target",
                }
            )
            scatter_data = scatter_data.dropna(subset=["baseline", "target"])
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

            reference_name = scatter_data["reference_display_name"].iloc[0]
            target_name = scatter_data["target_display_name"].iloc[0]
            ax.set_title(f"{DOMAIN_DISPLAY_NAMES.get(domain, domain)}: {target_name}")
            ax.set_xlabel(rf"{reference_name} $\mathrm{{LPPD}}_{{norm}}^{{test}}$")
            ax.set_ylabel(rf"{target_name} $\mathrm{{LPPD}}_{{norm}}^{{test}}$")

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
    results_csv: str | Path = DEFAULT_RESULTS_CSV,
    delta_csv: str | Path = DEFAULT_DELTA_CSV,
    output_dir: str | Path | None = None,
    output_prefix: str | None = None,
    lppd_clip_limits: tuple[float, float] | None = DEFAULT_LPPD_CLIP_LIMITS,
) -> None:
    setup_matplotlib()
    results_csv = Path(results_csv)
    delta_csv = Path(delta_csv)
    results_df = add_domain_display_names(load_results(results_csv))
    delta_df = add_domain_display_names(load_delta(delta_csv))
    delta_long = build_delta_long(delta_df)
    if delta_long.empty:
        raise RuntimeError(f"No overlapping refit delta rows were found in {delta_csv}.")
    summary_df = build_summary_table(delta_long)
    absolute_lppd_df = build_absolute_metric_table(results_df, "lppd_test_norm")
    if absolute_lppd_df.empty:
        raise RuntimeError(f"No LPPD norm rows were available in {results_csv}.")

    resolved_output_dir = (
        Path(output_dir) if output_dir is not None else results_csv.parent
    )
    prefix = output_prefix or infer_output_prefix_name(results_csv)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = resolved_output_dir / f"{prefix}_refit_summary.csv"
    delta_long_path = resolved_output_dir / f"{prefix}_refit_delta_long.csv"
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
        resolved_output_dir / f"{prefix}_delta_lppd_norm_vs_reference.pdf"
    )

    summary_df.to_csv(summary_path, index=False)
    delta_long.to_csv(delta_long_path, index=False)
    plot_delta_barplot(delta_long, barplot_path, robust=True)
    plot_delta_barplot(delta_long, barplot_full_path, robust=False)
    plot_lppd_norm_performance_by_model(absolute_lppd_df, absolute_lppd_path)
    plot_lppd_norm_distribution(absolute_lppd_df, lppd_distribution_path)
    plot_delta_lppd_norm_vs_reference(delta_long, delta_lppd_path)
    plot_lppd_scatter(
        delta_df,
        scatter_path,
        robust=False,
        clip_limits=lppd_clip_limits,
    )
    plot_lppd_scatter(
        delta_df,
        scatter_quantile_path,
        robust=True,
        clip_limits=None,
    )
    plot_lppd_scatter(
        delta_df,
        scatter_full_path,
        robust=False,
        clip_limits=None,
    )

    print("Wrote outputs:")
    print(f"  summary: {summary_path}")
    print(f"  delta:   {delta_long_path}")
    print(f"  barplot: {barplot_path}")
    print(f"  barplot_full: {barplot_full_path}")
    print(f"  scatter: {scatter_path}")
    print(f"  scatter_quantile: {scatter_quantile_path}")
    print(f"  scatter_full: {scatter_full_path}")
    print(f"  lppd_by_model: {absolute_lppd_path}")
    print(f"  lppd_distribution: {lppd_distribution_path}")
    print(f"  lppd_delta_vs_reference: {delta_lppd_path}")

    lppd_summary = summary_df[summary_df["metric"] == "lppd_test_norm"].copy()
    if not lppd_summary.empty:
        print("\nAverage improvement in normalized LPPD:")
        for _, row in lppd_summary.iterrows():
            print(
                f"  {row['domain_display_name']} / "
                f"{row['model_display_name']}: {row['mean_delta']:.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Reimplement the refit-figure style analysis for compare_refit_delta_performance CSVs."
        )
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Results CSV produced by compare_refit_delta_performance.py.",
    )
    parser.add_argument(
        "--delta-csv",
        type=Path,
        default=DEFAULT_DELTA_CSV,
        help="Delta CSV produced by compare_refit_delta_performance.py.",
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
        help="Filename prefix for generated outputs. Defaults to the input results CSV stem.",
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
        results_csv=args.results_csv,
        delta_csv=args.delta_csv,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        lppd_clip_limits=parse_clip_limits(args.lppd_clip_limits),
    )
