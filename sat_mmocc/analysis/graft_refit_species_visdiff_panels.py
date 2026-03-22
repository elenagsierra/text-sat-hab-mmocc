#!/usr/bin/env -S uv run --python 3.11 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pillow",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Render one per-species panel with LPPD norm, VisDiff text, and ranked imagery.

This script reuses the wide comparison CSV used by
`graft_refit_figures_from_csv.py`, then combines:

1. baseline / Sentinel / NAIP `lppd_test_norm` values
2. top VisDiff phrases from the step-08 descriptor CSVs
3. likely occupied (group A) and unlikely occupied (group B) aerial imagery

One figure is written per species.
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from sat_mmocc.config import cache_path, default_image_backbone, fig_page_width
from sat_mmocc.interpretability_utils import (
    compute_site_scores,
    load_fit_results,
    rank_image_groups,
    resolve_fit_results_path,
)
from sat_mmocc.plot_utils import setup_matplotlib

LOGGER = logging.getLogger(__name__)

BASELINE_LABEL = "original_sat_env"
COMPARISON_METRIC = "lppd_test_norm"
DEFAULT_MODALITIES = ("covariates", "sat")
DEFAULT_WIDE_CSV = (
    Path(__file__).resolve().parent / "outputs" / "my_graft_compare_wide.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "outputs" / "graft_refit_species_visdiff_panels"
)
THUMB_SIZE = (192, 192)
MODEL_BAR_COLORS = {
    "Sat+env baseline": "#6c757d",
    "Sentinel GRAFT": "#2a9d8f",
    "NAIP GRAFT": "#e76f51",
}

MODEL_SPECS = {
    "sentinel": {
        "label": "refit_sentinel",
        "display_name": "Sentinel GRAFT",
        "fit_sat_backbone": "graft_visdiff_sentinel",
        "visdiff_csv": cache_path / "visdiff_sat_sentinel2_wi_prompt2.csv",
        "png_dir": cache_path / "sat_wi_rgb_images_png",
    },
    "naip": {
        "label": "refit_naip",
        "display_name": "NAIP GRAFT",
        "fit_sat_backbone": "graft_visdiff_naip",
        "visdiff_csv": cache_path / "visdiff_sat_naip_wi_prompt2.csv",
        "png_dir": cache_path / "naip_wi_images_png",
    },
}


def metric_column(model_label: str, metric: str) -> str:
    return f"{model_label}__{metric}"


def available_column(model_label: str) -> str:
    return f"{model_label}__fit_result_exists"


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def normalize_image_backbone(value: str | None) -> str:
    if value is None:
        return default_image_backbone
    if value.strip().lower() in {"", "none"}:
        return default_image_backbone
    return value


def parse_species_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return set(items) if items else None


def sanitize_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def format_metric(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):+.3f}"


def load_wide_results(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"taxon_id", "scientific_name", "common_name", "is_final_species"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    required_cols = [
        available_column(BASELINE_LABEL),
        metric_column(BASELINE_LABEL, COMPARISON_METRIC),
    ]
    for spec in MODEL_SPECS.values():
        required_cols.extend(
            [
                available_column(spec["label"]),
                metric_column(spec["label"], COMPARISON_METRIC),
            ]
        )
    missing_metric_cols = [col for col in required_cols if col not in df.columns]
    if missing_metric_cols:
        raise ValueError(
            f"Input CSV is missing required comparison columns: {sorted(missing_metric_cols)}"
        )

    df["taxon_id"] = df["taxon_id"].astype(str)
    df["is_final_species"] = normalize_bool(df["is_final_species"])
    df = df[df["is_final_species"]].copy()
    if df.empty:
        raise RuntimeError(f"No final-species rows were found in {csv_path}.")
    return df


def build_species_summary(wide_df: pd.DataFrame) -> pd.DataFrame:
    summary = wide_df.loc[
        :,
        [
            "taxon_id",
            "scientific_name",
            "common_name",
            metric_column(BASELINE_LABEL, COMPARISON_METRIC),
            available_column(BASELINE_LABEL),
            metric_column(MODEL_SPECS["sentinel"]["label"], COMPARISON_METRIC),
            available_column(MODEL_SPECS["sentinel"]["label"]),
            metric_column(MODEL_SPECS["naip"]["label"], COMPARISON_METRIC),
            available_column(MODEL_SPECS["naip"]["label"]),
        ],
    ].copy()

    summary.rename(
        columns={
            metric_column(BASELINE_LABEL, COMPARISON_METRIC): "baseline_lppd_norm",
            available_column(BASELINE_LABEL): "baseline_available",
            metric_column(MODEL_SPECS["sentinel"]["label"], COMPARISON_METRIC): "sentinel_lppd_norm",
            available_column(MODEL_SPECS["sentinel"]["label"]): "sentinel_available",
            metric_column(MODEL_SPECS["naip"]["label"], COMPARISON_METRIC): "naip_lppd_norm",
            available_column(MODEL_SPECS["naip"]["label"]): "naip_available",
        },
        inplace=True,
    )

    for col in ("baseline_lppd_norm", "sentinel_lppd_norm", "naip_lppd_norm"):
        summary[col] = pd.to_numeric(summary[col], errors="coerce")

    for col in ("baseline_available", "sentinel_available", "naip_available"):
        summary[col] = normalize_bool(summary[col])

    summary["sentinel_delta_lppd_norm"] = (
        summary["sentinel_lppd_norm"] - summary["baseline_lppd_norm"]
    )
    summary["naip_delta_lppd_norm"] = (
        summary["naip_lppd_norm"] - summary["baseline_lppd_norm"]
    )
    summary["display_name"] = summary["common_name"].fillna(summary["scientific_name"])
    summary["display_name"] = summary["display_name"].fillna(summary["taxon_id"])
    return summary.sort_values(["display_name", "taxon_id"], na_position="last").reset_index(
        drop=True
    )


def load_visdiff_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"VisDiff CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"taxon_id", "difference", "auroc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    df["taxon_id"] = df["taxon_id"].astype(str)
    df["auroc"] = pd.to_numeric(df["auroc"], errors="coerce")
    return df


def get_descriptor_rows(visdiff_df: pd.DataFrame, taxon_id: str) -> pd.DataFrame:
    subset = visdiff_df[visdiff_df["taxon_id"] == str(taxon_id)].copy()
    subset = subset.dropna(subset=["difference"])
    subset["difference"] = subset["difference"].astype(str).str.strip()
    subset = subset[subset["difference"] != ""]
    subset = subset.dropna(subset=["auroc"])
    subset = subset.sort_values("auroc", ascending=False)
    subset = subset.drop_duplicates(subset="difference")
    return subset.reset_index(drop=True)


def top_phrase_lines(visdiff_df: pd.DataFrame, taxon_id: str, top_n: int) -> list[str]:
    subset = get_descriptor_rows(visdiff_df, taxon_id).head(top_n)
    lines = []
    for idx, row in enumerate(subset.itertuples(index=False), start=1):
        lines.append(f"{idx}. {row.difference} ({float(row.auroc):.3f})")
    return lines


def build_ranked_image_paths(
    taxon_id: str,
    source: str,
    modalities: tuple[str, ...],
    image_backbone: str,
    top_k: int,
    unique_weight: float,
    mode: str,
    n_images: int,
) -> tuple[list[str], list[str]]:
    spec = MODEL_SPECS[source]
    fit_path, resolved_modalities, resolved_image_backbone, resolved_sat_backbone = (
        resolve_fit_results_path(
            str(taxon_id),
            modalities,
            image_backbone,
            str(spec["fit_sat_backbone"]),
        )
    )
    fit_results = load_fit_results(fit_path)
    site_scores, _ = compute_site_scores(
        str(taxon_id),
        resolved_modalities,
        resolved_image_backbone,
        resolved_sat_backbone,
        fit_results,
    )
    png_dir = Path(str(spec["png_dir"]))
    site_scores["image_path"] = site_scores["loc_id"].apply(
        lambda loc_id: str(png_dir / f"{loc_id}.png")
    )
    site_scores["image_exists"] = site_scores["image_path"].apply(lambda path: Path(path).exists())
    positives, negatives = rank_image_groups(
        site_scores,
        resolved_modalities,
        mode=mode,
        unique_weight=unique_weight,
        top_k=top_k,
        image_modality="sat",
        test=False,
    )
    pos_paths = positives["image_path"].dropna().astype(str).head(n_images).tolist()
    neg_paths = negatives["image_path"].dropna().astype(str).head(n_images).tolist()
    return pos_paths, neg_paths


def load_thumb(path: str) -> np.ndarray | None:
    try:
        return np.asarray(
            Image.open(path).convert("RGB").resize(THUMB_SIZE, Image.LANCZOS)
        )
    except Exception:
        return None


def plot_image_row(
    fig: plt.Figure,
    subplot_spec: gridspec.SubplotSpec,
    title: str,
    paths: list[str],
    n_images: int,
    edge_color: str,
) -> None:
    row_gs = gridspec.GridSpecFromSubplotSpec(
        1,
        n_images,
        subplot_spec=subplot_spec,
        wspace=0.04,
    )
    axes = [fig.add_subplot(row_gs[0, idx]) for idx in range(n_images)]
    for idx, ax in enumerate(axes):
        if idx < len(paths):
            thumb = load_thumb(paths[idx])
            if thumb is not None:
                ax.imshow(thumb)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
        else:
            ax.set_visible(False)
            continue
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(edge_color)
            spine.set_linewidth(2)
        if idx == 0:
            ax.set_ylabel(
                title,
                fontsize=9,
                fontweight="bold",
                rotation=90,
                labelpad=6,
            )


def draw_text_panel(ax: plt.Axes, title: str, lines: list[str]) -> None:
    ax.axis("off")
    wrapped_lines = [
        textwrap.fill(line, width=40, subsequent_indent="   ") for line in lines
    ]
    body = "\n\n".join(wrapped_lines) if wrapped_lines else "No VisDiff phrases found."
    ax.text(
        0.0,
        1.0,
        f"{title}\n\n{body}",
        ha="left",
        va="top",
        fontsize=10,
        transform=ax.transAxes,
    )


def draw_lppd_panel(ax: plt.Axes, row: pd.Series) -> None:
    labels = ["Sat+env baseline", "Sentinel GRAFT", "NAIP GRAFT"]
    values = np.array(
        [
            row["baseline_lppd_norm"],
            row["sentinel_lppd_norm"],
            row["naip_lppd_norm"],
        ],
        dtype=float,
    )
    deltas = [np.nan, row["sentinel_delta_lppd_norm"], row["naip_delta_lppd_norm"]]
    y_positions = np.arange(len(labels))
    colors = [MODEL_BAR_COLORS[label] for label in labels]

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        finite = np.array([0.0], dtype=float)
    span = float(np.nanmax(finite) - np.nanmin(finite))
    pad = max(0.08 * span, 0.1)
    x_min = float(np.nanmin(finite) - pad)
    x_max = float(np.nanmax(finite) + pad)
    if np.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5

    ax.barh(y_positions, values, color=colors, alpha=0.92)
    ax.axvline(0.0, color="gray", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\mathrm{LPPD}_{norm}^{test}$")
    ax.set_title("Per-species LPPD norm")
    ax.set_xlim(x_min, x_max)
    ax.invert_yaxis()

    for y_pos, value, delta, label in zip(y_positions, values, deltas, labels, strict=True):
        if not np.isfinite(value):
            ax.text(0.02, y_pos, "NA", va="center", ha="left", transform=ax.get_yaxis_transform())
            continue
        text = format_metric(value)
        if label != "Sat+env baseline" and np.isfinite(delta):
            text = f"{text}  (Δ {format_metric(delta)})"
        offset = 0.015 * (x_max - x_min)
        x_text = value + offset if value >= 0 else value - offset
        ha = "left" if value >= 0 else "right"
        ax.text(x_text, y_pos, text, va="center", ha=ha, fontsize=9)


def render_species_panel(
    row: pd.Series,
    sentinel_occ: list[str],
    sentinel_unocc: list[str],
    naip_occ: list[str],
    naip_unocc: list[str],
    sentinel_lines: list[str],
    naip_lines: list[str],
    output_path: Path,
    n_images: int,
) -> None:
    fig_width = max(fig_page_width * 1.9, 16)
    fig_height = max(fig_page_width * 1.05, 9)
    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[16, 7, 7],
        height_ratios=[4.8, 1.9],
        wspace=0.3,
        hspace=0.28,
    )

    image_gs = gridspec.GridSpecFromSubplotSpec(
        4,
        1,
        subplot_spec=outer[:, 0],
        hspace=0.16,
    )
    plot_image_row(
        fig,
        image_gs[0],
        "Sentinel group A\nlikely occupied",
        sentinel_occ,
        n_images,
        edge_color="#2a9d8f",
    )
    plot_image_row(
        fig,
        image_gs[1],
        "Sentinel group B\nunlikely occupied",
        sentinel_unocc,
        n_images,
        edge_color="#52796f",
    )
    plot_image_row(
        fig,
        image_gs[2],
        "NAIP group A\nlikely occupied",
        naip_occ,
        n_images,
        edge_color="#e76f51",
    )
    plot_image_row(
        fig,
        image_gs[3],
        "NAIP group B\nunlikely occupied",
        naip_unocc,
        n_images,
        edge_color="#bc6c25",
    )

    ax_sentinel = fig.add_subplot(outer[0, 1])
    draw_text_panel(ax_sentinel, "Sentinel VisDiff text", sentinel_lines)

    ax_naip = fig.add_subplot(outer[0, 2])
    draw_text_panel(ax_naip, "NAIP VisDiff text", naip_lines)

    ax_lppd = fig.add_subplot(outer[1, 1:])
    draw_lppd_panel(ax_lppd, row)

    fig.suptitle(
        f"{row['display_name']}  (taxon {row['taxon_id']})",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main(
    wide_csv: str | Path = DEFAULT_WIDE_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    species_ids: str | None = None,
    modalities: str = ",".join(DEFAULT_MODALITIES),
    image_backbone: str | None = default_image_backbone,
    n_images: int = 5,
    top_phrases: int = 5,
    top_k: int = 50,
    unique_weight: float = 2.0,
    mode: str = "standard",
    fmt: str = "pdf",
    overwrite: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    setup_matplotlib()

    selected_species = parse_species_ids(species_ids)
    image_backbone = normalize_image_backbone(image_backbone)
    modalities_tuple = tuple(item.strip() for item in modalities.split(",") if item.strip())
    if "sat" not in modalities_tuple:
        raise ValueError("Modalities must include 'sat'.")
    if fmt not in {"pdf", "png"}:
        raise ValueError("fmt must be one of: pdf, png")

    wide_df = load_wide_results(wide_csv)
    summary_df = build_species_summary(wide_df)
    if selected_species is not None:
        summary_df = summary_df[summary_df["taxon_id"].isin(selected_species)].copy()
    if summary_df.empty:
        raise RuntimeError("No final species matched the requested filters.")

    visdiff_tables = {
        source: load_visdiff_table(spec["visdiff_csv"]) for source, spec in MODEL_SPECS.items()
    }

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "final_species_lppdnorm_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    rendered = 0
    for row in summary_df.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        output_path = (
            figures_dir
            / f"{sanitize_filename(str(row_series['display_name']))}__{row_series['taxon_id']}.{fmt}"
        )
        if output_path.exists() and not overwrite:
            rendered += 1
            continue

        sentinel_lines = top_phrase_lines(
            visdiff_tables["sentinel"], row_series["taxon_id"], top_phrases
        )
        naip_lines = top_phrase_lines(
            visdiff_tables["naip"], row_series["taxon_id"], top_phrases
        )

        sentinel_occ: list[str] = []
        sentinel_unocc: list[str] = []
        naip_occ: list[str] = []
        naip_unocc: list[str] = []

        try:
            sentinel_occ, sentinel_unocc = build_ranked_image_paths(
                row_series["taxon_id"],
                "sentinel",
                modalities_tuple,
                image_backbone,
                top_k,
                unique_weight,
                mode,
                n_images,
            )
        except Exception as exc:
            LOGGER.warning("Sentinel ranking failed for %s: %s", row_series["taxon_id"], exc)

        try:
            naip_occ, naip_unocc = build_ranked_image_paths(
                row_series["taxon_id"],
                "naip",
                modalities_tuple,
                image_backbone,
                top_k,
                unique_weight,
                mode,
                n_images,
            )
        except Exception as exc:
            LOGGER.warning("NAIP ranking failed for %s: %s", row_series["taxon_id"], exc)

        render_species_panel(
            row_series,
            sentinel_occ,
            sentinel_unocc,
            naip_occ,
            naip_unocc,
            sentinel_lines,
            naip_lines,
            output_path,
            n_images,
        )
        rendered += 1
        LOGGER.info("Saved %s", output_path)

    print("Wrote outputs:")
    print(f"  summary: {summary_path}")
    print(f"  figure_dir: {figures_dir}")
    print(f"  figures_saved: {rendered}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Render one figure per final species with LPPD norm, VisDiff text, and "
            "likely occupied / unlikely occupied aerial imagery for Sentinel and NAIP."
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
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the summary CSV and per-species figures.",
    )
    parser.add_argument(
        "--species-ids",
        type=str,
        default=None,
        help="Optional comma-separated taxon IDs to render.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="covariates,sat",
        help="Modalities matching the comparison CSV experiments.",
    )
    parser.add_argument(
        "--image-backbone",
        type=str,
        default=default_image_backbone,
        help="Image backbone used by the fit results and cached ids.",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=5,
        help="Number of likely occupied / unlikely occupied images to show per source.",
    )
    parser.add_argument(
        "--top-phrases",
        type=int,
        default=5,
        help="Number of VisDiff phrases to display per source.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k ranked sites to consider before selecting displayed images.",
    )
    parser.add_argument(
        "--unique-weight",
        type=float,
        default=2.0,
        help="Uniqueness weight used when mode=unique.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        help="Ranking mode passed to rank_image_groups: standard or unique.",
    )
    parser.add_argument(
        "--fmt",
        type=str,
        default="pdf",
        help="Figure format: pdf or png.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-species figures.",
    )
    args = parser.parse_args()
    main(
        wide_csv=args.wide_csv,
        output_dir=args.output_dir,
        species_ids=args.species_ids,
        modalities=args.modalities,
        image_backbone=args.image_backbone,
        n_images=args.n_images,
        top_phrases=args.top_phrases,
        top_k=args.top_k,
        unique_weight=args.unique_weight,
        mode=args.mode,
        fmt=args.fmt,
        overwrite=args.overwrite,
    )
