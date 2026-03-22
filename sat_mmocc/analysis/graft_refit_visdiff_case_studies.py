#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pillow",   
#     "transformers",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Render per-species Sentinel/NAIP GRAFT refit panels for final species.

For each `is_final_species` row in the wide CSV produced by
`compare_graft_sat_env_performance.py`, this script writes a figure showing:
  - likely occupied satellite images
  - likely unoccupied satellite images
  - top 5 VisDiff phrases
  - LPPDnorm for baseline, Sentinel refit, and NAIP refit
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

from sat_mmocc.config import cache_path, default_image_backbone
from sat_mmocc.graft_utils import get_graft_text_encoder
from sat_mmocc.interpretability_utils import (
    load_fit_results,
    load_location_ids,
    rank_image_groups,
    resolve_fit_results_path,
)
from sat_mmocc.utils import load_data

LOGGER = logging.getLogger(__name__)

BASELINE_LABEL = "original_sat_env"
COMPARISON_METRIC = "lppd_test_norm"
DEFAULT_MODALITIES = ("covariates", "sat")
DEFAULT_WIDE_CSV = (
    Path(__file__).resolve().parent / "outputs" / "my_graft_compare_wide.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "outputs" / "graft_refit_final_species_panels"
)

THUMB_SIZE = (192, 192)

MODEL_SPECS = {
    "sentinel": {
        "label": "refit_sentinel",
        "display_name": "Sentinel GRAFT",
        "fit_sat_backbone": "graft_visdiff_sentinel",
        "sat_backbone_data": "graft",
        "visdiff_csv": cache_path / "visdiff_sat_wi_prompt2.csv",
        "png_dir": cache_path / "sat_wi_rgb_images_png",
    },
    "naip": {
        "label": "refit_naip",
        "display_name": "NAIP GRAFT",
        "fit_sat_backbone": "graft_visdiff_naip",
        "sat_backbone_data": "graft_naip",
        "visdiff_csv": cache_path / "visdiff_naip_wi_prompt2.csv",
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


def parse_species_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return set(items) if items else None


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

    for col in (
        "baseline_lppd_norm",
        "sentinel_lppd_norm",
        "naip_lppd_norm",
    ):
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
        phrase = str(row.difference)
        lines.append(f"{idx}. {phrase} ({float(row.auroc):.3f})")
    return lines


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def compute_similarity_features(
    embeddings: np.ndarray,
    descriptor_vectors: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    sat_unit = normalize_rows(embeddings.astype(np.float32, copy=False))
    desc_unit = normalize_rows(descriptor_vectors.astype(np.float32, copy=False))
    sims = sat_unit @ desc_unit.T
    if weights.size:
        scaled_weights = weights.astype(np.float32, copy=True)
        max_abs = np.max(np.abs(scaled_weights))
        if max_abs > 0:
            scaled_weights /= max_abs
        else:
            scaled_weights[:] = 1.0
        sims *= scaled_weights.reshape(1, -1)
    return sims.astype(np.float32)


def build_ranked_groups(
    taxon_id: str,
    source: str,
    visdiff_df: pd.DataFrame,
    text_encoder: Any,
    modalities: tuple[str, ...],
    image_backbone: str,
    top_k: int,
    unique_weight: float,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = MODEL_SPECS[source]
    fit_path, resolved_modalities, resolved_image_backbone, _ = resolve_fit_results_path(
        str(taxon_id),
        modalities,
        image_backbone,
        str(spec["fit_sat_backbone"]),
    )
    fit_results = load_fit_results(fit_path)
    resolved_image_backbone = normalize_image_backbone(resolved_image_backbone)

    (
        _,
        _,
        _,
        _,
        _,
        _,
        _scientific_name,
        _common_name,
        mask_train,
        mask_test,
        _too_close,
        _y_train,
        _y_test,
        _y_train_naive,
        _y_test_naive,
        features_modalities,
    ) = load_data(
        str(taxon_id),
        set(resolved_modalities),
        image_backbone_name=resolved_image_backbone,
        sat_backbone_name=str(spec["sat_backbone_data"]),
    )

    descriptor_rows = get_descriptor_rows(visdiff_df, taxon_id)
    if descriptor_rows.empty:
        raise RuntimeError(f"No VisDiff descriptors found for {source} / {taxon_id}")

    descriptor_texts = descriptor_rows["difference"].tolist()
    descriptor_scores = descriptor_rows["auroc"].astype(float).to_numpy(dtype=np.float32)
    descriptor_embeddings = text_encoder.encode(descriptor_texts)
    similarity_features = compute_similarity_features(
        features_modalities["sat"], descriptor_embeddings, descriptor_scores
    )

    scaler = fit_results["modalities_scaler"]["sat"]
    pca = fit_results["modalities_pca"]["sat"]
    coefficients = fit_results["modality_coefficients"]["sat"]
    sat_scores = pca.transform(scaler.transform(similarity_features)) @ coefficients

    ids_all = load_location_ids(resolved_image_backbone)
    mask_train = np.asarray(mask_train, dtype=bool)
    mask_test = np.asarray(mask_test, dtype=bool)
    if len(ids_all) != len(mask_train):
        raise ValueError("Location ids and masks are misaligned.")

    site_scores = pd.DataFrame(
        {
            "loc_id": ids_all,
            "is_train": mask_train,
            "is_test": mask_test,
            "score_sat": sat_scores,
        }
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
    return positives, negatives


def image_paths(df: pd.DataFrame, n_images: int) -> list[str]:
    if df.empty or "image_path" not in df.columns:
        return []
    return df["image_path"].dropna().astype(str).head(n_images).tolist()


def load_thumb(path: str) -> np.ndarray | None:
    try:
        return np.asarray(
            Image.open(path).convert("RGB").resize(THUMB_SIZE, Image.LANCZOS)
        )
    except Exception:
        return None


def plot_image_row(fig: plt.Figure, subplot_spec, title: str, paths: list[str], n_images: int) -> None:
    row_gs = gridspec.GridSpecFromSubplotSpec(1, n_images, subplot_spec=subplot_spec, wspace=0.04)
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
        if idx == 0:
            ax.set_ylabel(title, fontsize=9, fontweight="bold", rotation=90, labelpad=6)


def draw_phrase_panel(
    ax: plt.Axes,
    title: str,
    model_value: float,
    delta_value: float,
    lines: list[str],
) -> None:
    ax.axis("off")
    header = f"{title}\nLPPDnorm: {format_value(model_value)} (delta {format_value(delta_value)})"
    wrapped_lines = []
    for line in lines:
        wrapped_lines.append(textwrap.fill(line, width=42, subsequent_indent="   "))
    body = "\n\n".join(wrapped_lines) if wrapped_lines else "No VisDiff phrases found."
    ax.text(
        0.0,
        1.0,
        f"{header}\n\nTop 5 VisDiff phrases\n\n{body}",
        ha="left",
        va="top",
        fontsize=10,
        transform=ax.transAxes,
    )


def format_value(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):+.3f}"


def sanitize_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


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
    fig = plt.figure(figsize=(22, 10))
    outer = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[15, 6.5, 6.5],
        wspace=0.3,
    )
    image_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[0], hspace=0.18)

    plot_image_row(fig, image_gs[0], "Sentinel likely occupied", sentinel_occ, n_images)
    plot_image_row(fig, image_gs[1], "Sentinel likely unoccupied", sentinel_unocc, n_images)
    plot_image_row(fig, image_gs[2], "NAIP likely occupied", naip_occ, n_images)
    plot_image_row(fig, image_gs[3], "NAIP likely unoccupied", naip_unocc, n_images)

    ax_sentinel = fig.add_subplot(outer[1])
    draw_phrase_panel(
        ax_sentinel,
        "Sentinel GRAFT",
        row["sentinel_lppd_norm"],
        row["sentinel_delta_lppd_norm"],
        sentinel_lines,
    )

    ax_naip = fig.add_subplot(outer[2])
    draw_phrase_panel(
        ax_naip,
        "NAIP GRAFT",
        row["naip_lppd_norm"],
        row["naip_delta_lppd_norm"],
        naip_lines,
    )

    fig.suptitle(
        (
            f"{row['display_name']}  (taxon {row['taxon_id']})\n"
            f"Baseline LPPDnorm: {format_value(row['baseline_lppd_norm'])} | "
            f"Sentinel: {format_value(row['sentinel_lppd_norm'])} "
            f"(delta {format_value(row['sentinel_delta_lppd_norm'])}) | "
            f"NAIP: {format_value(row['naip_lppd_norm'])} "
            f"(delta {format_value(row['naip_delta_lppd_norm'])})"
        ),
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
    overwrite: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    selected_species = parse_species_ids(species_ids)
    image_backbone = normalize_image_backbone(image_backbone)
    modalities_tuple = tuple(item.strip() for item in modalities.split(",") if item.strip())
    if "sat" not in modalities_tuple:
        raise ValueError("Modalities must include 'sat'.")

    wide_df = load_wide_results(wide_csv)
    summary_df = build_species_summary(wide_df)
    if selected_species is not None:
        summary_df = summary_df[summary_df["taxon_id"].isin(selected_species)].copy()
    if summary_df.empty:
        raise RuntimeError("No final species matched the requested filters.")

    visdiff_tables = {
        source: load_visdiff_table(spec["visdiff_csv"]) for source, spec in MODEL_SPECS.items()
    }
    text_encoder = get_graft_text_encoder()

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "final_species_lppdnorm_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    rendered = 0
    for row in summary_df.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        output_path = figures_dir / f"{sanitize_filename(row_series['display_name'])}__{row_series['taxon_id']}.pdf"
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
            positives, negatives = build_ranked_groups(
                row_series["taxon_id"],
                "sentinel",
                visdiff_tables["sentinel"],
                text_encoder,
                modalities_tuple,
                image_backbone,
                top_k,
                unique_weight,
                mode,
            )
            sentinel_occ = image_paths(positives, n_images)
            sentinel_unocc = image_paths(negatives, n_images)
        except Exception as exc:
            LOGGER.warning("Sentinel ranking failed for %s: %s", row_series["taxon_id"], exc)

        try:
            positives, negatives = build_ranked_groups(
                row_series["taxon_id"],
                "naip",
                visdiff_tables["naip"],
                text_encoder,
                modalities_tuple,
                image_backbone,
                top_k,
                unique_weight,
                mode,
            )
            naip_occ = image_paths(positives, n_images)
            naip_unocc = image_paths(negatives, n_images)
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
            "Render one final-species panel per species with occupied/unoccupied images, "
            "top VisDiff phrases, and LPPDnorm deltas for Sentinel and NAIP GRAFT refits."
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
        help="Number of likely occupied/unoccupied images to show per source.",
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
        overwrite=args.overwrite,
    )
