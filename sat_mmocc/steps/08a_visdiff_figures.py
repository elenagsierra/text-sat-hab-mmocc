#!/usr/bin/env -S uv run --python 3.11 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "setuptools<81"
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Generate per-species VisDiff figures (PNG / PDF) and save them to a directory.

Usage examples
--------------
# All species in the default CSV → PNG files in CACHE_PATH/visdiff_figures/
./sat_mmocc/steps/09_visdiff_figures.py

# Specific CSV, NAIP imagery, PDF output
./sat_mmocc/steps/09_visdiff_figures.py \
    --visdiff_csv=/path/to/visdiff_naip_wi_prompt2.csv \
    --imagery_source=naip \
    --fmt=pdf

# Single species
./sat_mmocc/steps/09_visdiff_figures.py \
    --species_ids=00804e75-09ef-44e5-8984-85e365377d47
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence

import fire
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from sat_mmocc.config import (
    cache_path,
    default_image_backbone,
    default_sat_backbone,
)
from sat_mmocc.interpretability_utils import (
    compute_site_scores,
    load_fit_results,
    rank_image_groups,
    resolve_fit_results_path,
)
from sat_mmocc.utils import get_taxon_map

matplotlib.use("Agg")  # headless — no display needed

LOGGER = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_VISDIFF_CSV = cache_path / "visdiff_naip_wi_prompt2.csv"
DEFAULT_OUTPUT_DIR = cache_path / "visdiff_naip_figures_prompt2"
MODALITIES = ["image", "sat", "covariates"]
TOP_K = 50
UNIQUE_WEIGHT = 2.0
N_IMAGES = 5
N_HYPOTHESES = 15
THUMB_SIZE = (192, 192)

IMAGERY_SOURCE_PNG_DIRS = {
    "sentinel": cache_path / "sat_wi_rgb_images_png",
    "naip": cache_path / "naip_wi_images_png",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _unique_location_paths(
    df: pd.DataFrame,
    n: int,
    min_dist_km: float = 50.0,
) -> List[str]:
    """Return up to n image paths from spatially spread locations.

    Selects images in rank order, skipping any site that is within
    *min_dist_km* kilometres of an already-selected site.
    Falls back gracefully when Latitude/Longitude are missing.
    """
    has_coords = "Latitude" in df.columns and "Longitude" in df.columns
    seen_locs: set = set()
    selected_coords: List[tuple] = []  # (lat, lon) of accepted sites
    paths: List[str] = []

    for _, row in df.iterrows():
        loc = row.get("loc_id")
        if loc in seen_locs:
            continue

        if has_coords and min_dist_km > 0 and selected_coords:
            try:
                lat, lon = float(row["Latitude"]), float(row["Longitude"])
                too_close = any(
                    _haversine_km(lat, lon, slat, slon) < min_dist_km
                    for slat, slon in selected_coords
                )
                if too_close:
                    continue
                selected_coords.append((lat, lon))
            except (TypeError, ValueError):
                pass  # missing coords — include anyway

        seen_locs.add(loc)
        paths.append(str(row["image_path"]))
        if len(paths) == n:
            break

    return paths


def _load_thumb(path: str, size: tuple = THUMB_SIZE) -> Optional[np.ndarray]:
    try:
        img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
        return np.asarray(img)
    except Exception:
        return None


def _plot_image_row(axes_row: list, paths: List[str], color: str, label: str) -> None:
    for col_idx, ax in enumerate(axes_row):
        if col_idx < len(paths):
            arr = _load_thumb(paths[col_idx])
            if arr is not None:
                ax.imshow(arr)
                ax.set_title(Path(paths[col_idx]).stem[:24], fontsize=6, color="dimgray")
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
        else:
            ax.set_visible(False)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            spine.set_visible(True)
    if axes_row:
        axes_row[0].set_ylabel(
            label, fontsize=9, color=color, fontweight="bold",
            rotation=90, labelpad=4,
        )


def _plot_hypotheses(ax: plt.Axes, visdiff_df: pd.DataFrame, taxon_id: str, n: int) -> None:
    df = (
        visdiff_df[visdiff_df["taxon_id"] == str(taxon_id)]
        .sort_values("auroc", ascending=False)
        .head(n)
    )
    if df.empty:
        ax.text(0.5, 0.5, "No hypotheses", ha="center", va="center")
        ax.axis("off")
        return

    auroc_vals = df["auroc"].values[::-1]
    labels = [
        (lbl if len(lbl) <= 60 else lbl[:59] + "…")
        for lbl in df["difference"].values[::-1]
    ]

    cmap = plt.cm.RdYlGn
    colors = [cmap(v) for v in np.clip((auroc_vals - 0.4) / 0.4, 0, 1)]
    bars = ax.barh(range(len(labels)), auroc_vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUROC")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlim(
        left=max(0.0, float(auroc_vals.min()) - 0.05),
        right=min(1.0, float(auroc_vals.max()) + 0.05),
    )
    ax.set_title("Top VisDiff hypotheses (Group A > Group B)", fontsize=9)
    for bar, val in zip(bars, auroc_vals):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=7.5,
        )


def generate_species_figure(
    taxon_id: str,
    visdiff_df: pd.DataFrame,
    png_dir: Path,
    modalities: List[str] = MODALITIES,
    image_backbone: str = default_image_backbone,
    sat_backbone: str = default_sat_backbone,
    top_k: int = TOP_K,
    unique_weight: float = UNIQUE_WEIGHT,
    n_images: int = N_IMAGES,
    n_hypotheses: int = N_HYPOTHESES,
    mode: str = "standard",
    min_dist_km: float = 50.0,
) -> Optional[plt.Figure]:
    """Return a combined matplotlib Figure for a single taxon, or None on failure."""
    tid = str(taxon_id)
    try:
        fit_path, res_mod, res_img, res_sat = resolve_fit_results_path(
            tid, modalities, image_backbone, sat_backbone
        )
    except FileNotFoundError as exc:
        LOGGER.warning("Skipping %s — no fit results: %s", tid, exc)
        return None

    fit_results = load_fit_results(fit_path)
    site_scores, display_name = compute_site_scores(
        tid, res_mod, res_img, res_sat, fit_results
    )
    site_scores["image_path"] = site_scores["loc_id"].apply(
        lambda lid: str(png_dir / f"{lid}.png")
    )
    site_scores["image_exists"] = site_scores["image_path"].apply(
        lambda p: Path(p).exists()
    )

    pos_df, neg_df = rank_image_groups(
        site_scores,
        res_mod,
        mode=mode,
        unique_weight=unique_weight,
        top_k=top_k,
        image_modality="sat",
        test=False,
    )

    pos_paths = _unique_location_paths(pos_df, n_images, min_dist_km=min_dist_km)
    neg_paths = _unique_location_paths(neg_df, n_images, min_dist_km=min_dist_km)

    fig = plt.figure(figsize=(n_images * 2.2 + 10, max(7, 0.48 * n_hypotheses + 4)))
    fig.suptitle(
        f"{display_name}  (taxon {tid}, mode={mode})",
        fontsize=13, fontweight="bold", y=1.00,
    )

    outer = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[n_images * 2.2, 10], wspace=0.3,
    )
    left_gs = gridspec.GridSpecFromSubplotSpec(
        2, n_images, subplot_spec=outer[0], hspace=0.06, wspace=0.04,
    )

    for row_idx, (label, color, paths) in enumerate([
        ("Group A — Present", "#2196F3", pos_paths),
        ("Group B — Absent",  "#F44336", neg_paths),
    ]):
        axes_row = [fig.add_subplot(left_gs[row_idx, c]) for c in range(n_images)]
        _plot_image_row(axes_row, paths, color, label)

    ax_hyp = fig.add_subplot(outer[1])
    _plot_hypotheses(ax_hyp, visdiff_df, tid, n_hypotheses)

    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    visdiff_csv: str | Path = DEFAULT_VISDIFF_CSV,
    imagery_source: str = "sentinel",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    fmt: str = "png",
    species_ids: Sequence[str] | str | None = None,
    modalities: str = "image,sat,covariates",
    image_backbone: str = default_image_backbone,
    sat_backbone: str = default_sat_backbone,
    top_k: int = TOP_K,
    unique_weight: float = UNIQUE_WEIGHT,
    n_images: int = N_IMAGES,
    n_hypotheses: int = N_HYPOTHESES,
    mode: str = "standard",
    min_dist_km: float = 50.0,
    dpi: int = 150,
    overwrite: bool = False,
) -> None:
    """Render one figure per species and save to *output_dir*.

    Parameters
    ----------
    visdiff_csv:    Path to VisDiff descriptions CSV.
    imagery_source: "sentinel" or "naip" (controls which PNG directory is used).
    output_dir:     Directory to write figures into (created if needed).
    fmt:            Output format — "png" or "pdf".
    species_ids:    Comma-separated taxon IDs, or omit to process all species in CSV.
    modalities:     Comma-separated list of modalities (must match fit results).
    image_backbone: Image backbone name (must match fit results).
    sat_backbone:   Satellite backbone name (must match fit results).
    top_k:          Number of top/bottom sites to use per group.
    unique_weight:  Uniqueness weight for "unique" ranking mode.
    n_images:       Thumbnails per group row
    n_hypotheses:   Top-N hypotheses displayed in bar chart.
    mode:           Ranking mode: "standard" or "unique".
    min_dist_km:    Minimum great-circle distance (km) between displayed sites (default 50).
    dpi:            Resolution for PNG output.
    overwrite:      Re-render even if output file already exists.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    visdiff_csv = Path(visdiff_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_dir = IMAGERY_SOURCE_PNG_DIRS.get(
        imagery_source, cache_path / f"{imagery_source}_images_png"
    )
    mod_list = [m.strip() for m in modalities.split(",") if m.strip()]

    LOGGER.info("Loading VisDiff CSV: %s", visdiff_csv)
    visdiff_df = pd.read_csv(visdiff_csv)
    visdiff_df["auroc"] = pd.to_numeric(visdiff_df["auroc"], errors="coerce")
    visdiff_df["taxon_id"] = visdiff_df["taxon_id"].astype(str)

    if species_ids is None:
        focal_ids = visdiff_df["taxon_id"].unique().tolist()
    elif isinstance(species_ids, str):
        focal_ids = [s.strip() for s in species_ids.split(",") if s.strip()]
    else:
        focal_ids = list(species_ids)

    taxon_map = get_taxon_map()
    LOGGER.info(
        "Generating figures for %d species → %s (format=%s)",
        len(focal_ids), output_dir, fmt,
    )

    n_ok = 0
    for taxon_id in focal_ids:
        tid = str(taxon_id)
        species_name = taxon_map.get(tid, tid)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in species_name)
        out_path = output_dir / f"{safe_name}__{tid}.{fmt}"

        if out_path.exists() and not overwrite:
            LOGGER.info("Skipping %s — already exists (%s)", species_name, out_path.name)
            n_ok += 1
            continue

        LOGGER.info("Rendering %s (%s) …", species_name, tid)
        try:
            fig = generate_species_figure(
                tid,
                visdiff_df,
                png_dir,
                modalities=mod_list,
                image_backbone=image_backbone,
                sat_backbone=sat_backbone,
                top_k=top_k,
                unique_weight=unique_weight,
                n_images=n_images,
                n_hypotheses=n_hypotheses,
                mode=mode,
                min_dist_km=min_dist_km,
            )
        except Exception as exc:
            LOGGER.error("Failed to render %s: %s", species_name, exc, exc_info=True)
            continue

        if fig is None:
            continue

        save_kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            save_kwargs["dpi"] = dpi

        fig.savefig(out_path, **save_kwargs)
        plt.close(fig)
        LOGGER.info("Saved → %s", out_path)
        n_ok += 1

    LOGGER.info("Done. %d / %d figures saved to %s", n_ok, len(focal_ids), output_dir)


if __name__ == "__main__":
    fire.Fire(main)
