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
"""Generate per-species single-source VisDiff figures (PNG / PDF).

Each figure contains one source block with:
  - 10 Group A images arranged in two sub-rows
  - 10 Group B images arranged in two sub-rows
  - a readable VisDiff description panel
"""

import logging
import textwrap
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
from sat_mmocc.imagery_lookups import get_imagery_source_label, load_imagery_lookup
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
N_IMAGES = 10
N_HYPOTHESES = 15
THUMB_SIZE = (192, 192)
DESCRIPTION_WRAP_WIDTH = 44
DESCRIPTION_FONT_SIZE = 11
IMAGE_SUBROWS = 2
IMAGES_PER_SUBROW = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    radius_km = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def _select_unique_locations(
    df: pd.DataFrame,
    n: int,
    min_dist_km: float = 0.0,
    required_locs: set[str] | None = None,
    exclude_locs: set[str] | None = None,
) -> List[str]:
    """Return up to *n* unique loc_ids in rank order, optionally filtered."""
    has_coords = "Latitude" in df.columns and "Longitude" in df.columns
    excluded = exclude_locs or set()
    selected_coords: List[tuple[float, float]] = []
    loc_ids: List[str] = []

    for _, row in df.iterrows():
        loc = str(row.get("loc_id", "")).strip()
        if not loc or loc in excluded or loc in loc_ids:
            continue
        if required_locs is not None and loc not in required_locs:
            continue

        if has_coords and min_dist_km > 0 and selected_coords:
            try:
                lat, lon = float(row["Latitude"]), float(row["Longitude"])
                too_close = any(
                    _haversine_km(lat, lon, sel_lat, sel_lon) < min_dist_km
                    for sel_lat, sel_lon in selected_coords
                )
                if too_close:
                    continue
                selected_coords.append((lat, lon))
            except (TypeError, ValueError):
                pass
        elif has_coords:
            try:
                selected_coords.append((float(row["Latitude"]), float(row["Longitude"])))
            except (TypeError, ValueError):
                pass

        loc_ids.append(loc)
        if len(loc_ids) == n:
            break

    return loc_ids


def _paths_for_source(
    loc_ids: List[str],
    df: pd.DataFrame,
    fallback_lookup: dict[str, str],
) -> List[str]:
    """Map *loc_ids* to image paths, preferring ranked rows then falling back to lookup."""
    loc_to_path: dict[str, str] = {}
    if not df.empty:
        for _, row in df.iterrows():
            loc = str(row.get("loc_id", "")).strip()
            if loc and loc not in loc_to_path:
                loc_to_path[loc] = str(row.get("image_path", ""))
    return [loc_to_path.get(loc, fallback_lookup.get(loc, "")) for loc in loc_ids]


def _load_thumb(path: str, size: tuple[int, int] = THUMB_SIZE) -> Optional[np.ndarray]:
    try:
        image = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
        return np.asarray(image)
    except Exception:
        return None


def _plot_thumb(ax: plt.Axes, path: str, color: str) -> None:
    thumb = _load_thumb(path)
    if thumb is not None:
        ax.imshow(thumb)
    else:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
    ax.axis("off")
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)
        spine.set_visible(True)


def _plot_row_label(ax: plt.Axes, label: str, color: str) -> None:
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        label,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=color,
        rotation=90,
        transform=ax.transAxes,
    )


def _plot_group_label(ax: plt.Axes, label: str, color: str) -> None:
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        label,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=color,
        transform=ax.transAxes,
    )


def _get_visdiff_panel_text(
    visdiff_df: pd.DataFrame,
    taxon_id: str,
    n: int,
    wrap_width: int = DESCRIPTION_WRAP_WIDTH,
) -> tuple[str, int]:
    subset = visdiff_df[visdiff_df["taxon_id"] == str(taxon_id)].copy()
    if subset.empty:
        body = "No VisDiff descriptions found."
        return body, 1

    subset["difference"] = subset["difference"].fillna("").astype(str).str.strip()
    subset = subset[subset["difference"] != ""]
    subset["auroc"] = pd.to_numeric(subset["auroc"], errors="coerce")
    subset = subset.sort_values("auroc", ascending=False)
    subset = subset.drop_duplicates(subset="difference").head(n)

    if subset.empty:
        body = "No VisDiff descriptions found."
        return body, 1

    blocks = [
        textwrap.fill(
            f"{idx}. {row.difference}",
            width=wrap_width,
            subsequent_indent="    ",
        )
        for idx, row in enumerate(subset.itertuples(index=False), start=1)
    ]
    body = "\n\n".join(blocks)
    return body, body.count("\n") + 1


def _estimate_row_height(description_line_count: int) -> float:
    return max(3.0, 1.8 + 0.20 * description_line_count)


def _split_paths_into_subrows(
    paths: List[str],
    images_per_subrow: int = IMAGES_PER_SUBROW,
) -> List[List[str]]:
    return [
        paths[start : start + images_per_subrow]
        for start in range(0, len(paths), images_per_subrow)
    ]


def _plot_description_panel(
    ax: plt.Axes,
    title: str,
    body: str,
    font_size: float = DESCRIPTION_FONT_SIZE,
) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        title,
        ha="left",
        va="top",
        fontsize=font_size + 1,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.92,
        body,
        ha="left",
        va="top",
        fontsize=font_size,
        linespacing=1.35,
        transform=ax.transAxes,
    )


def _plot_column_header(ax: plt.Axes, label: str) -> None:
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        label,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        transform=ax.transAxes,
    )


def generate_species_figure(
    taxon_id: str,
    visdiff_df: pd.DataFrame,
    image_lookup: pd.DataFrame,
    imagery_label: str,
    modalities: List[str] = MODALITIES,
    image_backbone: str = default_image_backbone,
    sat_backbone: str = default_sat_backbone,
    top_k: int = TOP_K,
    unique_weight: float = UNIQUE_WEIGHT,
    n_images: int = N_IMAGES,
    n_hypotheses: int = N_HYPOTHESES,
    mode: str = "standard",
    min_dist_km: float = 0.0,
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
    site_scores = site_scores.join(image_lookup, on="loc_id", how="left")
    site_scores["image_exists"] = site_scores["image_exists"].fillna(False).astype(bool)
    site_scores["image_path"] = site_scores["image_path"].fillna("").astype(str)

    pos_df, neg_df = rank_image_groups(
        site_scores,
        res_mod,
        mode=mode,
        unique_weight=unique_weight,
        top_k=top_k,
        image_modality="sat",
        test=False,
    )

    image_path_lookup = {
        str(loc_id): str(row["image_path"])
        for loc_id, row in image_lookup.iterrows()
    }
    pos_locs = _select_unique_locations(pos_df, n_images, min_dist_km=min_dist_km)
    neg_locs = _select_unique_locations(neg_df, n_images, min_dist_km=min_dist_km)
    pos_paths = _paths_for_source(pos_locs, pos_df, image_path_lookup)
    neg_paths = _paths_for_source(neg_locs, neg_df, image_path_lookup)

    description_body, description_lines = _get_visdiff_panel_text(
        visdiff_df,
        tid,
        n_hypotheses,
    )

    pos_path_rows = _split_paths_into_subrows(pos_paths)
    neg_path_rows = _split_paths_into_subrows(neg_paths)
    source_height = _estimate_row_height(description_lines + 4.5)
    fig = plt.figure(figsize=(20, source_height + 1.4))
    fig.suptitle(
        f"{display_name}  (taxon {tid}, mode={mode})",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    image_columns = min(IMAGES_PER_SUBROW, max(1, n_images))
    width_ratios = [1.1, 1.7] + [1.0] * image_columns + [7.0]
    outer = gridspec.GridSpec(
        5,
        image_columns + 3,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=[
            0.45,
            source_height / 4,
            source_height / 4,
            source_height / 4,
            source_height / 4,
        ],
        wspace=0.04,
        hspace=0.10,
    )

    _plot_column_header(fig.add_subplot(outer[0, 0]), "Source")
    _plot_column_header(fig.add_subplot(outer[0, 1]), "Group")
    _plot_column_header(fig.add_subplot(outer[0, 2 : 2 + image_columns]), "Example images")
    _plot_column_header(fig.add_subplot(outer[0, -1]), "VisDiff descriptions")

    _plot_row_label(fig.add_subplot(outer[1:, 0]), imagery_label, "#1565C0")
    _plot_group_label(fig.add_subplot(outer[1:3, 1]), "Likely\noccupied", "#2196F3")
    _plot_group_label(fig.add_subplot(outer[3:5, 1]), "Likely\nunoccupied", "#F44336")
    for row_offset, row_paths in enumerate(pos_path_rows[:IMAGE_SUBROWS]):
        for col_idx, path in enumerate(row_paths[:image_columns]):
            _plot_thumb(fig.add_subplot(outer[1 + row_offset, 2 + col_idx]), path, "#2196F3")
    for row_offset, row_paths in enumerate(neg_path_rows[:IMAGE_SUBROWS]):
        for col_idx, path in enumerate(row_paths[:image_columns]):
            _plot_thumb(fig.add_subplot(outer[3 + row_offset, 2 + col_idx]), path, "#F44336")
    _plot_description_panel(
        fig.add_subplot(outer[1:, -1]),
        f"{imagery_label} top {n_hypotheses} VisDiff descriptions",
        description_body,
    )

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
    min_dist_km: float = 0.0,
    dpi: int = 150,
    overwrite: bool = False,
    png_dir: str | Path | None = None,
) -> None:
    """Render one figure per species and save to *output_dir*."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    visdiff_csv = Path(visdiff_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    imagery_label = get_imagery_source_label(imagery_source)
    image_lookup = load_imagery_lookup(
        imagery_source,
        png_dir=Path(png_dir) if png_dir is not None else None,
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
                image_lookup,
                imagery_label,
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
