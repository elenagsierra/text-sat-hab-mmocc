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
"""Compare NAIP vs Sentinel imagery side-by-side with per-source VisDiff outputs.

For each species, selects sites that rank highly under both sources, then renders:

  Left panel (image grid):
    Group A (Present) — NAIP row
    Group A (Present) — Sentinel row   } same locations
    Group B (Absent)  — NAIP row
    Group B (Absent)  — Sentinel row   } same locations

  Right panel (two columns):
    NAIP VisDiff hypotheses | Sentinel VisDiff hypotheses

Usage examples
--------------
# All species, default CSVs and backbones
./sat_mmocc/steps/08b_visdiff_compare_sources.py

# Specific CSV paths
./sat_mmocc/steps/08b_visdiff_compare_sources.py \\
    --naip_csv=/path/to/visdiff_naip.csv \\
    --sentinel_csv=/path/to/visdiff_sentinel.csv

# Single species, PDF output
./sat_mmocc/steps/08b_visdiff_compare_sources.py \\
    --species_ids=00804e75-09ef-44e5-8984-85e365377d47 \\
    --fmt=pdf
"""

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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
from sat_mmocc.imagery_lookups import (
    get_imagery_source_label,
    load_imagery_lookup,
    lookup_to_path_map,
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
DEFAULT_NAIP_CSV = cache_path / "visdiff_naip_wi_prompt2.csv"
DEFAULT_SENTINEL_CSV = cache_path / "visdiff_sat_wi_prompt2.csv"
DEFAULT_OUTPUT_DIR = cache_path / "visdiff_compare_figures"
DEFAULT_VISDIFF_CSVS = {
    "naip": DEFAULT_NAIP_CSV,
    "sentinel": DEFAULT_SENTINEL_CSV,
    "naip_v_graft": cache_path / "visdiff_naip_v_graft_descriptions.csv",
    "sentinel_v_graft": cache_path / "visdiff_sentinel_v_graft_descriptions.csv",
}

MODALITIES = ["image", "sat", "covariates"]
TOP_K = 50
UNIQUE_WEIGHT = 2.0
N_IMAGES = 5
N_HYPOTHESES = 12
THUMB_SIZE = (192, 192)


# ── Spatial helpers ────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _select_shared_locations(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    n: int,
    min_dist_km: float = 50.0,
) -> List[str]:
    """Return up to *n* loc_ids present in both DataFrames, spatially spread.

    Iterates *primary_df* in rank order and accepts a location only when:
      1. The loc_id also appears in *secondary_df*.
      2. No already-selected site is within *min_dist_km* km.

    Parameters
    ----------
    primary_df:     Ranked DataFrame for the reference source (e.g. NAIP).
    secondary_df:   Ranked DataFrame for the comparison source (e.g. Sentinel).
    n:              Maximum number of locations to return.
    min_dist_km:    Minimum inter-site distance (km).  Set to 0 to disable.
    """
    secondary_locs = set(secondary_df["loc_id"].dropna().astype(str))
    has_coords = "Latitude" in primary_df.columns and "Longitude" in primary_df.columns

    seen_locs: set = set()
    selected_coords: List[Tuple[float, float]] = []
    loc_ids: List[str] = []

    for _, row in primary_df.iterrows():
        loc = str(row.get("loc_id", ""))
        if not loc or loc in seen_locs:
            continue
        if loc not in secondary_locs:
            continue  # location not available in the other source

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
        loc_ids.append(loc)
        if len(loc_ids) == n:
            break

    return loc_ids


# ── Rendering helpers ──────────────────────────────────────────────────────────

def _load_thumb(path: str, size: Tuple[int, int] = THUMB_SIZE) -> Optional[np.ndarray]:
    try:
        img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
        return np.asarray(img)
    except Exception:
        return None


def _plot_image_row(
    axes_row: list,
    paths: List[str],
    color: str,
    label: str,
    label_side: str = "left",
) -> None:
    """Render one row of thumbnail images onto *axes_row*."""
    for col_idx, ax in enumerate(axes_row):
        if col_idx < len(paths):
            arr = _load_thumb(paths[col_idx])
            if arr is not None:
                ax.imshow(arr)
                ax.set_title(Path(paths[col_idx]).stem[:22], fontsize=5.5, color="dimgray")
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
        else:
            ax.set_visible(False)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    if axes_row and label_side == "left":
        axes_row[0].set_ylabel(
            label, fontsize=8, color=color, fontweight="bold",
            rotation=90, labelpad=4,
        )


def _plot_hypotheses(
    ax: plt.Axes,
    visdiff_df: pd.DataFrame,
    taxon_id: str,
    n: int,
    title: str = "VisDiff hypotheses",
) -> None:
    """Horizontal bar chart of top VisDiff hypotheses for *taxon_id*."""
    df = (
        visdiff_df[visdiff_df["taxon_id"] == str(taxon_id)]
        .sort_values("auroc", ascending=False)
        .head(n)
    )
    if df.empty:
        ax.text(0.5, 0.5, "No hypotheses", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    auroc_vals = df["auroc"].values[::-1]
    labels = [
        (lbl if len(lbl) <= 55 else lbl[:54] + "…")
        for lbl in df["difference"].values[::-1]
    ]

    cmap = plt.cm.RdYlGn
    colors = [cmap(v) for v in np.clip((auroc_vals - 0.4) / 0.4, 0, 1)]
    bars = ax.barh(range(len(labels)), auroc_vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("AUROC", fontsize=8)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlim(
        left=max(0.0, float(auroc_vals.min()) - 0.05),
        right=min(1.0, float(auroc_vals.max()) + 0.05),
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    for bar, val in zip(bars, auroc_vals):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=7,
        )


# ── Per-source ranked groups ───────────────────────────────────────────────────

def _get_ranked_groups(
    taxon_id: str,
    image_lookup: pd.DataFrame,
    modalities: List[str],
    image_backbone: str,
    sat_backbone: str,
    top_k: int,
    unique_weight: float,
    mode: str,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load fit results and return (pos_df, neg_df) ranked DataFrames, or None."""
    tid = str(taxon_id)
    try:
        fit_path, res_mod, res_img, res_sat = resolve_fit_results_path(
            tid, modalities, image_backbone, sat_backbone
        )
    except FileNotFoundError as exc:
        LOGGER.warning("No fit results for %s (sat_backbone=%s): %s", tid, sat_backbone, exc)
        return None

    fit_results = load_fit_results(fit_path)
    site_scores, _ = compute_site_scores(tid, res_mod, res_img, res_sat, fit_results)
    site_scores = site_scores.join(image_lookup, on="loc_id", how="left")
    site_scores["image_exists"] = site_scores["image_exists"].fillna(False).astype(bool)

    pos_df, neg_df = rank_image_groups(
        site_scores,
        res_mod,
        mode=mode,
        unique_weight=unique_weight,
        top_k=top_k,
        image_modality="sat",
        test=False,
    )
    return pos_df, neg_df


# ── Main figure builder ────────────────────────────────────────────────────────

def generate_comparison_figure(
    taxon_id: str,
    naip_visdiff_df: pd.DataFrame,
    sentinel_visdiff_df: pd.DataFrame,
    naip_imagery_source: str,
    sentinel_imagery_source: str,
    naip_png_dir: Path | None = None,
    sentinel_png_dir: Path | None = None,
    modalities: List[str] = MODALITIES,
    image_backbone: str = default_image_backbone,
    naip_sat_backbone: str = default_sat_backbone,
    sentinel_sat_backbone: str = default_sat_backbone,
    top_k: int = TOP_K,
    unique_weight: float = UNIQUE_WEIGHT,
    n_images: int = N_IMAGES,
    n_hypotheses: int = N_HYPOTHESES,
    mode: str = "standard",
    min_dist_km: float = 50.0,
) -> Optional[plt.Figure]:
    """Return a side-by-side NAIP/Sentinel comparison figure, or None on failure."""
    tid = str(taxon_id)
    naip_lookup = load_imagery_lookup(naip_imagery_source, png_dir=naip_png_dir)
    sentinel_lookup = load_imagery_lookup(
        sentinel_imagery_source, png_dir=sentinel_png_dir
    )
    naip_label = get_imagery_source_label(naip_imagery_source)
    sentinel_label = get_imagery_source_label(sentinel_imagery_source)

    naip_result = _get_ranked_groups(
        tid, naip_lookup, modalities, image_backbone, naip_sat_backbone,
        top_k, unique_weight, mode,
    )
    sentinel_result = _get_ranked_groups(
        tid, sentinel_lookup, modalities, image_backbone, sentinel_sat_backbone,
        top_k, unique_weight, mode,
    )

    if naip_result is None and sentinel_result is None:
        LOGGER.warning("Skipping %s — no fit results for either source.", tid)
        return None

    # Retrieve display name from whichever source succeeded
    display_name = tid
    for sat_backbone in [naip_sat_backbone, sentinel_sat_backbone]:
        try:
            fit_path, res_mod, res_img, res_sat = resolve_fit_results_path(
                tid, modalities, image_backbone, sat_backbone
            )
            _, display_name = compute_site_scores(
                tid, res_mod, res_img, res_sat, load_fit_results(fit_path)
            )
            break
        except Exception:
            continue

    # ── Select shared locations per group ──────────────────────────────────────
    def _paths_for_source(
        loc_ids: List[str],
        df: pd.DataFrame,
        fallback_lookup: dict[str, str],
    ) -> List[str]:
        """Map selected loc_ids → image paths (in loc_id order) from a ranked df."""
        loc_to_path: dict = {}
        for _, row in df.iterrows():
            loc = str(row.get("loc_id", ""))
            if loc not in loc_to_path:
                loc_to_path[loc] = str(row.get("image_path", ""))
        return [loc_to_path.get(lid, fallback_lookup.get(lid, "")) for lid in loc_ids]

    # Fallback to empty df when one source is missing
    empty_df = pd.DataFrame(columns=["loc_id", "image_path", "image_exists", "Latitude", "Longitude"])

    naip_pos_df, naip_neg_df = naip_result if naip_result else (empty_df, empty_df)
    sent_pos_df, sent_neg_df = sentinel_result if sentinel_result else (empty_df, empty_df)

    # Primary DFs for location ordering: prefer whichever source gave results
    primary_pos = naip_pos_df if naip_result else sent_pos_df
    primary_neg = naip_neg_df if naip_result else sent_neg_df
    secondary_pos = sent_pos_df if naip_result else empty_df
    secondary_neg = sent_neg_df if naip_result else empty_df
    primary_label = naip_label if naip_result else sentinel_label

    shared_pos_locs = _select_shared_locations(primary_pos, secondary_pos, n_images, min_dist_km)
    shared_neg_locs = _select_shared_locations(primary_neg, secondary_neg, n_images, min_dist_km)

    # Fall back to best-available locations when no overlap
    if not shared_pos_locs:
        LOGGER.warning("%s: no shared Group A locations; falling back to %s.", tid, primary_label)
        shared_pos_locs = (
            primary_pos["loc_id"].dropna().astype(str).unique()[:n_images].tolist()
        )
    if not shared_neg_locs:
        LOGGER.warning("%s: no shared Group B locations; falling back to %s.", tid, primary_label)
        shared_neg_locs = (
            primary_neg["loc_id"].dropna().astype(str).unique()[:n_images].tolist()
        )

    naip_path_map = lookup_to_path_map(naip_lookup)
    sentinel_path_map = lookup_to_path_map(sentinel_lookup)
    naip_pos_paths = _paths_for_source(shared_pos_locs, naip_pos_df, naip_path_map)
    sent_pos_paths = _paths_for_source(shared_pos_locs, sent_pos_df, sentinel_path_map)
    naip_neg_paths = _paths_for_source(shared_neg_locs, naip_neg_df, naip_path_map)
    sent_neg_paths = _paths_for_source(shared_neg_locs, sent_neg_df, sentinel_path_map)

    # ── Figure layout ──────────────────────────────────────────────────────────
    #  [ image grid (4 rows × n_images cols) | hypothesis A | hypothesis B ]
    img_width = n_images * 2.2
    hyp_width = 9.0
    fig_h = max(8, 0.45 * n_hypotheses + 5)

    fig = plt.figure(figsize=(img_width + 2 * hyp_width + 0.8, fig_h))
    fig.suptitle(
        f"{display_name}  (taxon {tid})",
        fontsize=13, fontweight="bold", y=1.01,
    )

    outer = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[img_width, hyp_width, hyp_width],
        wspace=0.35,
    )

    # Left: 4-row image grid (pos-NAIP, pos-Sentinel, neg-NAIP, neg-Sentinel)
    left_gs = gridspec.GridSpecFromSubplotSpec(
        4, n_images, subplot_spec=outer[0], hspace=0.12, wspace=0.04,
    )

    row_specs = [
        (f"Group A (present) — {naip_label}",     "#2196F3", naip_pos_paths),
        (f"Group A (present) — {sentinel_label}", "#64B5F6", sent_pos_paths),
        (f"Group B (absent)  — {naip_label}",     "#F44336", naip_neg_paths),
        (f"Group B (absent)  — {sentinel_label}", "#EF9A9A", sent_neg_paths),
    ]
    for row_idx, (label, color, paths) in enumerate(row_specs):
        axes_row = [fig.add_subplot(left_gs[row_idx, c]) for c in range(n_images)]
        _plot_image_row(axes_row, paths, color, label)

    # Middle: NAIP hypotheses
    ax_naip = fig.add_subplot(outer[1])
    _plot_hypotheses(ax_naip, naip_visdiff_df, tid, n_hypotheses, title=f"{naip_label} — VisDiff hypotheses")

    # Right: Sentinel hypotheses
    ax_sent = fig.add_subplot(outer[2])
    _plot_hypotheses(ax_sent, sentinel_visdiff_df, tid, n_hypotheses, title=f"{sentinel_label} — VisDiff hypotheses")

    plt.tight_layout()
    return fig


# ── CLI entry point ────────────────────────────────────────────────────────────

def main(
    naip_csv: str | Path | None = None,
    sentinel_csv: str | Path | None = None,
    naip_png_dir: str | Path | None = None,
    sentinel_png_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    fmt: str = "png",
    species_ids: Sequence[str] | str | None = None,
    modalities: str = "image,sat,covariates",
    image_backbone: str = default_image_backbone,
    naip_sat_backbone: str = default_sat_backbone,
    sentinel_sat_backbone: str = default_sat_backbone,
    naip_imagery_source: str = "naip",
    sentinel_imagery_source: str = "sentinel",
    top_k: int = TOP_K,
    unique_weight: float = UNIQUE_WEIGHT,
    n_images: int = N_IMAGES,
    n_hypotheses: int = N_HYPOTHESES,
    mode: str = "standard",
    min_dist_km: float = 50.0,
    dpi: int = 150,
    overwrite: bool = False,
) -> None:
    """Render per-species NAIP vs Sentinel comparison figures and save to *output_dir*.

    Parameters
    ----------
    naip_csv:              Path to NAIP-side VisDiff descriptions CSV.
    sentinel_csv:          Path to Sentinel-side VisDiff descriptions CSV.
    naip_png_dir:          Optional directory override for standard NAIP PNGs.
    sentinel_png_dir:      Optional directory override for standard Sentinel PNGs.
    output_dir:            Directory to write figures into (created if needed).
    fmt:                   Output format — "png" or "pdf".
    species_ids:           Comma-separated taxon IDs, or omit for all species in CSVs.
    modalities:            Comma-separated modality list (must match fit results).
    image_backbone:        Camera-trap image backbone name.
    naip_sat_backbone:     Satellite backbone used for NAIP fit results.
    sentinel_sat_backbone: Satellite backbone used for Sentinel fit results.
    naip_imagery_source:   Imagery source used for the NAIP-side panels.
    sentinel_imagery_source: Imagery source used for the Sentinel-side panels.
    top_k:                 Number of top/bottom sites per group.
    unique_weight:         Uniqueness weight for "unique" ranking mode.
    n_images:              Thumbnails per source row (5 → 20 thumbnails total per species).
    n_hypotheses:          Top-N hypotheses shown per source.
    mode:                  Ranking mode: "standard" or "unique".
    min_dist_km:           Minimum great-circle distance (km) between displayed sites.
    dpi:                   Resolution for PNG output.
    overwrite:             Re-render even if output file already exists.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    naip_csv = Path(naip_csv) if naip_csv is not None else DEFAULT_VISDIFF_CSVS.get(naip_imagery_source, DEFAULT_NAIP_CSV)
    sentinel_csv = Path(sentinel_csv) if sentinel_csv is not None else DEFAULT_VISDIFF_CSVS.get(sentinel_imagery_source, DEFAULT_SENTINEL_CSV)
    naip_png_dir = Path(naip_png_dir) if naip_png_dir is not None else None
    sentinel_png_dir = Path(sentinel_png_dir) if sentinel_png_dir is not None else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mod_list = [m.strip() for m in modalities.split(",") if m.strip()]

    LOGGER.info("Loading NAIP CSV:     %s", naip_csv)
    LOGGER.info("Loading Sentinel CSV: %s", sentinel_csv)
    LOGGER.info("NAIP imagery source:     %s", naip_imagery_source)
    LOGGER.info("Sentinel imagery source: %s", sentinel_imagery_source)

    def _load_visdiff(path: Path) -> pd.DataFrame:
        if not path.exists():
            LOGGER.warning("CSV not found: %s — proceeding with empty DataFrame", path)
            return pd.DataFrame(columns=["taxon_id", "auroc", "difference"])
        df = pd.read_csv(path)
        df["auroc"] = pd.to_numeric(df["auroc"], errors="coerce")
        df["taxon_id"] = df["taxon_id"].astype(str)
        return df

    naip_visdiff = _load_visdiff(naip_csv)
    sentinel_visdiff = _load_visdiff(sentinel_csv)

    # Union of species across both CSVs
    all_ids = set(naip_visdiff["taxon_id"].unique()) | set(sentinel_visdiff["taxon_id"].unique())

    if species_ids is None:
        focal_ids = sorted(all_ids)
    elif isinstance(species_ids, str):
        focal_ids = [s.strip() for s in species_ids.split(",") if s.strip()]
    else:
        focal_ids = list(species_ids)

    taxon_map = get_taxon_map()
    LOGGER.info(
        "Generating comparison figures for %d species → %s (format=%s)",
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
            fig = generate_comparison_figure(
                tid,
                naip_visdiff_df=naip_visdiff,
                sentinel_visdiff_df=sentinel_visdiff,
                naip_imagery_source=naip_imagery_source,
                sentinel_imagery_source=sentinel_imagery_source,
                naip_png_dir=naip_png_dir,
                sentinel_png_dir=sentinel_png_dir,
                modalities=mod_list,
                image_backbone=image_backbone,
                naip_sat_backbone=naip_sat_backbone,
                sentinel_sat_backbone=sentinel_sat_backbone,
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

        save_kwargs: dict = {"bbox_inches": "tight"}
        if fmt == "png":
            save_kwargs["dpi"] = dpi

        fig.savefig(out_path, **save_kwargs)
        plt.close(fig)
        LOGGER.info("Saved → %s", out_path)
        n_ok += 1

    LOGGER.info("Done. %d / %d figures saved to %s", n_ok, len(focal_ids), output_dir)


if __name__ == "__main__":
    fire.Fire(main)
