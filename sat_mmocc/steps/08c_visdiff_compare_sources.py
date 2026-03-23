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
"""Compare NAIP, Sentinel, and ground-level (camera-trap) imagery with per-source VisDiff outputs.

For each species, selects shared locations across all three imagery sources, then renders:

  Left panel (image grid — 6 rows):
    Group A (present) — NAIP
    Group A (present) — Sentinel      } same locations
    Group A (present) — Camera trap   }
    Group B (absent)  — NAIP
    Group B (absent)  — Sentinel      } same locations
    Group B (absent)  — Camera trap   }

  Right panel (3 hypothesis columns):
    NAIP VisDiff  |  Sentinel VisDiff  |  Camera-trap VisDiff

Shared locations are resolved by intersecting all three ranked DataFrames, with the same
haversine spatial-spread filter used in 08a / 08b.  When no 3-way overlap exists, the
script falls back to the 2-way satellite overlap and then to the primary source alone.

Usage examples
--------------
# All species, default CSVs and backbones
./sat_mmocc/steps/08c_visdiff_compare_sources.py

# Specific CSV paths
./sat_mmocc/steps/08c_visdiff_compare_sources.py \\
    --naip_csv=/path/to/visdiff_naip.csv \\
    --sentinel_csv=/path/to/visdiff_sentinel.csv \\
    --ground_csv=/path/to/visdiff_descriptions.csv

# Single species, PDF output
./sat_mmocc/steps/08c_visdiff_compare_sources.py \\
    --species_ids=00804e75-09ef-44e5-8984-85e365377d47 \\
    --fmt=pdf
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from sat_mmocc.utils import get_taxon_map, get_submitit_executor

matplotlib.use("Agg")  # headless — no display needed

LOGGER = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_NAIP_CSV = cache_path / "visdiff_naip_wi_prompt2.csv"
DEFAULT_SENTINEL_CSV = cache_path / "visdiff_sat_wi_prompt2.csv"
DEFAULT_GROUND_CSV = cache_path / "visdiff_descriptions.csv"
DEFAULT_OUTPUT_DIR = cache_path / "visdiff_compare_naipwi_figures"
DEFAULT_PAIRING_CSV = cache_path / "camera_satellite_pairings_v_graft.csv"

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


def _select_locations(
    primary_df: pd.DataFrame,
    n: int,
    required_locs: Optional[set] = None,
    min_dist_km: float = 50.0,
) -> List[str]:
    """Return up to *n* loc_ids from *primary_df* in rank order, spatially spread.

    Parameters
    ----------
    primary_df:    Ranked DataFrame to iterate (must contain 'loc_id').
    n:             Maximum number of locations to return.
    required_locs: When provided, only accept loc_ids that appear in this set.
                   Set to None to impose no cross-source constraint.
    min_dist_km:   Minimum great-circle inter-site distance (km).  0 disables.
    """
    has_coords = "Latitude" in primary_df.columns and "Longitude" in primary_df.columns

    seen_locs: set = set()
    selected_coords: List[Tuple[float, float]] = []
    loc_ids: List[str] = []

    for _, row in primary_df.iterrows():
        loc = str(row.get("loc_id", ""))
        if not loc or loc in seen_locs:
            continue
        if required_locs is not None and loc not in required_locs:
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
) -> None:
    """Render a single row of thumbnail images."""
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

    if axes_row:
        axes_row[0].set_ylabel(
            label, fontsize=7.5, color=color, fontweight="bold",
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
        (lbl if len(lbl) <= 52 else lbl[:51] + "…")
        for lbl in df["difference"].values[::-1]
    ]

    cmap = plt.cm.RdYlGn
    colors = [cmap(v) for v in np.clip((auroc_vals - 0.4) / 0.4, 0, 1)]
    bars = ax.barh(range(len(labels)), auroc_vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
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
            f"{val:.3f}", va="center", fontsize=6.5,
        )


# ── Per-source ranked groups ───────────────────────────────────────────────────

def _get_satellite_groups(
    taxon_id: str,
    image_lookup: pd.DataFrame,
    modalities: List[str],
    image_backbone: str,
    sat_backbone: str,
    top_k: int,
    unique_weight: float,
    mode: str,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, str]]:
    """Load satellite fit results and return (pos_df, neg_df, display_name), or None."""
    tid = str(taxon_id)
    try:
        fit_path, res_mod, res_img, res_sat = resolve_fit_results_path(
            tid, modalities, image_backbone, sat_backbone
        )
    except FileNotFoundError as exc:
        LOGGER.warning("No sat fit results for %s (backbone=%s): %s", tid, sat_backbone, exc)
        return None

    fit_results = load_fit_results(fit_path)
    site_scores, display_name = compute_site_scores(tid, res_mod, res_img, res_sat, fit_results)
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
    return pos_df, neg_df, display_name


def _get_ground_groups(
    taxon_id: str,
    image_lookup: pd.DataFrame,
    modalities: List[str],
    image_backbone: str,
    sat_backbone: str,
    top_k: int,
    unique_weight: float,
    mode: str,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, str]]:
    """Load ground-level (camera-trap) fit results and return (pos_df, neg_df, display_name).

    Camera-trap images are resolved via *image_lookup* indexed by loc_id, built
    from the pairing manifest so each displayed site maps to an explicit triplet.
    Ranking uses ``image_modality="image"`` so that the image-modality score drives
    selection, matching how ``08_visdiff.py`` selects images for VisDiff.
    """
    tid = str(taxon_id)
    try:
        fit_path, res_mod, res_img, res_sat = resolve_fit_results_path(
            tid, modalities, image_backbone, sat_backbone
        )
    except FileNotFoundError as exc:
        LOGGER.warning("No ground fit results for %s: %s", tid, exc)
        return None

    fit_results = load_fit_results(fit_path)
    site_scores, display_name = compute_site_scores(tid, res_mod, res_img, res_sat, fit_results)

    # Join camera-trap image paths from the image lookup (indexed by loc_id)
    site_scores = site_scores.join(image_lookup, on="loc_id", how="left")
    site_scores["image_exists"] = site_scores["image_exists"].fillna(False).astype(bool)
    site_scores["image_path"] = site_scores["image_path"].fillna("").astype(str)

    pos_df, neg_df = rank_image_groups(
        site_scores,
        res_mod,
        mode=mode,
        unique_weight=unique_weight,
        top_k=top_k,
        image_modality="image",
        test=False,
    )
    return pos_df, neg_df, display_name


# ── Shared location resolution ─────────────────────────────────────────────────

def _resolve_shared_locations(
    naip_df: pd.DataFrame,
    sent_df: pd.DataFrame,
    ground_df: pd.DataFrame,
    n: int,
    min_dist_km: float,
    label: str,
) -> List[str]:
    """Find shared loc_ids for one group (pos or neg) across all three sources.

    Falls back progressively:
      1. Try 3-way intersection (NAIP ∩ Sentinel ∩ Camera trap)
      2. Try 2-way intersection (NAIP ∩ Sentinel)
      3. Use primary source (NAIP or whichever is non-empty) unconstrained
    """
    def _locs(df: pd.DataFrame) -> set:
        return set(df["loc_id"].dropna().astype(str)) if not df.empty else set()

    naip_locs = _locs(naip_df)
    sent_locs = _locs(sent_df)
    gnd_locs  = _locs(ground_df)

    primary_df = naip_df if not naip_df.empty else (sent_df if not sent_df.empty else ground_df)

    # 3-way
    three_way = naip_locs & sent_locs & gnd_locs
    locs = _select_locations(primary_df, n, required_locs=three_way, min_dist_km=min_dist_km)
    if locs:
        return locs

    LOGGER.warning("%s: no 3-way shared locations; falling back to satellite 2-way overlap.", label)
    two_way = naip_locs & sent_locs
    locs = _select_locations(primary_df, n, required_locs=two_way, min_dist_km=min_dist_km)
    if locs:
        return locs

    LOGGER.warning("%s: no 2-way shared locations; using primary source unconstrained.", label)
    locs = _select_locations(primary_df, n, required_locs=None, min_dist_km=min_dist_km)
    return locs


# ── Image-path mapping ─────────────────────────────────────────────────────────

def _load_pairing_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Pairing manifest not found: {path}")
    df = pd.read_csv(path)
    required = {
        "loc_id",
        "ground_date_time",
        "Latitude",
        "Longitude",
        "ground_image_path",
        "ground_image_exists",
        "sentinel_image_path",
        "sentinel_exists",
        "naip_image_path",
        "naip_exists",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Pairing manifest {path} missing required columns: {missing}")

    df["loc_id"] = df["loc_id"].astype(str)
    df["ground_date_time"] = pd.to_datetime(df["ground_date_time"], errors="coerce")
    for path_col, exists_col in [
        ("ground_image_path", "ground_image_exists"),
        ("sentinel_image_path", "sentinel_exists"),
        ("naip_image_path", "naip_exists"),
    ]:
        df[path_col] = df[path_col].fillna("").astype(str)
        if exists_col in df.columns:
            df[exists_col] = df[exists_col].fillna(False).astype(bool)
        else:
            df[exists_col] = df[path_col].apply(lambda p: Path(p).exists() if p else False)
    return df


def _build_complete_triplet_lookup(
    pairing_df: pd.DataFrame,
    path_column: str,
    exists_column: str,
) -> pd.DataFrame:
    subset = pairing_df.copy()
    subset = subset[
        subset["ground_image_exists"]
        & subset["sentinel_exists"]
        & subset["naip_exists"]
    ].copy()
    subset = subset[subset[path_column].astype(str).str.strip() != ""]
    subset = subset.sort_values("ground_date_time", na_position="last")
    subset = subset.drop_duplicates(subset="loc_id", keep="first")
    if subset.empty:
        return pd.DataFrame(columns=["image_path", "image_exists", "Latitude", "Longitude"])
    lookup = subset.set_index("loc_id")[[path_column, exists_column, "Latitude", "Longitude"]].rename(
        columns={path_column: "image_path", exists_column: "image_exists"}
    )
    return lookup[["image_path", "image_exists", "Latitude", "Longitude"]]


def _lookup_to_path_map(image_lookup: pd.DataFrame) -> Dict[str, str]:
    if image_lookup.empty or "image_path" not in image_lookup.columns:
        return {}
    return {str(lid): str(row["image_path"]) for lid, row in image_lookup.iterrows()}


def _paths_for_source(
    loc_ids: List[str],
    df: pd.DataFrame,
    fallback_lookup: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Map *loc_ids* → image paths, in loc_id order, from a ranked DataFrame."""
    loc_to_path: Dict[str, str] = {}
    if not df.empty and "image_path" in df.columns:
        for _, row in df.iterrows():
            loc = str(row.get("loc_id", ""))
            if loc and loc not in loc_to_path:
                loc_to_path[loc] = str(row["image_path"])

    result: List[str] = []
    for lid in loc_ids:
        if lid in loc_to_path:
            result.append(loc_to_path[lid])
        elif fallback_lookup is not None and lid in fallback_lookup:
            result.append(fallback_lookup[lid])
        else:
            result.append("")
    return result


# ── Main figure builder ────────────────────────────────────────────────────────

def generate_comparison_figure(
    taxon_id: str,
    naip_visdiff_df: pd.DataFrame,
    sentinel_visdiff_df: pd.DataFrame,
    ground_visdiff_df: pd.DataFrame,
    naip_lookup: pd.DataFrame,
    sentinel_lookup: pd.DataFrame,
    ground_lookup: pd.DataFrame,
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
    """Return a 3-source comparison figure, or None if all sources fail."""
    tid = str(taxon_id)

    naip_result = _get_satellite_groups(tid, naip_lookup, modalities, image_backbone, naip_sat_backbone, top_k, unique_weight, mode)
    sentinel_result = _get_satellite_groups(tid, sentinel_lookup, modalities, image_backbone, sentinel_sat_backbone, top_k, unique_weight, mode)
    ground_result = _get_ground_groups(tid, ground_lookup, modalities, image_backbone, naip_sat_backbone, top_k, unique_weight, mode)

    if naip_result is None and sentinel_result is None and ground_result is None:
        LOGGER.warning("Skipping %s — no fit results for any source.", tid)
        return None

    # Display name from first available source
    display_name = tid
    for result in (naip_result, sentinel_result, ground_result):
        if result is not None:
            display_name = result[2]
            break

    empty_df = pd.DataFrame(columns=["loc_id", "image_path", "image_exists", "Latitude", "Longitude"])

    naip_pos_df,  naip_neg_df  = (naip_result[0],     naip_result[1])     if naip_result     else (empty_df, empty_df)
    sent_pos_df,  sent_neg_df  = (sentinel_result[0], sentinel_result[1]) if sentinel_result else (empty_df, empty_df)
    gnd_pos_df,   gnd_neg_df   = (ground_result[0],   ground_result[1])   if ground_result   else (empty_df, empty_df)

    # ── Shared location selection ──────────────────────────────────────────────
    shared_pos_locs = _resolve_shared_locations(
        naip_pos_df, sent_pos_df, gnd_pos_df, n_images, min_dist_km,
        label=f"{tid} Group A",
    )
    shared_neg_locs = _resolve_shared_locations(
        naip_neg_df, sent_neg_df, gnd_neg_df, n_images, min_dist_km,
        label=f"{tid} Group B",
    )

    # ── Build image path lists per source (all referencing same locations) ─────
    naip_path_lookup = _lookup_to_path_map(naip_lookup)
    sentinel_path_lookup = _lookup_to_path_map(sentinel_lookup)
    ground_path_lookup = _lookup_to_path_map(ground_lookup)

    naip_pos_paths = _paths_for_source(shared_pos_locs, naip_pos_df, naip_path_lookup)
    sent_pos_paths = _paths_for_source(shared_pos_locs, sent_pos_df, sentinel_path_lookup)
    gnd_pos_paths = _paths_for_source(shared_pos_locs, gnd_pos_df, ground_path_lookup)

    naip_neg_paths = _paths_for_source(shared_neg_locs, naip_neg_df, naip_path_lookup)
    sent_neg_paths = _paths_for_source(shared_neg_locs, sent_neg_df, sentinel_path_lookup)
    gnd_neg_paths = _paths_for_source(shared_neg_locs, gnd_neg_df, ground_path_lookup)

    # ── Figure layout ──────────────────────────────────────────────────────────
    #  columns: [image grid (6 rows)] | [NAIP hyp] | [Sentinel hyp] | [Ground hyp]
    img_width = n_images * 2.2
    hyp_width = 8.5
    fig_h = max(10, 0.42 * n_hypotheses + 6)

    fig = plt.figure(figsize=(img_width + 3 * hyp_width + 1.2, fig_h))
    fig.suptitle(
        f"{display_name}  (taxon {tid})",
        fontsize=13, fontweight="bold", y=1.01,
    )

    outer = gridspec.GridSpec(
        1, 4, figure=fig,
        width_ratios=[img_width, hyp_width, hyp_width, hyp_width],
        wspace=0.38,
    )

    # Left: 6-row image grid
    left_gs = gridspec.GridSpecFromSubplotSpec(
        6, n_images, subplot_spec=outer[0], hspace=0.10, wspace=0.04,
    )

    row_specs = [
        ("Group A — NAIP",         "#1565C0", naip_pos_paths),
        ("Group A — Sentinel",     "#42A5F5", sent_pos_paths),
        ("Group A — Camera trap",  "#00796B", gnd_pos_paths),
        ("Group B — NAIP",         "#B71C1C", naip_neg_paths),
        ("Group B — Sentinel",     "#EF5350", sent_neg_paths),
        ("Group B — Camera trap",  "#FF8F00", gnd_neg_paths),
    ]
    for row_idx, (label, color, paths) in enumerate(row_specs):
        axes_row = [fig.add_subplot(left_gs[row_idx, c]) for c in range(n_images)]
        _plot_image_row(axes_row, paths, color, label)

    # Hypothesis panels
    source_panels = [
        (outer[1], naip_visdiff_df,     "NAIP — VisDiff"),
        (outer[2], sentinel_visdiff_df, "Sentinel — VisDiff"),
        (outer[3], ground_visdiff_df,   "Camera trap — VisDiff"),
    ]
    for subplot_spec, vdf, title in source_panels:
        ax = fig.add_subplot(subplot_spec)
        _plot_hypotheses(ax, vdf, tid, n_hypotheses, title=title)

    plt.tight_layout()
    return fig


# ── CLI entry point ────────────────────────────────────────────────────────────

def main(
    naip_csv: str | Path = DEFAULT_NAIP_CSV,
    sentinel_csv: str | Path = DEFAULT_SENTINEL_CSV,
    ground_csv: str | Path = DEFAULT_GROUND_CSV,
    pairing_csv: str | Path = DEFAULT_PAIRING_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    fmt: str = "png",
    species_ids: Sequence[str] | str | None = None,
    modalities: str = "image,sat,covariates",
    image_backbone: str = default_image_backbone,
    naip_sat_backbone: str = default_sat_backbone,
    sentinel_sat_backbone: str = default_sat_backbone,
    top_k: int = TOP_K,
    unique_weight: float = UNIQUE_WEIGHT,
    n_images: int = N_IMAGES,
    n_hypotheses: int = N_HYPOTHESES,
    mode: str = "standard",
    min_dist_km: float = 50.0,
    dpi: int = 150,
    overwrite: bool = False,
) -> None:
    """Render per-species 3-source comparison figures (NAIP / Sentinel / Camera trap).

    Parameters
    ----------
    naip_csv:              Path to NAIP VisDiff descriptions CSV.
    sentinel_csv:          Path to Sentinel VisDiff descriptions CSV.
    ground_csv:            Path to ground-level (camera-trap) VisDiff descriptions CSV.
    pairing_csv:           Pairing manifest linking one camera image to one Sentinel and one NAIP image.
    output_dir:            Output directory (created if needed).
    fmt:                   "png" or "pdf".
    species_ids:           Comma-separated taxon IDs, or omit for all species in CSVs.
    modalities:            Comma-separated modality list (must match fit results).
    image_backbone:        Camera-trap image backbone name (used for all fit results).
    naip_sat_backbone:     Satellite backbone used for NAIP fit results.
    sentinel_sat_backbone: Satellite backbone used for Sentinel fit results.
    top_k:                 Number of top/bottom sites per group.
    unique_weight:         Uniqueness weight for "unique" ranking mode.
    n_images:              Thumbnails per source row (5 → 30 thumbnails total per species).
    n_hypotheses:          Top-N hypotheses per source column.
    mode:                  Ranking mode: "standard" or "unique".
    min_dist_km:           Minimum great-circle distance (km) between displayed sites.
    dpi:                   Resolution for PNG output.
    overwrite:             Re-render even if output file already exists.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    naip_csv = Path(naip_csv)
    sentinel_csv = Path(sentinel_csv)
    ground_csv = Path(ground_csv)
    pairing_csv = Path(pairing_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mod_list = [m.strip() for m in modalities.split(",") if m.strip()]

    LOGGER.info("Loading NAIP CSV:     %s", naip_csv)
    LOGGER.info("Loading Sentinel CSV: %s", sentinel_csv)
    LOGGER.info("Loading Ground CSV:   %s", ground_csv)
    LOGGER.info("Loading pairing CSV:  %s", pairing_csv)

    def _load_visdiff(path: Path) -> pd.DataFrame:
        if not path.exists():
            LOGGER.warning("CSV not found: %s — using empty DataFrame", path)
            return pd.DataFrame(columns=["taxon_id", "auroc", "difference"])
        df = pd.read_csv(path)
        df["auroc"] = pd.to_numeric(df["auroc"], errors="coerce")
        df["taxon_id"] = df["taxon_id"].astype(str)
        return df

    naip_visdiff    = _load_visdiff(naip_csv)
    sentinel_visdiff= _load_visdiff(sentinel_csv)
    ground_visdiff  = _load_visdiff(ground_csv)

    all_ids = (
        set(naip_visdiff["taxon_id"].unique())
        | set(sentinel_visdiff["taxon_id"].unique())
        | set(ground_visdiff["taxon_id"].unique())
    )

    if species_ids is None:
        focal_ids = sorted(all_ids)
    elif isinstance(species_ids, str):
        focal_ids = [s.strip() for s in species_ids.split(",") if s.strip()]
    else:
        focal_ids = list(species_ids)

    pairing_manifest = _load_pairing_manifest(pairing_csv)
    naip_lookup = _build_complete_triplet_lookup(
        pairing_manifest, "naip_image_path", "naip_exists"
    )
    sentinel_lookup = _build_complete_triplet_lookup(
        pairing_manifest, "sentinel_image_path", "sentinel_exists"
    )
    ground_lookup = _build_complete_triplet_lookup(
        pairing_manifest, "ground_image_path", "ground_image_exists"
    )

    taxon_map = get_taxon_map()
    LOGGER.info(
        "Generating 3-source comparison figures for %d species → %s (format=%s)",
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
                ground_visdiff_df=ground_visdiff,
                naip_lookup=naip_lookup,
                sentinel_lookup=sentinel_lookup,
                ground_lookup=ground_lookup,
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
