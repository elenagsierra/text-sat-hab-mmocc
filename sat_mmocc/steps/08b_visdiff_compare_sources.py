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
"""Compare NAIP vs Sentinel imagery with one readable row per source."""

import logging
import textwrap
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
N_IMAGES = 10
N_HYPOTHESES = 12
THUMB_SIZE = (192, 192)
DESCRIPTION_WRAP_WIDTH = 44
DESCRIPTION_FONT_SIZE = 11
IMAGE_SUBROWS = 2
IMAGES_PER_SUBROW = 5


# ── Spatial helpers ────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    radius_km = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def _seed_coords(df: pd.DataFrame, seed_locs: Sequence[str] | None) -> List[Tuple[float, float]]:
    if not seed_locs or "Latitude" not in df.columns or "Longitude" not in df.columns:
        return []
    seed_set = {str(loc) for loc in seed_locs}
    coords: List[Tuple[float, float]] = []
    for _, row in df.iterrows():
        loc = str(row.get("loc_id", "")).strip()
        if loc not in seed_set:
            continue
        try:
            coords.append((float(row["Latitude"]), float(row["Longitude"])))
        except (TypeError, ValueError):
            continue
    return coords


def _select_unique_locations(
    df: pd.DataFrame,
    n: int,
    min_dist_km: float = 50.0,
    required_locs: set[str] | None = None,
    exclude_locs: set[str] | None = None,
    seed_locs: Sequence[str] | None = None,
) -> List[str]:
    """Return up to *n* unique loc_ids in rank order, optionally filtered."""
    has_coords = "Latitude" in df.columns and "Longitude" in df.columns
    excluded = exclude_locs or set()
    selected_coords = _seed_coords(df, seed_locs)
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


# ── Rendering helpers ──────────────────────────────────────────────────────────

def _load_thumb(path: str, size: Tuple[int, int] = THUMB_SIZE) -> Optional[np.ndarray]:
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


def _paths_for_source(
    loc_ids: List[str],
    df: pd.DataFrame,
    fallback_lookup: dict[str, str],
) -> List[str]:
    loc_to_path: dict[str, str] = {}
    if not df.empty and "image_path" in df.columns:
        for _, row in df.iterrows():
            loc = str(row.get("loc_id", "")).strip()
            if loc and loc not in loc_to_path:
                loc_to_path[loc] = str(row.get("image_path", ""))
    return [loc_to_path.get(loc, fallback_lookup.get(loc, "")) for loc in loc_ids]


def _build_satellite_group_paths(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    primary_lookup: dict[str, str],
    secondary_lookup: dict[str, str],
    n: int,
    min_dist_km: float,
) -> tuple[List[str], List[str]]:
    shared_loc_ids = _select_unique_locations(
        primary_df,
        n,
        min_dist_km=min_dist_km,
        required_locs=set(secondary_df["loc_id"].dropna().astype(str)),
    )

    primary_locs = list(shared_loc_ids)
    secondary_locs = list(shared_loc_ids)

    if len(primary_locs) < n:
        primary_locs.extend(
            _select_unique_locations(
                primary_df,
                n - len(primary_locs),
                min_dist_km=min_dist_km,
                exclude_locs=set(primary_locs),
                seed_locs=primary_locs,
            )
        )
    if len(secondary_locs) < n:
        secondary_locs.extend(
            _select_unique_locations(
                secondary_df,
                n - len(secondary_locs),
                min_dist_km=min_dist_km,
                exclude_locs=set(secondary_locs),
                seed_locs=secondary_locs,
            )
        )

    return (
        _paths_for_source(primary_locs, primary_df, primary_lookup),
        _paths_for_source(secondary_locs, secondary_df, secondary_lookup),
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
    min_dist_km: float = 0.0,
) -> Optional[plt.Figure]:
    """Return a side-by-side NAIP/Sentinel comparison figure, or None on failure."""
    tid = str(taxon_id)
    naip_lookup = load_imagery_lookup(naip_imagery_source, png_dir=naip_png_dir)
    sentinel_lookup = load_imagery_lookup(
        sentinel_imagery_source,
        png_dir=sentinel_png_dir,
    )
    naip_label = get_imagery_source_label(naip_imagery_source)
    sentinel_label = get_imagery_source_label(sentinel_imagery_source)

    naip_result = _get_ranked_groups(
        tid,
        naip_lookup,
        modalities,
        image_backbone,
        naip_sat_backbone,
        top_k,
        unique_weight,
        mode,
    )
    sentinel_result = _get_ranked_groups(
        tid,
        sentinel_lookup,
        modalities,
        image_backbone,
        sentinel_sat_backbone,
        top_k,
        unique_weight,
        mode,
    )

    if naip_result is None and sentinel_result is None:
        LOGGER.warning("Skipping %s — no fit results for either source.", tid)
        return None

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

    empty_df = pd.DataFrame(
        columns=["loc_id", "image_path", "image_exists", "Latitude", "Longitude"]
    )
    naip_pos_df, naip_neg_df = naip_result if naip_result else (empty_df, empty_df)
    sent_pos_df, sent_neg_df = sentinel_result if sentinel_result else (empty_df, empty_df)

    naip_path_map = lookup_to_path_map(naip_lookup)
    sentinel_path_map = lookup_to_path_map(sentinel_lookup)
    naip_pos_paths, sent_pos_paths = _build_satellite_group_paths(
        naip_pos_df,
        sent_pos_df,
        naip_path_map,
        sentinel_path_map,
        n_images,
        min_dist_km,
    )
    naip_neg_paths, sent_neg_paths = _build_satellite_group_paths(
        naip_neg_df,
        sent_neg_df,
        naip_path_map,
        sentinel_path_map,
        n_images,
        min_dist_km,
    )

    naip_body, naip_lines = _get_visdiff_panel_text(naip_visdiff_df, tid, n_hypotheses)
    sent_body, sent_lines = _get_visdiff_panel_text(
        sentinel_visdiff_df,
        tid,
        n_hypotheses,
    )
    naip_pos_rows = _split_paths_into_subrows(naip_pos_paths)
    naip_neg_rows = _split_paths_into_subrows(naip_neg_paths)
    sent_pos_rows = _split_paths_into_subrows(sent_pos_paths)
    sent_neg_rows = _split_paths_into_subrows(sent_neg_paths)
    source_heights = [
        _estimate_row_height(naip_lines + 2),
        _estimate_row_height(sent_lines + 2),
    ]

    fig = plt.figure(figsize=(20, sum(source_heights) + 1.8))
    fig.suptitle(
        f"{display_name}  (taxon {tid})",
        fontsize=14,
        fontweight="bold",
    )

    image_columns = min(IMAGES_PER_SUBROW, max(1, n_images))
    width_ratios = [1.1, 1.7] + [1.0] * image_columns + [7.0]
    outer = gridspec.GridSpec(
        9,
        image_columns + 3,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=[
            0.45,
            source_heights[0] / 4,
            source_heights[0] / 4,
            source_heights[0] / 4,
            source_heights[0] / 4,
            source_heights[1] / 4,
            source_heights[1] / 4,
            source_heights[1] / 4,
            source_heights[1] / 4,
        ],
        wspace=0.04,
        hspace=0.12,
    )

    _plot_column_header(fig.add_subplot(outer[0, 0]), "Source")
    _plot_column_header(fig.add_subplot(outer[0, 1]), "Group")
    _plot_column_header(fig.add_subplot(outer[0, 2 : 2 + image_columns]), "Example images")
    _plot_column_header(fig.add_subplot(outer[0, -1]), "VisDiff descriptions")

    row_specs = [
        (1, naip_label, "#1565C0", naip_pos_rows, naip_neg_rows, naip_body),
        (5, sentinel_label, "#00796B", sent_pos_rows, sent_neg_rows, sent_body),
    ]
    for row_idx, label, row_color, pos_rows, neg_rows, body in row_specs:
        _plot_row_label(fig.add_subplot(outer[row_idx : row_idx + 4, 0]), label, row_color)
        _plot_group_label(
            fig.add_subplot(outer[row_idx : row_idx + 2, 1]),
            "Likely\noccupied",
            "#2196F3",
        )
        _plot_group_label(
            fig.add_subplot(outer[row_idx + 2 : row_idx + 4, 1]),
            "Likely\nunoccupied",
            "#F44336",
        )
        for subrow_idx, row_paths in enumerate(pos_rows[:IMAGE_SUBROWS]):
            for col_idx, path in enumerate(row_paths[:image_columns]):
                _plot_thumb(
                    fig.add_subplot(outer[row_idx + subrow_idx, 2 + col_idx]),
                    path,
                    "#2196F3",
                )
        for subrow_idx, row_paths in enumerate(neg_rows[:IMAGE_SUBROWS]):
            for col_idx, path in enumerate(row_paths[:image_columns]):
                _plot_thumb(
                    fig.add_subplot(outer[row_idx + 2 + subrow_idx, 2 + col_idx]),
                    path,
                    "#F44336",
                )
        _plot_description_panel(
            fig.add_subplot(outer[row_idx : row_idx + 4, -1]),
            f"{label} top {n_hypotheses} VisDiff descriptions",
            body,
        )

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
    min_dist_km: float = 0.0,
    dpi: int = 150,
    overwrite: bool = False,
) -> None:
    """Render per-species NAIP vs Sentinel comparison figures and save to *output_dir*."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    naip_csv = (
        Path(naip_csv)
        if naip_csv is not None
        else DEFAULT_VISDIFF_CSVS.get(naip_imagery_source, DEFAULT_NAIP_CSV)
    )
    sentinel_csv = (
        Path(sentinel_csv)
        if sentinel_csv is not None
        else DEFAULT_VISDIFF_CSVS.get(sentinel_imagery_source, DEFAULT_SENTINEL_CSV)
    )
    naip_png_dir = Path(naip_png_dir) if naip_png_dir is not None else None
    sentinel_png_dir = Path(sentinel_png_dir) if sentinel_png_dir is not None else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mod_list = [m.strip() for m in modalities.split(",") if m.strip()]

    LOGGER.info("Loading NAIP CSV: %s", naip_csv)
    LOGGER.info("Loading Sentinel CSV: %s", sentinel_csv)
    LOGGER.info("NAIP imagery source: %s", naip_imagery_source)
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

    all_ids = set(naip_visdiff["taxon_id"].unique()) | set(
        sentinel_visdiff["taxon_id"].unique()
    )
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
