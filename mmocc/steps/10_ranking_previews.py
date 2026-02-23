#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "setuptools",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Generate qualitative HTML previews of satellite imagery rankings from multimodal
occupancy model results."""

import base64
import html
import logging
import math
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import fire
import pandas as pd
from PIL import Image

from mmocc.config import (
    cache_path,
    default_image_backbone,
    default_sat_backbone,
)
from mmocc.interpretability_utils import (
    compute_site_scores,
    load_fit_results,
    load_sat_lookup,
    rank_sat_groups,
    resolve_fit_results_path,
)
from mmocc.utils import (
    get_focal_species_ids,
    get_submitit_executor,
    get_taxon_map,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_MODALITY = "sat,covariates"
ALLOWED_MODES = {"standard", "unique"}
DEFAULT_MODES = ("standard", "unique")
OUTPUT_DIR = cache_path / "html_previews"
SENTINEL_LAYER = "s2cloudless-2020_3857"
SENTINEL_TILE_SIZE = 256
SENTINEL_PIXEL_SIZE_METERS = 10.0
PLACEHOLDER_PIXEL = "data:image/gif;base64,R0lGODlhAQABAAD/ACw="

CATEGORY_DESCRIPTIONS = {
    ("standard", "positive"): (
        "Most confidently occupied",
        "These are the satellite images the model rates as most likely to be occupied by the species of interest, irrespective of other predictors like environmental covariates.",
    ),
    ("standard", "negative"): (
        "Most confidently NOT occupied",
        "These are the satellite images the model rates as most likely to NOT be occupied by the species of interest, irrespective of other predictors like environmental covariates.",
    ),
    ("unique", "positive"): (
        "Most confidently occupied (maximum disagreement with covariates)",
        "These are the satellite images the model rates as most likely to be occupied, but where other predictors such as environmental covariates would suggest otherwise.",
    ),
    ("unique", "negative"): (
        "Most confidently NOT occupied (maximum disagreement with covariates)",
        "These are the satellite images the model rates as least likely to be occupied, but where other predictors such as environmental covariates would suggest otherwise.",
    ),
}


def _ensure_list(value, default: Iterable[str]) -> List[str]:
    if value is None:
        value = list(default)
    if isinstance(value, str):
        return [value]
    return list(value)


def _latlon_to_web_mercator(lat: float, lon: float) -> Tuple[float, float]:
    origin_shift = 20037508.342789244
    x = lon * origin_shift / 180.0
    lat_clamped = max(min(lat, 89.9), -89.9)
    y = (
        math.log(math.tan((90 + lat_clamped) * math.pi / 360.0))
        * origin_shift
        / math.pi
    )
    return x, y


def build_s2cloudless_tile_url(
    latitude: float | None,
    longitude: float | None,
    size: int = SENTINEL_TILE_SIZE,
    pixel_size_meters: float = SENTINEL_PIXEL_SIZE_METERS,
    layer: str = SENTINEL_LAYER,
) -> str:
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return ""
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return ""
    center_x, center_y = _latlon_to_web_mercator(lat, lon)
    half_width = (size * pixel_size_meters) / 2.0
    half_height = (size * pixel_size_meters) / 2.0
    bbox = (
        f"{center_x - half_width},{center_y - half_height},"
        f"{center_x + half_width},{center_y + half_height}"
    )
    params = dict(
        SERVICE="WMS",
        REQUEST="GetMap",
        VERSION="1.1.1",
        LAYERS=layer,
        STYLES="",
        FORMAT="image/jpeg",
        SRS="EPSG:3857",
        WIDTH=size,
        HEIGHT=size,
        BBOX=bbox,
    )
    query = "&".join(f"{key}={value}" for key, value in params.items())
    return f"https://tiles.maps.eox.at/wms?{query}"


def build_external_link(latitude: float | None, longitude: float | None) -> str:
    try:
        lat = float(latitude)  # type: ignore
        lon = float(longitude)  # type: ignore
    except (TypeError, ValueError):
        return ""
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return ""
    return f"https://www.google.com/maps/place/{lat:.6f},{lon:.6f}"


def parse_modalities(value: str) -> List[str]:
    return [modality.strip() for modality in value.split(",") if modality.strip()]


def normalize_modalities_arg(value: Sequence[str] | str | None) -> List[List[str]]:
    modality_specs = _ensure_list(value, [DEFAULT_MODALITY])
    modalities = [parse_modalities(spec) for spec in modality_specs]
    for spec in modalities:
        if "sat" not in spec:
            raise ValueError("Each modality set must include the 'sat' modality.")
    return modalities


def normalize_backbones(
    value: Sequence[str] | str | None, default_value: str
) -> List[str]:
    return _ensure_list(value, [default_value])


def normalize_modes(value: Sequence[str] | str | None) -> List[str]:
    modes = _ensure_list(value, list(DEFAULT_MODES))
    invalid = [mode for mode in modes if mode not in ALLOWED_MODES]
    if invalid:
        raise ValueError(
            f"Unsupported modes: {invalid}. Allowed modes: {sorted(ALLOWED_MODES)}"
        )
    return modes


def normalize_species_ids(value: Sequence[str] | str | None) -> List[str]:
    species = _ensure_list(value, get_focal_species_ids())
    if not species:
        raise ValueError("No species IDs provided via arguments or focal list.")
    return species


def sat_to_data_uri(path: str) -> str:
    """Convert satellite image to data URI for embedding in HTML."""
    if not path:
        return ""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            with BytesIO() as buffer:
                img.save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return f"data:image/png;base64,{encoded}"
    except Exception as exc:
        LOGGER.warning("Failed to encode satellite image %s: %s", path, exc)
        return ""


def create_sat_grid_html(
    df: pd.DataFrame, title: str, description: str | None = None
) -> str:
    """Create HTML grid for satellite imagery."""
    if df.empty:
        return ""

    description_html = f"<p>{html.escape(description)}</p>" if description else ""

    tiles: List[str] = []
    for _, row in df.iterrows():
        # Use local satellite image if available, otherwise use web service
        sat_path = row.get("sat_path", "")
        if sat_path and Path(sat_path).exists():
            data_uri = sat_to_data_uri(str(sat_path))
        else:
            data_uri = ""
        
        caption = row.get("loc_id") or row.get("site_id") or ""
        caption = html.escape(str(caption)) or "Satellite image"
        
        # Build S2 cloudless tile as fallback/additional view
        s2_url = build_s2cloudless_tile_url(row.get("Latitude"), row.get("Longitude"))
        external_link = build_external_link(row.get("Latitude"), row.get("Longitude"))
        
        sat_alt = (
            f"Satellite imagery near {caption}"
            if caption.strip()
            else "Satellite imagery"
        )
        
        # Primary satellite image (local or S2 cloudless)
        primary_src = data_uri if data_uri else s2_url
        if not primary_src:
            continue
            
        sat_img_tag = (
            f'<img class="sat-image lazy-img" src="{PLACEHOLDER_PIXEL}" '
            f'data-src="{html.escape(primary_src, quote=True)}" alt="{sat_alt}" '
            f'title="{sat_alt}" loading="lazy" decoding="async" />'
        )
        if external_link:
            sat_img_tag = (
                f'<a href="{html.escape(external_link, quote=True)}" target="_blank" '
                f'rel="noopener noreferrer">{sat_img_tag}</a>'
            )
        tiles.append(
            f'<figure class="sat-tile">{sat_img_tag}'
            f'<figcaption>{caption}</figcaption></figure>'
        )

    if not tiles:
        return ""

    return f"""
    <section class="image-section">
        <h2>{html.escape(title)}</h2>
        {description_html}
        <div class="image-grid">
            {' '.join(tiles)}
        </div>
    </section>
    """


def render_species_page(
    species_name: str,
    taxon_id: str,
    summary: Dict[str, str | int],
    sections: List[Tuple[str, str, pd.DataFrame]],
    output_path: Path,
):
    sections_html = []
    for mode, polarity, df in sections:
        title, description = CATEGORY_DESCRIPTIONS.get(
            (mode, polarity),
            (f"{mode.title()} ({polarity})", None),
        )
        html_block = create_sat_grid_html(df, title, description)
        if html_block:
            sections_html.append(html_block)

    if not sections_html:
        raise RuntimeError("No sections available to render.")

    summary_html = f"""
        <ul>
            <li><strong>Taxon ID:</strong> {html.escape(taxon_id)}</li>
            <li><strong>Modalities:</strong> {html.escape(str(summary['modalities']))}</li>
            <li><strong>Satellite backbone:</strong> {html.escape(str(summary['sat_backbone']))}</li>
            <li><strong>Unique test sites:</strong> {summary['unique_test_sites']}</li>
            <li><strong>Satellite images shown:</strong> {summary['num_images']}</li>
        </ul>
    """

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Satellite-based Habitat Differences for {html.escape(species_name)}</title>
    <style>
        :root {{ --img-scale: 1; }}
        body {{ font-family: sans-serif; margin: 2em; max-width: 1100px; }}
        h1 {{ margin-bottom: 0.25em; }}
        h2 {{ margin-top: 2em; }}
        .page-header {{ display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }}
        .zoom-control {{ display: flex; align-items: center; gap: 0.75rem; font-size: 0.95rem; flex-wrap: wrap; }}
        .zoom-value {{ font-variant-numeric: tabular-nums; min-width: 3ch; text-align: right; }}
        .image-grid {{ display: flex; flex-wrap: wrap; gap: 12px; padding: 10px; border: 1px solid #ccc; margin-bottom: 20px; }}
        .sat-tile {{ margin: 0; display: flex; flex-direction: column; align-items: center; gap: 4px; padding: 6px; border-radius: 6px; background: #f9f9f9; }}
        .sat-tile img {{ border: 1px solid #ddd; border-radius: 4px; background: #fff; }}
        .sat-tile figcaption {{ font-size: 0.8em; color: #666; max-width: calc(256px * var(--img-scale)); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .sat-image {{ width: calc(256px * var(--img-scale)); height: calc(256px * var(--img-scale)); object-fit: cover; }}
    </style>
</head>
<body>
    <div class="page-header">
        <div>
            <h1>{html.escape(species_name)}</h1>
            <p>Qualitative comparison of satellite imagery ranked by multimodal occupancy models.</p>
        </div>
        <div class="zoom-control">
            <label for="zoom-slider">Zoom:</label>
            <input type="range" id="zoom-slider" min="0.2" max="8" step="0.1" value="1" />
            <span id="zoom-value" class="zoom-value"></span>
        </div>
    </div>
    <section>
        {summary_html}
    </section>
    <p>
        These satellite images correspond to locations from the <a href="https://www.snapshot-usa.org/">Snapshot USA network</a> and are scored by our multimodal occupancy models.
        Images near the top of a group are the most representative for that hypothesis. We provide four categories:
        confident positives, confident negatives, and their counterparts that disagree with covariate-based modalities.
    </p>
    {''.join(sections_html)}
    <script>
        document.addEventListener("DOMContentLoaded", () => {{
            const slider = document.getElementById("zoom-slider");
            const valueLabel = document.getElementById("zoom-value");
            const updateScale = () => {{
                const scale = parseFloat(slider.value);
                document.documentElement.style.setProperty("--img-scale", scale.toFixed(2));
                valueLabel.textContent = `${{Math.round(scale * 100)}}%`;
            }};
            slider.addEventListener("input", updateScale);
            updateScale();

            const lazyImages = Array.from(document.querySelectorAll("img[data-src]"));
            const loadImage = (img) => {{
                const src = img.getAttribute("data-src");
                if (src) {{
                    img.src = src;
                    img.removeAttribute("data-src");
                }}
            }};
            if ("IntersectionObserver" in window) {{
                const observer = new IntersectionObserver((entries) => {{
                    entries.forEach((entry) => {{
                        if (entry.isIntersecting) {{
                            loadImage(entry.target);
                            observer.unobserve(entry.target);
                        }}
                    }});
                }}, {{ rootMargin: "100px" }});
                lazyImages.forEach((img) => observer.observe(img));
            }} else {{
                lazyImages.forEach(loadImage);
            }}
        }});
    </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    LOGGER.info("Wrote HTML preview to %s", output_path)
    return output_path


def write_index(entries: List[Dict[str, str]], output_dir: Path) -> Path:
    if not entries:
        raise RuntimeError("No HTML previews were generated.")

    rows = []
    for entry in sorted(entries, key=lambda item: item["species"].lower()):
        rows.append(
            f"<tr>"
            f"<td><a href=\"{html.escape(entry['filename'])}\">{html.escape(entry['species'])}</a></td>"
            f"<td>{html.escape(entry['modalities'])}</td>"
            f"<td>{html.escape(entry['sat_backbone'])}</td>"
            f"<td>{entry['unique_test_sites']}</td>"
            f"<td>{entry['num_images']}</td>"
            f"</tr>"
        )

    rows_html = "\n".join(rows)
    index_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Satellite Imagery Ranking Previews</title>
    <style>
        body {{ font-family: sans-serif; margin: 2em; }}
        table {{ border-collapse: collapse; width: 100%; max-width: 1100px; }}
        th, td {{ border: 1px solid #ccc; padding: 0.6rem 0.75rem; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        tbody tr:nth-child(even) {{ background-color: #fafafa; }}
    </style>
</head>
<body>
    <h1>Qualitative Satellite Ranking Previews</h1>
    <p>Each row links to a qualitative gallery summarizing the satellite imagery used in VisDiff analyses.</p>
    <table>
        <thead>
            <tr>
                <th>Species</th>
                <th>Modalities</th>
                <th>Satellite Backbone</th>
                <th>Unique Test Sites</th>
                <th>Satellite Images</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</body>
</html>
"""

    output_path = output_dir / "index.html"
    output_path.write_text(index_html, encoding="utf-8")
    LOGGER.info("Wrote HTML index to %s", output_path)
    return output_path


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "species"


def generate_html_previews(
    species_ids: Sequence[str] | str | None = None,
    modalities: Sequence[str] | str | None = None,
    sat_backbones: Sequence[str] | str | None = None,
    modes: Sequence[str] | str | None = None,
    top_k: int = 20,
    unique_weight: float = 2.0,
    output_dir: str | Path = OUTPUT_DIR,
) -> Path:
    species_list = normalize_species_ids(species_ids)
    modality_sets = normalize_modalities_arg(modalities)
    sat_backbone_list = normalize_backbones(sat_backbones, default_sat_backbone)
    modes_list = normalize_modes(modes)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sat_lookup = load_sat_lookup()
    taxon_map = get_taxon_map()
    entries: List[Dict[str, str]] = []

    for taxon_id in species_list:
        species_name = taxon_map.get(taxon_id, taxon_id)
        for modalities_set in modality_sets:
            for sat_backbone in sat_backbone_list:
                try:
                    (
                        fit_path,
                        resolved_modalities,
                        resolved_image_backbone,
                        resolved_sat_backbone,
                    ) = resolve_fit_results_path(
                        taxon_id,
                        modalities_set,
                        None,
                        sat_backbone,
                    )
                except FileNotFoundError as exc:
                    LOGGER.warning(
                        "Skipping %s (%s): %s", species_name, taxon_id, exc
                    )
                    continue

                fit_results = load_fit_results(fit_path)
                site_scores, display_name = compute_site_scores(
                    taxon_id,
                    resolved_modalities,
                    resolved_image_backbone,
                    resolved_sat_backbone,
                    fit_results,
                )
                site_scores = site_scores.join(
                    sat_lookup, on="loc_id", how="left"
                )
                site_scores["sat_exists"] = (
                    site_scores["sat_exists"].fillna(False).astype(bool)
                )

                sections: List[Tuple[str, str, pd.DataFrame]] = []
                for mode in modes_list:
                    positives, negatives = rank_sat_groups(
                        site_scores,
                        resolved_modalities,
                        mode,
                        unique_weight,
                        top_k,
                        test=True,
                    )
                    if not positives.empty:
                        sections.append((mode, "positive", positives))
                    if not negatives.empty:
                        sections.append((mode, "negative", negatives))

                if not sections:
                    LOGGER.warning(
                        "Insufficient satellite imagery to build previews for %s (%s).",
                        display_name,
                        taxon_id,
                    )
                    continue

                unique_sites = int(
                    site_scores.loc[site_scores["is_test"], "loc_id"]
                    .dropna()
                    .nunique()
                )
                num_images = sum(len(df) for _, _, df in sections)

                summary = dict(
                    modalities=", ".join(resolved_modalities),
                    sat_backbone=resolved_sat_backbone or default_sat_backbone,
                    unique_test_sites=unique_sites,
                    num_images=num_images,
                )

                slug = slugify(display_name)
                filename = (
                    f"{slug}_{resolved_sat_backbone or 'default'}_"
                    f"{'-'.join(sorted(resolved_modalities))}.html"
                )
                output_path = output_dir / filename
                try:
                    render_species_page(
                        display_name,
                        taxon_id,
                        summary,
                        sections,
                        output_path,
                    )
                except RuntimeError as exc:
                    LOGGER.warning(
                        "Skipping HTML page for %s (%s): %s",
                        display_name,
                        taxon_id,
                        exc,
                    )
                    continue

                entries.append(
                    dict(
                        species=display_name,
                        filename=filename,
                        modalities=str(summary["modalities"]),
                        sat_backbone=str(summary["sat_backbone"]),
                        unique_test_sites=str(summary["unique_test_sites"]),
                        num_images=str(summary["num_images"]),
                    )
                )

    if not entries:
        raise RuntimeError("No HTML previews were generated for any species.")

    write_index(entries, output_dir)
    return output_dir


def main(
    species_ids: Sequence[str] | str | None = None,
    modalities: Sequence[str] | str | None = None,
    sat_backbones: Sequence[str] | str | None = None,
    modes: Sequence[str] | str | None = None,
    top_k: int = 20,
    unique_weight: float = 2.0,
    output_dir: str | Path = OUTPUT_DIR,
):
    executor = get_submitit_executor("ranking_previews")
    executor.update_parameters(
        slurm_mem="32G",
        cpus_per_task=4,
        slurm_additional_parameters=dict(gpus=0),
    )
    job = executor.submit(
        generate_html_previews,
        species_ids,
        modalities,
        sat_backbones,
        modes,
        top_k,
        unique_weight,
        output_dir,
    )
    print(job.result())


if __name__ == "__main__":
    fire.Fire(main)
