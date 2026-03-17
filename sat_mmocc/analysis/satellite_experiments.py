#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "pandas",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Shared helpers for analysis of satellite-specific experiment artifacts.

These utilities mirror the cache naming conventions used by:
  - step 08: VisDiff descriptor generation
  - step 16: GRAFT satellite feature extraction
  - step 17: GRAFT descriptor refits

Keeping the mappings here avoids re-encoding the same Sentinel/NAIP logic in
each analysis script or notebook.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from sat_mmocc.config import cache_path, default_image_backbone, default_sat_backbone
from sat_mmocc.utils import experiment_to_filename, get_focal_species_ids

DEFAULT_IMAGERY_SOURCE = "sentinel"
VALID_DESCRIPTOR_SOURCES = frozenset({"visdiff", "expert"})

IMAGERY_SOURCE_PNG_DIRS: dict[str, Path] = {
    "sentinel": cache_path / "sat_wi_rgb_images_png",
    "naip": cache_path / "naip_wi_images_png",
}

IMAGERY_SOURCE_VISDIFF_FILES: dict[str, Path] = {
    "sentinel": cache_path / "visdiff_sat_sentinel2_wi_prompt2.csv",
    "naip": cache_path / "visdiff_sat_naip_wi_prompt2.csv",
}

IMAGERY_SOURCE_GRAFT_FEATURE_BACKBONES: dict[str, str] = {
    "sentinel": "graft",
    "naip": "graft_naip",
}

EXPERT_DESCRIPTOR_FILE = cache_path / "expert_habitat_descriptions.csv"


@dataclass(frozen=True)
class FitExperimentSpec:
    label: str
    modalities: tuple[str, ...] = ("covariates", "image", "sat")
    image_backbone: str | None = default_image_backbone
    sat_backbone: str | None = default_sat_backbone
    imagery_source: str | None = None
    descriptor_source: str | None = None


def normalize_imagery_source(imagery_source: str | None) -> str:
    value = (imagery_source or DEFAULT_IMAGERY_SOURCE).strip().lower()
    if value not in IMAGERY_SOURCE_VISDIFF_FILES:
        raise ValueError(
            f"Unknown imagery_source '{imagery_source}'. "
            f"Choose from: {sorted(IMAGERY_SOURCE_VISDIFF_FILES)}"
        )
    return value


def normalize_descriptor_source(source: str) -> str:
    value = source.strip().lower()
    if value not in VALID_DESCRIPTOR_SOURCES:
        raise ValueError(
            f"Unknown descriptor source '{source}'. "
            f"Choose from: {sorted(VALID_DESCRIPTOR_SOURCES)}"
        )
    return value


def parse_csv_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def load_species_ids(
    species_ids: str | Sequence[str] | None = None,
    species_ids_file: str | Path | None = None,
) -> list[str]:
    if species_ids is not None and species_ids_file is not None:
        raise ValueError("Provide either species_ids or species_ids_file, not both.")
    if species_ids_file is not None:
        text = Path(species_ids_file).read_text().strip()
        if not text:
            return []
        normalized = text.replace("\n", ",")
        return [token.strip() for token in normalized.split(",") if token.strip()]
    if species_ids is not None:
        return parse_csv_list(species_ids)
    return get_focal_species_ids()


def get_png_dir(imagery_source: str = DEFAULT_IMAGERY_SOURCE) -> Path:
    return IMAGERY_SOURCE_PNG_DIRS[normalize_imagery_source(imagery_source)]


def get_visdiff_descriptor_path(imagery_source: str = DEFAULT_IMAGERY_SOURCE) -> Path:
    return IMAGERY_SOURCE_VISDIFF_FILES[normalize_imagery_source(imagery_source)]


def get_descriptor_path(
    source: str,
    imagery_source: str = DEFAULT_IMAGERY_SOURCE,
    override: str | Path | None = None,
) -> Path:
    if override is not None:
        return Path(override)
    source = normalize_descriptor_source(source)
    if source == "expert":
        return EXPERT_DESCRIPTOR_FILE
    return get_visdiff_descriptor_path(imagery_source)


def get_graft_feature_backbone(imagery_source: str = DEFAULT_IMAGERY_SOURCE) -> str:
    return IMAGERY_SOURCE_GRAFT_FEATURE_BACKBONES[
        normalize_imagery_source(imagery_source)
    ]


def get_graft_refit_sat_backbone(
    descriptor_source: str,
    imagery_source: str = DEFAULT_IMAGERY_SOURCE,
) -> str:
    descriptor_source = normalize_descriptor_source(descriptor_source)
    if descriptor_source == "expert":
        return "graft_expert"
    imagery_source = normalize_imagery_source(imagery_source)
    return f"graft_visdiff_{imagery_source}"


def build_fit_results_path(
    taxon_id: str,
    modalities: Sequence[str],
    image_backbone: str | None,
    sat_backbone: str | None,
) -> Path:
    filename = experiment_to_filename(
        taxon_id, set(modalities), image_backbone, sat_backbone, "pkl"
    )
    return cache_path / "fit_results" / filename


def build_refit_combo_specs(
    imagery_source: str = DEFAULT_IMAGERY_SOURCE,
    include_baseline: bool = True,
    include_clip: bool = True,
    include_graft: bool = True,
    modalities: Sequence[str] = ("covariates", "image", "sat"),
) -> dict[str, FitExperimentSpec]:
    imagery_source = normalize_imagery_source(imagery_source)
    specs: dict[str, FitExperimentSpec] = {}
    if include_baseline:
        specs["full"] = FitExperimentSpec(
            label="full",
            modalities=tuple(modalities),
            image_backbone=default_image_backbone,
            sat_backbone=default_sat_backbone,
            imagery_source=imagery_source,
        )
    if include_clip:
        specs["clip_visdiff"] = FitExperimentSpec(
            label="clip_visdiff",
            modalities=tuple(modalities),
            image_backbone="visdiff_clip",
            sat_backbone=default_sat_backbone,
            imagery_source=imagery_source,
            descriptor_source="visdiff",
        )
        specs["clip_expert"] = FitExperimentSpec(
            label="clip_expert",
            modalities=tuple(modalities),
            image_backbone="expert_clip",
            sat_backbone=default_sat_backbone,
            imagery_source=imagery_source,
            descriptor_source="expert",
        )
    if include_graft:
        specs["graft_visdiff"] = FitExperimentSpec(
            label="graft_visdiff",
            modalities=tuple(modalities),
            image_backbone=default_image_backbone,
            sat_backbone=get_graft_refit_sat_backbone("visdiff", imagery_source),
            imagery_source=imagery_source,
            descriptor_source="visdiff",
        )
        specs["graft_expert"] = FitExperimentSpec(
            label="graft_expert",
            modalities=tuple(modalities),
            image_backbone=default_image_backbone,
            sat_backbone=get_graft_refit_sat_backbone("expert", imagery_source),
            imagery_source=imagery_source,
            descriptor_source="expert",
        )
    return specs


def load_fit_results_frame(
    experiment_specs: Mapping[str, FitExperimentSpec],
    species_ids: Sequence[str] | None = None,
    drop_mcmc_samples: bool = True,
    error_on_missing: bool = False,
) -> pd.DataFrame:
    species_ids = list(species_ids or get_focal_species_ids())
    rows: list[dict] = []
    for label, spec in experiment_specs.items():
        for taxon_id in species_ids:
            path = build_fit_results_path(
                taxon_id,
                spec.modalities,
                spec.image_backbone,
                spec.sat_backbone,
            )
            if not path.exists():
                if error_on_missing:
                    raise FileNotFoundError(f"Missing fit results at {path}")
                continue
            with path.open("rb") as handle:
                record = pickle.load(handle)
            if drop_mcmc_samples:
                record.pop("mcmc_samples", None)
            record.setdefault("experiment_label", label)
            record.setdefault("imagery_source", spec.imagery_source)
            record.setdefault("descriptor_source", spec.descriptor_source)
            rows.append(record)
    return pd.DataFrame(rows)


def build_refit_availability_table(
    experiment_specs: Mapping[str, FitExperimentSpec],
    species_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    species_ids = list(species_ids or get_focal_species_ids())
    rows: list[dict[str, object]] = []
    for label, spec in experiment_specs.items():
        existing = 0
        sample_path: Path | None = None
        for taxon_id in species_ids:
            path = build_fit_results_path(
                taxon_id,
                spec.modalities,
                spec.image_backbone,
                spec.sat_backbone,
            )
            if sample_path is None:
                sample_path = path
            if path.exists():
                existing += 1
        rows.append(
            {
                "experiment_label": label,
                "imagery_source": spec.imagery_source,
                "descriptor_source": spec.descriptor_source,
                "image_backbone": spec.image_backbone,
                "sat_backbone": spec.sat_backbone,
                "existing_species": existing,
                "requested_species": len(species_ids),
                "sample_fit_path": str(sample_path) if sample_path is not None else None,
            }
        )
    return pd.DataFrame(rows)


def main(
    imagery_source: str = DEFAULT_IMAGERY_SOURCE,
    species_ids: str | Sequence[str] | None = None,
    species_ids_file: str | Path | None = None,
    include_baseline: bool = True,
    include_clip: bool = True,
    include_graft: bool = True,
    load_fit_results: bool = False,
    drop_mcmc_samples: bool = True,
    error_on_missing: bool = False,
    output_csv: str | Path | None = None,
) -> None:
    selected_species = load_species_ids(species_ids, species_ids_file)
    specs = build_refit_combo_specs(
        imagery_source=imagery_source,
        include_baseline=include_baseline,
        include_clip=include_clip,
        include_graft=include_graft,
    )

    availability = build_refit_availability_table(specs, selected_species)
    print("Experiment availability:")
    print(availability.to_string(index=False))

    if not load_fit_results:
        return

    df = load_fit_results_frame(
        specs,
        species_ids=selected_species,
        drop_mcmc_samples=drop_mcmc_samples,
        error_on_missing=error_on_missing,
    )
    print(f"\nLoaded {len(df)} fit result rows.")
    if df.empty:
        return
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Wrote fit results table to {output_path}")
    else:
        preview = [
            "experiment_label",
            "taxon_id",
            "common_name",
            "image_backbone",
            "sat_backbone",
            "imagery_source",
            "descriptor_source",
        ]
        preview = [col for col in preview if col in df.columns]
        print("\nPreview:")
        print(df[preview].head().to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect satellite experiment combinations and optionally load fit results."
    )
    parser.add_argument(
        "--imagery-source",
        type=str,
        default=DEFAULT_IMAGERY_SOURCE,
        help="Imagery source to resolve: sentinel or naip.",
    )
    parser.add_argument(
        "--species-ids",
        type=str,
        default=None,
        help="Optional comma-separated taxon IDs.",
    )
    parser.add_argument(
        "--species-ids-file",
        type=Path,
        default=None,
        help="Optional text file containing comma- or newline-separated taxon IDs.",
    )
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the baseline full-model combo.",
    )
    parser.add_argument(
        "--include-clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the CLIP refit combos.",
    )
    parser.add_argument(
        "--include-graft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the GRAFT refit combos.",
    )
    parser.add_argument(
        "--load-fit-results",
        action="store_true",
        help="Load matching fit result pickles into a DataFrame.",
    )
    parser.add_argument(
        "--drop-mcmc-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop mcmc_samples when loading fit result records.",
    )
    parser.add_argument(
        "--error-on-missing",
        action="store_true",
        help="Raise if any requested fit result file is missing.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to write the loaded fit result table as CSV.",
    )
    args = parser.parse_args()
    main(
        imagery_source=args.imagery_source,
        species_ids=args.species_ids,
        species_ids_file=args.species_ids_file,
        include_baseline=args.include_baseline,
        include_clip=args.include_clip,
        include_graft=args.include_graft,
        load_fit_results=args.load_fit_results,
        drop_mcmc_samples=args.drop_mcmc_samples,
        error_on_missing=args.error_on_missing,
        output_csv=args.output_csv,
    )
