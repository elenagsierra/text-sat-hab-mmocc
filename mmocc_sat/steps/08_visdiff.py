#!/usr/bin/env -S uv run --python 3.11 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "pyvisdiff",
#     "setuptools",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# pyvisdiff = { git = "https://github.com/timmh/pyvisdiff.git" }
# ///
"""Run VisDiff to describe what makes environments where a species is present different
from where it is absent, using remote sensing satellite imagery."""

import logging
import math
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

import fire
import pandas as pd
import submitit

from mmocc_sat.config import (
    cache_path,
    default_image_backbone,
    default_sat_backbone,
    visdiff_model_name,
)
from mmocc_sat.interpretability_utils import (
    compute_site_scores,
    load_fit_results,
    load_sat_lookup,
    resolve_fit_results_path,
    select_sat_groups,
)
from mmocc_sat.utils import (
    experiment_to_filename,
    get_focal_species_ids,
    get_submitit_executor,
    get_taxon_map,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_MODALITY = "sat,covariates"
ALLOWED_MODES = {"standard", "unique"}
DEFAULT_MODES = ("standard", "unique")
DEFAULT_CACHE_DIR = cache_path / "visdiff_cache"
VISDIFF_DESCRIPTIONS_FILE = cache_path / "visdiff_sat_descriptions.csv"
DEFAULT_HYPOTHESES_LIMIT = None


def _should_stub_cv2(exc: BaseException) -> bool:
    message = str(exc)
    markers = (
        "numpy.core.multiarray failed to import",
        "module compiled against ABI version",
        "ImportError: numpy.core.multiarray failed to import",
        "cv2",
    )
    return any(marker in message for marker in markers)


def _purge_modules(prefixes: Sequence[str]) -> None:
    for name in list(sys.modules):
        for prefix in prefixes:
            if name == prefix or name.startswith(f"{prefix}."):
                sys.modules.pop(name, None)
                break


class _Cv2AttributeProxy:
    """Placeholder returned for every attribute on the cv2 stub module."""

    def __init__(self, name: str):
        self._name = name

    def _raise(self) -> None:
        raise RuntimeError(
            f"OpenCV operation '{self._name}' is unavailable: the provided wheels target "
            "NumPy 1.x while this step runs with NumPy 2.x. Install an opencv-python "
            "build compiled for NumPy 2.x to fully enable BLIP-based captioners."
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._raise()

    def __getattr__(self, item: str) -> "_Cv2AttributeProxy":
        return _Cv2AttributeProxy(f"{self._name}.{item}")

    def __int__(self) -> int:  # type: ignore[override]
        self._raise()

    def __float__(self) -> float:  # type: ignore[override]
        self._raise()

    def __repr__(self) -> str:
        return f"<cv2 stub attribute {self._name}>"


def _install_cv2_stub() -> None:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.__dict__["__version__"] = "0.0-stub"
    cv2_stub.__dict__["__doc__"] = (
        "OpenCV stub injected by mmocc/steps/07_visdiff.py to avoid numpy/cv2 ABI issues."
    )

    def _getattr(name: str) -> _Cv2AttributeProxy:
        if name.startswith("__"):
            raise AttributeError(name)
        return _Cv2AttributeProxy(name)

    cv2_stub.__getattr__ = _getattr  # type: ignore[assignment]
    sys.modules["cv2"] = cv2_stub


def _load_pyvisdiff_entrypoint():
    try:
        from pyvisdiff import run_visdiff as pyvisdiff_run

        return pyvisdiff_run
    except ImportError as exc:
        if not _should_stub_cv2(exc):
            raise
        LOGGER.warning(
            "pyvisdiff import failed because the bundled OpenCV wheels are incompatible "
            "with NumPy 2.x; injecting a cv2 stub instead. If BLIP actually exercises "
            "any OpenCV APIs the code will raise at runtime with a clearer error. %s",
            exc,
        )
        _purge_modules(("pyvisdiff", "serve", "lavis", "cv2"))
        _install_cv2_stub()
        from pyvisdiff import run_visdiff as pyvisdiff_run

        return pyvisdiff_run


def _ensure_list(value, default: List[str]) -> List[str]:
    if value is None:
        value = default
    if isinstance(value, str):
        return [value]
    return list(value)


def normalize_modalities_arg(value: Sequence[str] | str | None) -> List[List[str]]:
    modality_specs = _ensure_list(value, [DEFAULT_MODALITY])
    modalities = [parse_modalities(spec) for spec in modality_specs]
    for spec in modalities:
        if "sat" not in spec:
            raise ValueError(
                "Each modality set must include the 'sat' modality for VisDiff."
            )
    return modalities


def normalize_backbone_arg(
    value: Sequence[str] | str | None, default: str
) -> List[str]:
    return _ensure_list(value, [default])


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
        raise ValueError(
            "No species IDs provided via arguments or get_focal_species_ids()."
        )
    return species


def parse_modalities(value: str) -> List[str]:
    return [modality.strip() for modality in value.split(",") if modality.strip()]


def clean_hypothesis_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) >= 2 and (
        (text.startswith('"') and text.endswith('"'))
        or (text.startswith("'") and text.endswith("'"))
    ):
        text = text[1:-1].strip()
    return text


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_visdiff_rows(
    taxon_id: str,
    species_name: str,
    result: Dict,
    *,
    limit: int | None = DEFAULT_HYPOTHESES_LIMIT,
) -> List[Dict[str, str | float]]:
    ranked: Sequence[Dict[str, Any]] = result.get("ranked_hypotheses") or []
    rows: List[Dict[str, str | float]] = []
    if limit is not None:
        ranked = ranked[:limit]
    for entry in ranked:
        if not isinstance(entry, dict):
            continue
        difference = clean_hypothesis_text(entry.get("hypothesis"))
        if not difference:
            continue
        score = None
        scores = {}
        score_keys = ("auroc", "correct_delta", "diff", "score1", "score2", "t_stat")
        for key in score_keys:
            val = safe_float(entry.get(key))
            if score is not None and val is not None:
                score = val
            scores[key] = val
        if score is None:
            score = math.nan
        rows.append(
            {
                "taxon_id": taxon_id,
                "species": species_name,
                "difference": difference,
                "score": score,
                **scores,
            }
        )
    return rows


def aggregate_visdiff_rows(rows: Sequence[Dict[str, str | float]]) -> pd.DataFrame:
    minimum_columns = ["taxon_id", "species", "difference", "auroc"]
    if not rows:
        return pd.DataFrame(columns=minimum_columns)
    df = pd.DataFrame(rows)
    missing = [col for col in minimum_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in VisDiff rows: {missing}")
    df["auroc"] = pd.to_numeric(df["auroc"], errors="coerce")
    df = df.sort_values("auroc", ascending=False).drop_duplicates(
        subset=["taxon_id", "species", "difference"]
    )
    df = df.sort_values(["taxon_id", "auroc"], ascending=[True, False]).reset_index(
        drop=True
    )
    return df


def load_existing_visdiff_rows(
    output_file: Path | str,
) -> List[Dict[str, str | float]]:
    output_path = Path(output_file)
    if not output_path.exists():
        return []
    try:
        df = pd.read_csv(output_path)
    except Exception as exc:
        LOGGER.warning(
            "Failed to read existing VisDiff descriptions at %s; starting fresh. %s",
            output_path,
            exc,
        )
        return []
    required_columns = {"taxon_id", "species", "difference"}
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        LOGGER.warning(
            "Existing VisDiff descriptions at %s are missing %s; ignoring file.",
            output_path,
            missing,
        )
        return []
    return cast(List[Dict[str, str | float]], df.to_dict(orient="records"))


def write_visdiff_descriptions(
    rows: Sequence[Dict[str, str | float]], output_file: Path | str
) -> Path:
    output_path = Path(output_file)
    df = aggregate_visdiff_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Wrote %d VisDiff hypotheses to %s", len(df), output_path)
    return output_path


def run_pyvisdiff(
    species_name: str,
    positives: List[str],
    negatives: List[str],
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_dir: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = cache_path / "visdiff_cache",
) -> Dict:
    captioner_prompt = (
        "Describe this satellite imagery in detail. Focus on land cover, vegetation, "
        "terrain features, and environmental characteristics visible from above."
    )
    proposer_prompt = """
        The following are the result of captioning two groups of satellite images:

        {text}

        I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data.

        Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*"). For example:
        * "dense forest cover"
        * "agricultural fields"
        * "mountainous terrain"
        * "water bodies present"

        Focus on landscape, vegetation, terrain, and environmental features visible in satellite imagery. Do not talk about the caption, e.g., "caption with one word" and do not list more than one concept. The hypothesis should be a caption, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. Also do not enumerate possibilities within parentheses. Here are examples of bad outputs and their corrections:
        * INCORRECT: "various landscapes including forests, fields, and rivers" CORRECTED: "mixed landscapes"
        * INCORRECT: "presence of water bodies" CORRECTED: "water bodies"
        * INCORRECT: "images showing urban development" CORRECTED: "urban areas"
        * INCORRECT: "Different types of vegetation including trees, shrubs, and grass" CORRECTED: "diverse vegetation"

        Again, I want to figure out what kind of distribution shift are there. List properties that hold more often for the satellite images (not captions) in group A compared to group B. Answer with a list (separated by bullet points "*"). Your response:
    """

    pyvisdiff_run = _load_pyvisdiff_entrypoint()

    wandb_dir = (
        (wandb_dir or cache_path / "visdiff_wandb")
        if wandb_entity or wandb_project
        else None
    )

    return pyvisdiff_run(
        dataset_a_images=positives,
        dataset_b_images=negatives,
        dataset_a_description=f"Environments where {species_name} is likely to be present",
        dataset_b_description=f"Environments where {species_name} is likely to be absent",
        config_overrides={
            "captioner": {"prompt": captioner_prompt},
            "proposer": {"model": visdiff_model_name, "prompt": proposer_prompt},
            "ranker": {"model": visdiff_model_name},
            "evaluator": {"method": "NullEvaluator"},
        },
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_dir=wandb_dir,
        llm_api_key=os.getenv("OPENAI_API_KEY"),
        cache_dir=cache_dir,
    )


def run_species_visdiff_job(
    taxon_id: str,
    modalities: Sequence[str],
    image_backbone: str | None,
    sat_backbone: str | None,
    top_k: int,
    modes: Sequence[str],
    unique_weight: float,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    cache_dir: Path | str = cache_path / "visdiff_cache",
):
    logger = logging.getLogger(__name__)
    taxon_map = get_taxon_map()
    species_name = taxon_map.get(taxon_id, taxon_id)
    sat_lookup = load_sat_lookup()

    try:
        (
            fit_path,
            resolved_modalities,
            resolved_image_backbone,
            resolved_sat_backbone,
        ) = resolve_fit_results_path(taxon_id, modalities, image_backbone, sat_backbone)
    except FileNotFoundError as exc:
        logger.warning("Skipping %s (%s): %s", taxon_id, species_name, exc)
        return

    try:
        display_path = fit_path.relative_to(cache_path)
    except ValueError:
        display_path = fit_path
    logger.info(
        "Loading fit results for %s (%s) from %s",
        species_name,
        taxon_id,
        display_path,
    )

    fit_results = load_fit_results(fit_path)
    site_scores, display_name = compute_site_scores(
        taxon_id,
        resolved_modalities,
        resolved_image_backbone,
        resolved_sat_backbone,
        fit_results,
    )
    site_scores = site_scores.join(sat_lookup, on="loc_id", how="left")
    site_scores["sat_exists"] = site_scores["sat_exists"].fillna(False).astype(bool)

    available_sat = int(
        (site_scores["sat_exists"] & site_scores["is_train"]).sum()
    )

    collected_rows: List[Dict[str, str | float]] = []

    for mode in modes:
        positives, negatives = select_sat_groups(
            site_scores,
            resolved_modalities,
            mode,
            unique_weight,
            top_k,
            test=False,
        )
        if not positives or not negatives:
            logger.warning(
                "Insufficient satellite imagery for %s (%s mode). Need %d valid images but only found %d.",
                species_name,
                mode,
                top_k,
                available_sat,
            )
            continue

        logger.info(
            "Running VisDiff for %s (%s mode) with %d positives / %d negatives.",
            display_name,
            mode,
            len(positives),
            len(negatives),
        )

        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)

        cache_dir_visdiff = cache_dir / "visdiff"
        wandb_cache_dir = cache_dir / "wandb"
        cache_dir_visdiff.mkdir(parents=True, exist_ok=True)
        wandb_cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = run_pyvisdiff(
                display_name,
                positives,
                negatives,
                wandb_entity,
                wandb_project,
                wandb_cache_dir,
                cache_dir_visdiff,
            )
        except Exception as exc:
            logger.error("VisDiff failed for %s (%s mode): %s", display_name, mode, exc)
            raise

        logger.info("VisDiff result for %s (%s): %s", display_name, mode, result)
        collected_rows.extend(
            build_visdiff_rows(
                taxon_id, display_name, result, limit=DEFAULT_HYPOTHESES_LIMIT
            )
        )

    return collected_rows


def main(
    modalities: Sequence[str] | str | None = None,
    image_backbones: Sequence[str] | str | None = None,
    sat_backbones: Sequence[str] | str | None = None,
    species_ids: Sequence[str] | str | None = None,
    top_k: int = 50,
    modes: Sequence[str] | str | None = None,
    unique_weight: float = 2.0,
    wandb_entity: str | None = os.getenv("VISDIFF_WANDB_ENTITY"),
    wandb_project: str | None = os.getenv("VISDIFF_WANDB_PROJECT"),
    cache_dir: str | None = None,
    output_file: str | Path = VISDIFF_DESCRIPTIONS_FILE,
):

    modality_sets = normalize_modalities_arg(modalities)
    image_backbone_list = normalize_backbone_arg(
        image_backbones, default_image_backbone
    )
    sat_backbone_list = normalize_backbone_arg(sat_backbones, default_sat_backbone)
    species_list = normalize_species_ids(species_ids)
    modes_list = normalize_modes(modes)
    cache_dir = cache_dir or str(DEFAULT_CACHE_DIR)
    output_path = Path(output_file)

    existing_rows = load_existing_visdiff_rows(output_path)
    existing_species = {
        str(row["taxon_id"])
        for row in existing_rows
        if isinstance(row, dict)
        and "taxon_id" in row
        and pd.notna(row["taxon_id"])
        and str(row["taxon_id"]).strip()
    }
    if existing_species:
        initial_count = len(species_list)
        species_list = [
            taxon_id for taxon_id in species_list if taxon_id not in existing_species
        ]
        skipped = initial_count - len(species_list)
        if skipped:
            LOGGER.info(
                "Skipping %d species already present in %s",
                skipped,
                output_path,
            )
    if not species_list:
        if existing_rows:
            LOGGER.info(
                "All requested species already have VisDiff descriptions in %s; no jobs scheduled.",
                output_path,
            )
        else:
            LOGGER.warning("No species selected for VisDiff; exiting.")
        return

    executor = get_submitit_executor("visdiff")
    executor.update_parameters(slurm_additional_parameters=dict(gpus=1))

    jobs = []
    with executor.batch():
        for modalities_set in modality_sets:
            for image_backbone in image_backbone_list:
                for sat_backbone in sat_backbone_list:
                    for taxon_id in species_list:

                        experiment_dir = experiment_to_filename(
                            taxon_id,
                            modalities_set,
                            image_backbone,
                            sat_backbone,
                        )
                        experiment_cache_dir = Path(cache_dir) / experiment_dir
                        experiment_cache_dir.mkdir(parents=True, exist_ok=True)

                        job = executor.submit(
                            run_species_visdiff_job,
                            taxon_id,
                            modalities_set,
                            image_backbone,
                            sat_backbone,
                            top_k,
                            modes_list,
                            unique_weight,
                            wandb_entity,
                            wandb_project,
                            experiment_cache_dir,
                        )
                        jobs.append(job)

    new_rows: List[Dict[str, str | float]] = []
    for job in submitit.helpers.as_completed(jobs):
        try:
            result_rows = job.result()
        except Exception as exc:
            LOGGER.error("VisDiff job %s failed: %s", job.job_id, exc, exc_info=True)
            continue
        if result_rows:
            new_rows.extend(result_rows)

    if not new_rows:
        if existing_rows:
            LOGGER.info(
                "No new VisDiff hypotheses produced; keeping existing file at %s",
                output_path,
            )
        else:
            LOGGER.warning(
                "VisDiff completed but produced no ranked hypotheses; skipping CSV write."
            )
        return

    write_visdiff_descriptions(existing_rows + new_rows, output_path)


if __name__ == "__main__":
    fire.Fire(main)
