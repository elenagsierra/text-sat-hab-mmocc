#!/usr/bin/env -S uv run --python 3.11 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "pyvisdiff",
#     "setuptools<81"
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# pyvisdiff = { git = "https://github.com/timmh/pyvisdiff.git" }
# ///
"""Run VisDiff to describe what makes environments where a species is present different
from where it is absent."""

import logging
import math
import os
import re
import sys
import types
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

import fire
import pandas as pd
import submitit

from sat_mmocc.config import (
    cache_path,
    default_image_backbone,
    default_sat_backbone,
    visdiff_model_name,
)
from sat_mmocc.imagery_lookups import load_imagery_lookup
from sat_mmocc.interpretability_utils import (
    compute_site_scores,
    load_fit_results,
    resolve_fit_results_path,
    select_image_groups,
)
from sat_mmocc.utils import (
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
VISDIFF_DESCRIPTIONS_FILE = cache_path / "visdiff_sat_sentinel2_wi_prompt4.csv"
VISDIFF_NAIP_DESCRIPTIONS_FILE = cache_path / "visdiff_sat_naip_wi_prompt4.csv"
IMAGERY_SOURCE_OUTPUT_FILES = {
    "sentinel": VISDIFF_DESCRIPTIONS_FILE,
    "naip": VISDIFF_NAIP_DESCRIPTIONS_FILE,
    "sentinel_v_graft": cache_path / "visdiff_sentinel_v_graft_descriptions_p4.csv",
    "naip_v_graft": cache_path / "visdiff_naip_v_graft_descriptions_p4.csv",
}
DEFAULT_HYPOTHESES_LIMIT = None
MAX_HYPOTHESES_PER_RUN = 8
MAX_HYPOTHESES_PER_SPECIES = 24
GENERIC_DIFFERENCE_TOKENS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "to", "from", "with",
    "within", "through", "around", "into", "across", "between", "along", "by",
    "visible", "broad", "large", "small", "narrow", "wide", "dark", "light", "green",
    "brown", "gray", "grey", "mixed", "mosaic", "mottled", "patchy", "scattered",
    "continuous", "dense", "sparse", "closed", "open", "unbroken", "otherwise",
    "otherwise", "dominated", "style", "like", "forming", "creating", "cover", "texture",
    "surface", "tones", "tone", "areas", "area", "view", "visible", "running", "cutting",
    "meeting", "stretching", "bordered", "bordering", "dotted", "same", "little", "few",
    "more", "less", "minimal", "gently", "uneven", "flat", "tall", "low", "high",
    "height", "layered", "tightly", "packed", "curving", "winding", "alternating",
    "uniform", "interrupted", "uninterrupted", "continuous", "closed-canopy", "mixed-height",
}
TOKEN_SYNONYMS = {
    "woods": "forest", "woodland": "forest", "wooded": "forest", "tree": "forest",
    "trees": "forest", "treetops": "forest", "crowns": "forest", "crown": "forest",
    "canopy": "forest", "stand": "forest", "stands": "forest", "understory": "forest",
    "coniferous": "conifer", "needleleaf": "conifer", "evergreen": "conifer",
    "broadleaf-dominated": "broadleaf", "deciduous": "broadleaf",
    "river": "stream", "creek": "stream", "watercourse": "stream", "channel": "stream",
    "channels": "stream", "streams": "stream", "meandering": "stream",
    "ponds": "pond", "lakes": "pond", "lakeshore": "pond", "shoreline": "pond",
    "wetlands": "wetland", "marsh": "wetland", "marshy": "wetland",
    "field": "agriculture", "fields": "agriculture", "cropland": "agriculture",
    "crop": "agriculture", "crops": "agriculture", "agricultural": "agriculture",
    "cultivated": "agriculture", "plowed": "agriculture", "harvested": "agriculture",
    "irrigated": "agriculture", "pasture": "grass", "prairie": "grass", "meadow": "grass",
    "grassland": "grass", "grassy": "grass", "grasses": "grass",
    "shrubs": "shrub", "shrubland": "shrub", "bushes": "shrub",
    "sandy": "sand", "soil": "bare", "earth": "bare", "barren": "bare",
    "road": "road", "track": "road", "path": "road", "trail": "road",
}


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
        "OpenCV stub injected by mmocc/steps/08_visdiff.py to avoid numpy/cv2 ABI issues."
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


def score_hypothesis_entry(entry: Dict[str, Any]) -> float:
    for key in ("auroc", "correct_delta", "diff", "score1", "score2", "t_stat"):
        val = safe_float(entry.get(key))
        if val is not None and not math.isnan(val):
            return val
    return math.nan


def _canonicalize_token(token: str) -> str:
    token = TOKEN_SYNONYMS.get(token, token)
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        token = token[:-1]
    token = TOKEN_SYNONYMS.get(token, token)
    return token


def normalize_difference_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[()]", " ", text)
    text = text.replace("/", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = []
    for raw_token in text.split():
        token = _canonicalize_token(raw_token)
        if len(token) <= 1 or token in GENERIC_DIFFERENCE_TOKENS:
            continue
        tokens.append(token)
    return " ".join(tokens)


def semantic_signature(text: str) -> tuple[str, ...]:
    normalized = normalize_difference_text(text)
    if not normalized:
        return tuple()
    tokens = []
    seen = set()
    for token in normalized.split():
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tuple(sorted(tokens))


def is_near_duplicate(candidate: str, kept: Sequence[str]) -> bool:
    candidate_norm = normalize_difference_text(candidate)
    candidate_sig = set(semantic_signature(candidate))
    if not candidate_norm:
        return True

    for existing in kept:
        existing_norm = normalize_difference_text(existing)
        existing_sig = set(semantic_signature(existing))

        if candidate_norm == existing_norm:
            return True

        if candidate_sig and existing_sig:
            overlap = len(candidate_sig & existing_sig)
            union = len(candidate_sig | existing_sig)
            jaccard = overlap / union if union else 0.0

            if candidate_sig == existing_sig:
                return True
            if overlap >= 2 and jaccard >= 0.67:
                return True
            if min(len(candidate_sig), len(existing_sig)) == 1 and jaccard == 1.0:
                return True

        seq_ratio = SequenceMatcher(None, candidate_norm, existing_norm).ratio()
        if seq_ratio >= 0.86:
            return True

    return False


def dedupe_ranked_rows(
    rows: Sequence[Dict[str, str | float]], *, max_rows: int | None = None
) -> List[Dict[str, str | float]]:
    kept: List[Dict[str, str | float]] = []
    kept_text: List[str] = []

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -safe_float(row.get("auroc")) if safe_float(row.get("auroc")) is not None else float("inf")
        ),
    )

    for row in sorted_rows:
        difference = str(row.get("difference", "")).strip()
        if not difference or is_near_duplicate(difference, kept_text):
            continue
        kept.append(row)
        kept_text.append(difference)
        if max_rows is not None and len(kept) >= max_rows:
            break

    return kept


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
        scores = {}
        score_keys = ("auroc", "correct_delta", "diff", "score1", "score2", "t_stat")
        for key in score_keys:
            scores[key] = safe_float(entry.get(key))
        rows.append(
            {
                "taxon_id": taxon_id,
                "species": species_name,
                "difference": difference,
                "score": score_hypothesis_entry(entry),
                **scores,
            }
        )
    return dedupe_ranked_rows(rows, max_rows=MAX_HYPOTHESES_PER_RUN)


def aggregate_visdiff_rows(rows: Sequence[Dict[str, str | float]]) -> pd.DataFrame:
    minimum_columns = ["taxon_id", "species", "difference", "auroc"]
    if not rows:
        return pd.DataFrame(columns=minimum_columns)
    df = pd.DataFrame(rows)
    missing = [col for col in minimum_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in VisDiff rows: {missing}")
    df["auroc"] = pd.to_numeric(df["auroc"], errors="coerce")
    df = df.sort_values(["taxon_id", "auroc"], ascending=[True, False]).reset_index(drop=True)

    deduped_rows: List[Dict[str, Any]] = []
    for _, group in df.groupby("taxon_id", sort=False):
        group_rows = cast(List[Dict[str, Any]], group.to_dict(orient="records"))
        deduped_rows.extend(
            dedupe_ranked_rows(group_rows, max_rows=MAX_HYPOTHESES_PER_SPECIES)
        )

    out = pd.DataFrame(deduped_rows)
    out = out.sort_values(["taxon_id", "auroc"], ascending=[True, False]).reset_index(drop=True)
    return out


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
    captioner_prompt = """
    Describe only the visible habitat features in this image at the local site scale.

    Focus on land cover, vegetation structure, hydrology, exposed substrate, habitat edges, patch mosaics, fragmentation, disturbance, and management patterns.

    Use specific observable feature phrases. Do not mention image source, sensor, resolution, Earth, space, weather, location, region, biome, ecoregion, or any proper noun. Do not guess where the scene is. Replace named landscapes with generic visual descriptions.

    Example: say "sand dunes with sparse vegetation", not "Sahara desert landscape".
    """
    proposer_prompt = """
    The following text contains captions for two groups of habitat images:

    {text}

    List 10 concepts more likely in Group A than Group B.

    Each concept must be a short noun phrase describing one visible habitat feature. The 10 bullets must be materially different from one another. Do not restate the same idea with minor wording changes, especially for repeated forest, water, agriculture, or open-ground patterns. Prefer one specific phrase over several paraphrases of the same feature. Use only generic observable descriptors. Do not mention modality, image quality, proper nouns, place names, geographic labels, biome labels, or inferred location. Rewrite any named place or regional term into a purely visual habitat phrase.

    Answer using bullet points starting with "*".
    """

    pyvisdiff_run = _load_pyvisdiff_entrypoint()

    wandb_dir = (
        (wandb_dir or cache_path / "visdiff_wandb")
        if wandb_entity or wandb_project
        else None
    )

    species_label = str(species_name).strip() or "species"
    dataset_a_label = f"present: {species_label}"[:48]
    dataset_b_label = f"absent: {species_label}"[:48]

    return pyvisdiff_run(
        dataset_a_images=positives,
        dataset_b_images=negatives,
        dataset_a_description=dataset_a_label,
        dataset_b_description=dataset_b_label,
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
    imagery_source: str = "sentinel",
):
    logger = logging.getLogger(__name__)
    taxon_map = get_taxon_map()
    species_name = taxon_map.get(taxon_id, taxon_id)

    # Match the convention used during training: backbone is None when its
    # modality is not in the requested set (mirrors 07_fit.py logic).
    lookup_image_backbone = image_backbone if "image" in modalities else None
    lookup_sat_backbone = sat_backbone if "sat" in modalities else None

    try:
        (
            fit_path,
            resolved_modalities,
            resolved_image_backbone,
            resolved_sat_backbone,
        ) = resolve_fit_results_path(taxon_id, modalities, lookup_image_backbone, lookup_sat_backbone)
    except FileNotFoundError as exc:
        logger.warning("Skipping %s (%s): %s", taxon_id, species_name, exc)
        return

    if "sat" not in resolved_modalities:
        raise RuntimeError(
            "VisDiff ranking for step 8 requires a fitted model with the 'sat' modality, "
            f"but {taxon_id} resolved to {fit_path.name} with modalities={sorted(resolved_modalities)}. "
            f"Requested modalities={sorted(modalities)} and sat_backbone={lookup_sat_backbone!r}. "
            "If you only want to swap which satellite PNGs are shown, keep "
            f"--imagery_source={imagery_source} but use a sat backbone that already has fit results. "
            "If you want to rank using this new satellite backbone, fit matching sat-enabled results first."
        )

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

    image_lookup = load_imagery_lookup(imagery_source)
    site_scores = site_scores.join(image_lookup, on="loc_id", how="left")
    site_scores["image_exists"] = site_scores["image_exists"].fillna(False).astype(bool)

    available_images = int(
        (site_scores["image_exists"] & site_scores["is_train"]).sum()
    )

    collected_rows: List[Dict[str, str | float]] = []

    for mode in modes:
        # UPDATE: pass image_modality="sat" to rank based on remote sensing features
        positives, negatives = select_image_groups(
            site_scores,
            resolved_modalities,
            mode,
            unique_weight,
            top_k,
            image_modality="sat", 
            test=False,
        )
        if not positives or not negatives:
            logger.warning(
                "Insufficient image coverage for %s (%s mode). Need %d valid images but only found %d.",
                species_name,
                mode,
                top_k,
                available_images,
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
    output_file: str | Path | None = None,
    imagery_source: str = "sentinel",
):

    modality_sets = normalize_modalities_arg(modalities)
    image_backbone_list = normalize_backbone_arg(
        image_backbones, default_image_backbone
    )
    sat_backbone_list = normalize_backbone_arg(sat_backbones, default_sat_backbone)
    species_list = normalize_species_ids(species_ids)
    modes_list = normalize_modes(modes)
    cache_dir = cache_dir or str(DEFAULT_CACHE_DIR)
    if output_file is None:
        output_file = IMAGERY_SOURCE_OUTPUT_FILES.get(
            imagery_source, cache_path / f"visdiff_{imagery_source}_descriptions.csv"
        )
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
                            imagery_source,
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
