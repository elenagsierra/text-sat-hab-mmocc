import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from mmocc_sat.config import (
    cache_path,
    default_image_backbone,
    default_sat_backbone,
    wi_image_path,
)
from mmocc_sat.utils import (
    experiment_to_filename,
    filename_to_experiment,
    load_data,
)

LOGGER = logging.getLogger(__name__)
FIT_RESULTS_DIR = cache_path / "fit_results"
FEATURES_DIR = cache_path / "features"


def _normalize_backbone_name(value: str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none"}:
        return None
    return value


def load_image_lookup() -> pd.DataFrame:
    df = pd.read_pickle(cache_path / "wi_blank_images.pkl")
    if df.empty:
        raise RuntimeError("No cached Wildlife Insights blank images found.")
    df = df.sort_values("Date_Time")
    df_valid_images = pd.read_csv(
        cache_path / "wi_blank_images_valid.txt", header=None, names=["FilePath"]
    )
    df = pd.merge(df, df_valid_images, how="inner", on="FilePath")
    df["image_path"] = df["FilePath"].str.replace("gs://", f"{wi_image_path}/", n=1)
    df["image_exists"] = df["image_path"].apply(lambda path: Path(path).exists())
    return df.drop_duplicates(subset="loc_id").set_index("loc_id")[
        ["image_path", "image_exists", "Latitude", "Longitude"]
    ]


def resolve_fit_results_path(
    taxon_id: str,
    modalities: Iterable[str],
    image_backbone: str | None,
    sat_backbone: str | None,
) -> Tuple[Path, List[str], str | None, str | None]:
    requested_modalities = set(modalities)
    requested_image_backbone = _normalize_backbone_name(image_backbone)
    requested_sat_backbone = _normalize_backbone_name(sat_backbone)
    filename = experiment_to_filename(
        taxon_id,
        requested_modalities,
        requested_image_backbone,
        requested_sat_backbone,
        "pkl",
    )
    root = FIT_RESULTS_DIR
    if not root.exists():
        raise FileNotFoundError(f"Fit results directory does not exist: {root}")
    candidate = root / filename
    if candidate.exists():
        return (
            candidate,
            sorted(list(requested_modalities)),
            requested_image_backbone,
            requested_sat_backbone,
        )

    available: List[Path] = sorted(root.glob(f"{taxon_id}_modalities_*.pkl"))
    if not available:
        raise FileNotFoundError(
            f"No fit results found for taxon_id={taxon_id}. Looked for {filename} in {FIT_RESULTS_DIR}."
        )

    parsed: List[Tuple[Path, set[str], str | None, str | None]] = []
    for path in available:
        fallback_taxon, fallback_modalities, fallback_image, fallback_sat = (
            filename_to_experiment(path.name)
        )
        if fallback_taxon != taxon_id:
            continue
        parsed.append(
            (
                path,
                set(fallback_modalities),
                _normalize_backbone_name(fallback_image),
                _normalize_backbone_name(fallback_sat),
            )
        )
    if not parsed:
        raise FileNotFoundError(
            f"No parseable fit results for taxon_id={taxon_id} in {FIT_RESULTS_DIR}."
        )

    # Prefer experiments that best match requested modalities/backbones.
    def score(entry: Tuple[Path, set[str], str | None, str | None]) -> int:
        _, mods, img_bb, sat_bb = entry
        s = 0
        if requested_modalities.issubset(mods):
            s += 100
        else:
            s += 10 * len(requested_modalities & mods)
        if "sat" in requested_modalities and "sat" in mods:
            s += 20
        if "image" in requested_modalities and "image" in mods:
            s += 10
        if requested_image_backbone is not None and img_bb == requested_image_backbone:
            s += 3
        if requested_sat_backbone is not None and sat_bb == requested_sat_backbone:
            s += 3
        return s

    fallback, fallback_modalities, fallback_image, fallback_sat = max(
        parsed, key=score
    )
    LOGGER.warning(
        "Requested experiment '%s' not found. Using fallback results '%s' which were trained with modalities=%s, "
        "image_backbone=%s, sat_backbone=%s.",
        filename,
        fallback.name,
        sorted(list(fallback_modalities)),
        fallback_image,
        fallback_sat,
    )
    return fallback, sorted(list(fallback_modalities)), fallback_image, fallback_sat


def load_fit_results(path: Path) -> Dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_location_ids(
    image_backbone: str | None, sat_backbone: str | None
) -> np.ndarray:
    image_backbone = _normalize_backbone_name(image_backbone) or default_image_backbone
    sat_backbone = _normalize_backbone_name(sat_backbone) or default_sat_backbone

    image_ids_path = FEATURES_DIR / f"wi_blank_image_features_{image_backbone}_ids.npy"
    sat_ids_path = FEATURES_DIR / f"wi_blank_sat_features_{sat_backbone}_ids.npy"

    if image_ids_path.exists():
        return np.load(image_ids_path, allow_pickle=True)
    if sat_ids_path.exists():
        return np.load(sat_ids_path, allow_pickle=True)
    raise FileNotFoundError(
        f"Missing location id files at {image_ids_path} and {sat_ids_path}."
    )


def compute_site_scores(
    taxon_id: str,
    modalities: Sequence[str],
    image_backbone: str | None,
    sat_backbone: str | None,
    fit_results: Dict,
) -> Tuple[pd.DataFrame, str]:
    image_backbone = _normalize_backbone_name(image_backbone)
    sat_backbone = _normalize_backbone_name(sat_backbone)

    (
        _,
        _,
        _,
        _,
        _,
        _,
        scientific_name,
        common_name,
        mask_train,
        mask_test,
        _,
        _,
        _,
        _,
        _,
        features_modalities,
    ) = load_data(taxon_id, set(modalities), image_backbone, sat_backbone)

    ids_all = load_location_ids(image_backbone, sat_backbone)
    mask_train = np.asarray(mask_train, dtype=bool)
    mask_test = np.asarray(mask_test, dtype=bool)
    valid_mask = mask_train | mask_test
    if not (
        len(ids_all) == len(mask_train) == len(mask_test)
    ):  # pragma: no cover - defensive
        raise ValueError("Location ids and masks are misaligned.")

    scaler = fit_results["modalities_scaler"]
    pca = fit_results["modalities_pca"]
    coefficients = fit_results["modality_coefficients"]

    contributions: Dict[str, np.ndarray] = {}
    combined_valid = np.zeros(valid_mask.sum(), dtype=np.float32)
    for modality in modalities:
        if modality not in features_modalities:
            raise KeyError(f"Raw features for modality '{modality}' were not loaded.")
        features_valid = features_modalities[modality][valid_mask]
        transformed = scaler[modality].transform(features_valid)
        reduced = pca[modality].transform(transformed)
        modality_scores = reduced @ coefficients[modality]
        combined_valid += modality_scores
        full_scores = np.full(len(mask_train), np.nan, dtype=np.float32)
        full_scores[valid_mask] = modality_scores
        contributions[modality] = full_scores

    combined = np.full(len(mask_train), np.nan, dtype=np.float32)
    combined[valid_mask] = combined_valid
    site_scores = pd.DataFrame(
        {
            "loc_id": ids_all,
            "is_train": mask_train,
            "is_test": mask_test,
            "score_total": combined,
        }
    )
    for modality, scores in contributions.items():
        site_scores[f"score_{modality}"] = scores

    display_name = (
        common_name
        if common_name is not None and not pd.isna(common_name)
        else scientific_name
    )
    return site_scores, display_name


def rank_image_groups(
    site_scores: pd.DataFrame,
    modalities: Sequence[str],
    mode: str,
    unique_weight: float,
    top_k: int,
    image_modality: str = "image",
    test: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    column = f"score_{image_modality}"
    if column not in site_scores.columns:
        raise KeyError(
            f"{column} not found in site scores. Available columns: {site_scores.columns.tolist()}"
        )

    subset_name = "is_test" if test else "is_train"
    subset = site_scores[site_scores["image_exists"] & site_scores[subset_name]].copy()
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    if mode == "unique":
        other_columns = [
            f"score_{mod}"
            for mod in modalities
            if mod != image_modality and f"score_{mod}" in subset
        ]
        penalty = subset[other_columns].sum(axis=1) if other_columns else 0.0
        subset["rank_score"] = subset[column] * unique_weight - penalty
    else:
        subset["rank_score"] = subset[column]

    subset = subset.sort_values("rank_score", ascending=False)
    limit = min(top_k, len(subset) // 2)
    if limit <= 0:
        return pd.DataFrame(), pd.DataFrame()
    positives = subset.head(limit).copy()
    negatives = subset.tail(limit).copy().sort_values("rank_score", ascending=True)
    return positives.reset_index(drop=True), negatives.reset_index(drop=True)


def select_image_groups(
    site_scores: pd.DataFrame,
    modalities: Sequence[str],
    mode: str,
    unique_weight: float,
    top_k: int,
    image_modality: str = "image",
    test: bool = False,
) -> Tuple[List[str], List[str]]:
    positives, negatives = rank_image_groups(
        site_scores, modalities, mode, unique_weight, top_k, image_modality, test=test
    )
    return (
        positives["image_path"].tolist() if not positives.empty else [],
        negatives["image_path"].tolist() if not negatives.empty else [],
    )


def load_sat_lookup() -> pd.DataFrame:
    """Load satellite imagery lookup table with paths and coordinates."""
    df = pd.read_pickle(cache_path / "wi_blank_images.pkl")
    if df.empty:
        raise RuntimeError("No cached Wildlife Insights blank images found.")
    df = df.sort_values("Date_Time")
    # Build satellite image paths based on location IDs
    df["sat_path"] = df["loc_id"].apply(
        lambda loc_id: str(cache_path / "sat_images" / f"{loc_id}.png")
    )
    df["sat_exists"] = df["sat_path"].apply(lambda path: Path(path).exists())
    return df.drop_duplicates(subset="loc_id").set_index("loc_id")[
        ["sat_path", "sat_exists", "Latitude", "Longitude"]
    ]


def rank_sat_groups(
    site_scores: pd.DataFrame,
    modalities: Sequence[str],
    mode: str,
    unique_weight: float,
    top_k: int,
    sat_modality: str = "sat",
    test: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rank sites by satellite modality scores."""
    column = f"score_{sat_modality}"
    if column not in site_scores.columns:
        raise KeyError(
            f"{column} not found in site scores. Available columns: {site_scores.columns.tolist()}"
        )

    subset_name = "is_test" if test else "is_train"
    subset = site_scores[site_scores["sat_exists"] & site_scores[subset_name]].copy()
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    if mode == "unique":
        other_columns = [
            f"score_{mod}"
            for mod in modalities
            if mod != sat_modality and f"score_{mod}" in subset
        ]
        penalty = subset[other_columns].sum(axis=1) if other_columns else 0.0
        subset["rank_score"] = subset[column] * unique_weight - penalty
    else:
        subset["rank_score"] = subset[column]

    subset = subset.sort_values("rank_score", ascending=False)
    limit = min(top_k, len(subset) // 2)
    if limit <= 0:
        return pd.DataFrame(), pd.DataFrame()
    positives = subset.head(limit).copy()
    negatives = subset.tail(limit).copy().sort_values("rank_score", ascending=True)
    return positives.reset_index(drop=True), negatives.reset_index(drop=True)


def select_sat_groups(
    site_scores: pd.DataFrame,
    modalities: Sequence[str],
    mode: str,
    unique_weight: float,
    top_k: int,
    sat_modality: str = "sat",
    test: bool = False,
) -> Tuple[List[str], List[str]]:
    """Select positive and negative satellite image groups for VisDiff."""
    positives, negatives = rank_sat_groups(
        site_scores, modalities, mode, unique_weight, top_k, sat_modality, test=test
    )
    return (
        positives["sat_path"].tolist() if not positives.empty else [],
        negatives["sat_path"].tolist() if not negatives.empty else [],
    )
