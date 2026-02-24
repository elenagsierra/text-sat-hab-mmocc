#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "open-clip-torch==3.2.0",
#     "earthengine-api==1.4.0",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Relate habitat descriptor prediction errors to tree canopy cover."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import ee
import fire
import numpy as np
import open_clip
import pandas as pd
import torch
from joblib import Memory
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mmocc import utils as mm_utils
from mmocc.config import (
    cache_path,
    default_sat_backbone,
    num_habitat_descriptions,
    pca_dim,
)
from mmocc.utils import get_focal_species_ids, load_data

CLIP_MODEL_NAME = "ViT-bigG-14"
CLIP_PRETRAINED = "laion2b_s39b_b160k"
CLIP_IMAGE_BACKBONE = "clip_vitbigg14"
OUTPUT_DIR = cache_path / "habitat_explainability"

DESCRIPTOR_FILES: dict[str, Path] = {
    "visdiff_clip": cache_path / "visdiff_descriptions.csv",
    "expert_clip": cache_path / "expert_habitat_descriptions.csv",
}
CANOPY_COLLECTION = "MODIS/061/MOD44B"
CANOPY_BAND = "Percent_Tree_Cover"
CANOPY_SCALE_METERS = 250
CANOPY_CACHE = Memory(cache_path / "joblib_canopy", verbose=0)


@dataclass(frozen=True)
class DescriptorSettings:
    max_entries: int | None


class ClipTextEncoder:
    def __init__(self, model_name: str, pretrained: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model.eval()
        self.model = model.to(device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.device = device

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        tokens = open_clip.tokenize(list(texts))
        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = tokens[start : start + batch_size].to(self.device)
                features = self.model.encode_text(chunk)  # type: ignore[attr-defined]
                features = torch.nn.functional.normalize(features, dim=-1)
                outputs.append(features.detach().cpu().float())
        return torch.cat(outputs, dim=0).numpy()


_TEXT_ENCODER: ClipTextEncoder | None = None


def get_text_encoder() -> ClipTextEncoder:
    global _TEXT_ENCODER
    if _TEXT_ENCODER is None:
        _TEXT_ENCODER = ClipTextEncoder(CLIP_MODEL_NAME, CLIP_PRETRAINED)
    return _TEXT_ENCODER


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def load_descriptor_table(backbone: str) -> pd.DataFrame:
    path = DESCRIPTOR_FILES[backbone]
    if not path.exists():
        raise FileNotFoundError(f"Descriptor cache missing at {path}")
    df = pd.read_csv(path)
    if "taxon_id" not in df or "difference" not in df:
        raise ValueError(f"Descriptor file {path} missing required columns")
    df["taxon_id"] = df["taxon_id"].astype(str)
    return df


def select_descriptors(
    backbone: str, taxon_id: str, limit: int | None
) -> tuple[list[str], np.ndarray]:
    if limit is not None and limit <= 0:
        return [], np.empty((0,), dtype=np.float32)
    df = load_descriptor_table(backbone)
    subset = df[df["taxon_id"] == taxon_id].copy()
    if subset.empty:
        return [], np.empty((0,), dtype=np.float32)
    subset = subset.dropna(subset=["difference"])
    subset["difference"] = subset["difference"].astype(str).str.strip()
    subset = subset[subset["difference"] != ""]
    if subset.empty:
        return [], np.empty((0,), dtype=np.float32)
    subset = subset.dropna(subset=["auroc"])
    subset = subset.sort_values("auroc", ascending=False)
    subset = subset.drop_duplicates(subset="difference")
    if limit is not None:
        subset = subset.head(limit)
    texts = subset["difference"].tolist()
    scores = subset["auroc"].fillna(0.0).astype(float).to_numpy()
    if len(scores) == 0:
        scores = np.ones(len(texts), dtype=np.float32)
    return texts, scores.astype(np.float32)


def compute_similarity_features(
    embeddings: np.ndarray, descriptor_vectors: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    if descriptor_vectors.size == 0:
        raise ValueError("No descriptor embeddings available.")
    if embeddings.shape[0] == 0:
        return np.empty((0, descriptor_vectors.shape[0]), dtype=np.float32)
    image_unit = normalize_rows(embeddings.astype(np.float32, copy=False))
    descriptor_unit = normalize_rows(descriptor_vectors.astype(np.float32, copy=False))
    sims = image_unit @ descriptor_unit.T
    if weights.size:
        w = weights.astype(np.float32, copy=True)
        max_abs = np.max(np.abs(w))
        if max_abs > 0:
            w /= max_abs
        else:
            w[:] = 1.0
        sims *= w.reshape(1, -1)
    return sims.astype(np.float32)


@CANOPY_CACHE.cache
def fetch_canopy_fraction(
    latitudes: tuple[float, ...], longitudes: tuple[float, ...]
) -> np.ndarray:
    ee.Initialize()
    image = ee.ImageCollection(CANOPY_COLLECTION).select(CANOPY_BAND).mosaic()
    features = [
        ee.Feature(
            ee.Geometry.Point([float(lon), float(lat)]),
            {"index": int(idx)},
        )
        for idx, (lat, lon) in enumerate(zip(latitudes, longitudes))
    ]
    fc = ee.FeatureCollection(features)
    sampled = image.sampleRegions(
        collection=fc, scale=CANOPY_SCALE_METERS, geometries=False, tileScale=4
    ).sort("index")
    results = np.full(len(latitudes), np.nan, dtype=np.float32)
    records = sampled.getInfo().get("features", [])
    for rec in records:
        props = rec.get("properties", {})
        idx = int(props.get("index", -1))
        if idx < 0 or idx >= len(results):
            continue
        value = props.get(CANOPY_BAND)
        try:
            val_f = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(val_f):
            continue
        results[idx] = val_f
    return results


def parse_backbones(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return sorted(DESCRIPTOR_FILES.keys())
    if isinstance(value, str):
        candidates = [v.strip() for v in value.split(",") if v.strip()]
    else:
        candidates = list(value)
    unknown = [v for v in candidates if v not in DESCRIPTOR_FILES]
    if unknown:
        raise ValueError(f"Unknown descriptor backbones requested: {unknown}")
    return sorted(set(candidates))


def parse_species(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return get_focal_species_ids()
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v) for v in value]


def evaluate_species(
    taxon_id: str,
    descriptor_backbone: str,
    descriptor_limit: int | None,
    sat_backbone: str,
    image_backbone: str,
    image_pca_dim: int,
    use_range_filter: bool,
) -> tuple[list[dict], list[dict], dict]:
    original_limit = mm_utils.limit_to_range
    mm_utils.limit_to_range = use_range_filter
    try:
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
        ) = load_data(taxon_id, {"image", "sat"}, image_backbone, sat_backbone)
    finally:
        mm_utils.limit_to_range = original_limit

    descriptor_texts, descriptor_scores = select_descriptors(
        descriptor_backbone, taxon_id, descriptor_limit
    )
    if len(descriptor_texts) == 0:
        raise RuntimeError(
            f"No descriptors available for taxon {taxon_id} and backbone {descriptor_backbone}"
        )

    descriptor_embeddings = get_text_encoder().encode(descriptor_texts)
    habitat_features = compute_similarity_features(
        features_modalities["image"], descriptor_embeddings, descriptor_scores
    )

    sat_scaler = StandardScaler().fit(features_modalities["sat"][mask_train])
    sat_train = sat_scaler.transform(features_modalities["sat"][mask_train])
    sat_test = sat_scaler.transform(features_modalities["sat"][mask_test])

    sat_model = Ridge(alpha=1.0)
    sat_model.fit(sat_train, habitat_features[mask_train])
    preds_test = sat_model.predict(sat_test)
    abs_errors = np.abs(habitat_features[mask_test] - preds_test)

    feature_path = cache_path / "features"
    ids_all = np.load(
        feature_path / f"wi_blank_image_features_{image_backbone}_ids.npy",
        allow_pickle=True,
    )
    locs = np.load(feature_path / f"wi_blank_image_features_{image_backbone}_locs.npy", allow_pickle=True)
    loc_ids = ids_all[mask_test]
    locs = locs[mask_test]
    latitudes = locs[:, 0]
    longitudes = locs[:, 1]
    canopy = fetch_canopy_fraction(
        tuple(latitudes.tolist()), tuple(longitudes.tolist())
    )

    per_descriptor_rows: list[dict] = []
    for idx, text in enumerate(descriptor_texts):
        errors = abs_errors[:, idx]
        valid = np.isfinite(errors) & np.isfinite(canopy)
        if valid.sum() < 2:
            corr = math.nan
        else:
            corr = float(np.corrcoef(errors[valid], canopy[valid])[0, 1])
        per_descriptor_rows.append(
            dict(
                taxon_id=taxon_id,
                scientific_name=scientific_name,
                common_name=common_name,
                descriptor_backbone=descriptor_backbone,
                descriptor_index=idx,
                descriptor_text=text,
                descriptor_score=float(descriptor_scores[idx]),
                pearson_abs_error_canopy=corr,
                n_valid=int(valid.sum()),
                n_test=int(mask_test.sum()),
                sat_backbone=sat_backbone,
                image_backbone=image_backbone,
            )
        )

    per_location_rows: list[dict] = []
    for i, loc_id in enumerate(loc_ids):
        row = dict(
            loc_id=loc_id,
            taxon_id=taxon_id,
            canopy_fraction=float(canopy[i]),
            sat_backbone=sat_backbone,
            image_backbone=image_backbone,
            descriptor_backbone=descriptor_backbone,
        )
        for idx, text in enumerate(descriptor_texts):
            row[f"abs_error_{idx}"] = float(abs_errors[i, idx])
        per_location_rows.append(row)

    summary_row = dict(
        taxon_id=taxon_id,
        scientific_name=scientific_name,
        common_name=common_name,
        descriptor_backbone=descriptor_backbone,
        sat_backbone=sat_backbone,
        image_backbone=image_backbone,
        num_descriptors=len(descriptor_texts),
    )
    return per_descriptor_rows, per_location_rows, summary_row


def main(
    descriptor_backbones: Sequence[str] | str | None = None,
    descriptor_limit: int | None = num_habitat_descriptions,
    sat_backbone: str = default_sat_backbone,
    image_backbone: str = CLIP_IMAGE_BACKBONE,
    species_ids: Sequence[str] | str | None = None,
    image_pca_dim: int = pca_dim,
    output_suffix: str | None = None,
    use_range_filter: bool | str = True,
):
    if isinstance(use_range_filter, str):
        use_range_filter = use_range_filter.lower() in {"1", "true", "yes", "y"}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    backbones = parse_backbones(descriptor_backbones)
    descriptor_limit = int(descriptor_limit) if descriptor_limit is not None else None
    species_list = parse_species(species_ids)

    per_descriptor_rows: list[dict] = []
    per_location_rows: list[dict] = []
    summary_rows: list[dict] = []

    for backbone in backbones:
        descriptor_df = load_descriptor_table(backbone)
        available_species = set(descriptor_df["taxon_id"].unique().tolist())

        for taxon_id in tqdm(species_list, desc=f"{backbone} species"):
            if taxon_id not in available_species:
                continue
            try:
                desc_rows, loc_rows, summary = evaluate_species(
                    taxon_id,
                    backbone,
                    descriptor_limit,
                    sat_backbone,
                    image_backbone,
                    image_pca_dim,
                    use_range_filter,
                )
                per_descriptor_rows.extend(desc_rows)
                per_location_rows.extend(loc_rows)
                summary_rows.append(summary)
            except Exception as exc:  # pragma: no cover - defensive
                summary_rows.append(
                    dict(
                        taxon_id=taxon_id,
                        descriptor_backbone=backbone,
                        status=f"error: {exc}",
                    )
                )
                print(f"[WARN] Failed for taxon {taxon_id} ({backbone}): {exc}")

    if not per_descriptor_rows:
        raise RuntimeError("No results were produced; check descriptor availability.")

    suffix = f"_{output_suffix}" if output_suffix else ""
    base_name = f"habitat_canopy_error{suffix}"
    per_descriptor_path = OUTPUT_DIR / f"{base_name}_per_descriptor.csv"
    per_location_path = OUTPUT_DIR / f"{base_name}_per_location.csv"
    summary_path = OUTPUT_DIR / f"{base_name}_summary.csv"

    pd.DataFrame(per_descriptor_rows).to_csv(per_descriptor_path, index=False)
    pd.DataFrame(per_location_rows).to_csv(per_location_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(
        f"Wrote descriptor rows to {per_descriptor_path}, location rows to {per_location_path}, "
        f"summary rows to {summary_path}"
    )


if __name__ == "__main__":
    fire.Fire(main)
