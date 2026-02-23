#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "open-clip-torch==3.2.0",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Estimate how much habitat descriptor similarities are predictable from satellite
remote sensing features."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import fire
import numpy as np
import open_clip
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mmocc import utils as mm_utils
from mmocc.config import (
    cache_path,
    default_sat_backbone,
    num_habitat_descriptions,
)
from mmocc.utils import get_focal_species_ids, load_data

CLIP_MODEL_NAME = "ViT-bigG-14"
CLIP_PRETRAINED = "laion2b_s39b_b160k"
CLIP_SAT_BACKBONE = "clip_vitbigg14"
OUTPUT_DIR = cache_path / "habitat_explainability"

DESCRIPTOR_FILES: dict[str, Path] = {
    "visdiff_clip": cache_path / "visdiff_sat_descriptions.csv",
    "expert_clip": cache_path / "expert_habitat_descriptions.csv",
}


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


def safe_r2_scores(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    scores = np.empty(y_true.shape[1], dtype=np.float32)
    for idx in range(y_true.shape[1]):
        target = y_true[:, idx]
        pred = y_pred[:, idx]
        if np.allclose(target, target[0]):
            scores[idx] = np.nan
            continue
        scores[idx] = r2_score(target, pred)
    return scores


@dataclass(frozen=True)
class PredictabilityResult:
    r2_sat: np.ndarray
    n_train: int
    n_test: int
    sat_dim: int


def evaluate_predictability(
    habitat_features: np.ndarray,
    sat_features: np.ndarray,
    mask_train: np.ndarray,
    mask_test: np.ndarray,
) -> PredictabilityResult:
    n_train = int(mask_train.sum())
    n_test = int(mask_test.sum())
    if n_train < 2 or n_test < 2:
        empty = np.full(habitat_features.shape[1], np.nan, dtype=np.float32)
        return PredictabilityResult(
            r2_sat=empty,
            n_train=n_train,
            n_test=n_test,
            sat_dim=sat_features.shape[1],
        )

    sat_scaler = StandardScaler().fit(sat_features[mask_train])
    sat_train = sat_scaler.transform(sat_features[mask_train])
    sat_test = sat_scaler.transform(sat_features[mask_test])

    sat_model = Ridge(alpha=1.0)
    sat_model.fit(sat_train, habitat_features[mask_train])
    sat_pred = sat_model.predict(sat_test)
    r2_sat = safe_r2_scores(habitat_features[mask_test], sat_pred)

    return PredictabilityResult(
        r2_sat=r2_sat,
        n_train=n_train,
        n_test=n_test,
        sat_dim=sat_features.shape[1],
    )


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
    sat_pca_dim: int,
) -> tuple[list[dict], dict]:
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
    ) = load_data(taxon_id, {"sat", "covariates"}, None, sat_backbone)

    descriptor_texts, descriptor_scores = select_descriptors(
        descriptor_backbone, taxon_id, descriptor_limit
    )
    if len(descriptor_texts) == 0:
        raise RuntimeError(
            f"No descriptors available for taxon {taxon_id} and backbone {descriptor_backbone}"
        )

    descriptor_embeddings = get_text_encoder().encode(descriptor_texts)
    habitat_features = compute_similarity_features(
        features_modalities["sat"], descriptor_embeddings, descriptor_scores
    )

    predictability = evaluate_predictability(
        habitat_features,
        features_modalities["sat"],
        mask_train,
        mask_test,
    )

    rows: list[dict] = []
    for idx, text in enumerate(descriptor_texts):
        r2_sat = float(predictability.r2_sat[idx])
        rows.append(
            dict(
                taxon_id=taxon_id,
                scientific_name=scientific_name,
                common_name=common_name,
                descriptor_backbone=descriptor_backbone,
                descriptor_index=idx,
                descriptor_text=text,
                descriptor_score=float(descriptor_scores[idx]),
                r2_sat=r2_sat,
                r2_sat_clipped=(
                    float(max(r2_sat, 0.0)) if not math.isnan(r2_sat) else math.nan
                ),
                n_train=predictability.n_train,
                n_test=predictability.n_test,
                sat_backbone=sat_backbone,
                sat_dim=predictability.sat_dim,
                num_descriptors=len(descriptor_texts),
            )
        )

    summary_row = dict(
        taxon_id=taxon_id,
        scientific_name=scientific_name,
        common_name=common_name,
        descriptor_backbone=descriptor_backbone,
        sat_backbone=sat_backbone,
        num_descriptors=len(descriptor_texts),
        n_train=predictability.n_train,
        n_test=predictability.n_test,
        mean_r2_sat=float(np.nanmean(predictability.r2_sat)),
        mean_r2_sat_clipped=float(np.nanmean(np.maximum(predictability.r2_sat, 0.0))),
        status="ok",
    )

    return rows, summary_row


def main(
    descriptor_backbones: Sequence[str] | str | None = None,
    descriptor_limit: int | None = num_habitat_descriptions,
    sat_backbone: str = default_sat_backbone,
    species_ids: Sequence[str] | str | None = None,
    output_suffix: str | None = None,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    backbones = parse_backbones(descriptor_backbones)
    descriptor_limit = int(descriptor_limit) if descriptor_limit is not None else None
    species_list = parse_species(species_ids)

    per_descriptor_rows: list[dict] = []
    summary_rows: list[dict] = []

    for backbone in backbones:
        descriptor_df = load_descriptor_table(backbone)
        available_species = set(descriptor_df["taxon_id"].unique().tolist())

        for taxon_id in tqdm(species_list, desc=f"{backbone} species"):
            if taxon_id not in available_species:
                continue
            try:
                rows, summary = evaluate_species(
                    taxon_id,
                    backbone,
                    descriptor_limit,
                    sat_backbone,
                )
                per_descriptor_rows.extend(rows)
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
    base_name = f"habitat_rs_explainability{suffix}"
    per_descriptor_path = OUTPUT_DIR / f"{base_name}_per_descriptor.csv"
    summary_path = OUTPUT_DIR / f"{base_name}_summary.csv"

    pd.DataFrame(per_descriptor_rows).to_csv(per_descriptor_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(
        f"Wrote {len(per_descriptor_rows)} descriptor rows to {per_descriptor_path} "
        f"and {len(summary_rows)} summary rows to {summary_path}"
    )


if __name__ == "__main__":
    fire.Fire(main)
