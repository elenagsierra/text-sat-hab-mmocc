#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "earthengine-api==1.4.0",
#     "transformers",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Refit occupancy models using GRAFT satellite embeddings and VisDiff descriptions."""

import os
import pickle
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Sequence

import fire
import numpy as np
import pandas as pd
import submitit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mmocc.config import (
    cache_path,
    limit_to_range,
    num_habitat_descriptions,
    pca_dim,
)
from mmocc.graft_utils import get_graft_text_encoder
from mmocc.solvers.logistic import fit_logistic
from mmocc.utils import (
    experiment_to_filename,
    get_focal_species_ids,
    get_submitit_executor,
    load_data,
    run_biolith_in_process,
)

GRAFT_SAT_BACKBONE = "graft"

DESCRIPTOR_FILES: dict[str, Path] = {
    "visdiff": cache_path / "visdiff_sat_descriptions.csv",
    "expert": cache_path / "expert_habitat_descriptions.csv",
}

DESCRIPTOR_LABELS: dict[str, str] = {
    "visdiff": "graft_visdiff",
    "expert": "graft_expert",
}


@dataclass(frozen=True)
class DescriptorSettings:
    max_entries: int | None


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def load_descriptor_table(source: str, require_valid: bool = False) -> pd.DataFrame:
    df = pd.read_csv(DESCRIPTOR_FILES[source])
    if "taxon_id" not in df or "difference" not in df:
        raise ValueError(f"Descriptor file {DESCRIPTOR_FILES[source]} missing required columns")
    df["taxon_id"] = df["taxon_id"].astype(str)
    if require_valid:
        df = df.dropna(subset=["difference"]).copy()
        df["difference"] = df["difference"].astype(str).str.strip()
        df = df[df["difference"] != ""]
    return df


def select_descriptors(
    source: str, taxon_id: str, limit: int | None
) -> tuple[list[str], np.ndarray]:
    if limit is not None and limit <= 0:
        return [], np.empty((0,), dtype=np.float32)
    df = load_descriptor_table(source)
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
    sat_unit = normalize_rows(embeddings.astype(np.float32, copy=False))
    desc_unit = normalize_rows(descriptor_vectors.astype(np.float32, copy=False))
    sims = sat_unit @ desc_unit.T
    if weights.size:
        w = weights.astype(np.float32, copy=True)
        max_abs = np.max(np.abs(w))
        if max_abs > 0:
            w /= max_abs
        else:
            w[:] = 1.0
        sims *= w.reshape(1, -1)
    return sims.astype(np.float32)


def fit(
    taxon_id: str,
    modalities: set[str],
    descriptor_source: str,
    descriptor_settings: DescriptorSettings,
    sat_backbone_data: str,
):
    modalities_list = sorted(list(modalities))
    if "sat" not in modalities_list:
        raise ValueError("GRAFT descriptor refits require the 'sat' modality.")

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
        too_close,
        y_train,
        y_test,
        y_train_naive,
        y_test_naive,
        features_modalities,
    ) = load_data(
        taxon_id,
        modalities,
        image_backbone_name=None,
        sat_backbone_name=sat_backbone_data,
    )

    descriptor_texts, descriptor_scores = select_descriptors(
        descriptor_source, taxon_id, descriptor_settings.max_entries
    )
    if len(descriptor_texts) == 0:
        raise RuntimeError(
            f"No descriptors available for taxon {taxon_id} and source {descriptor_source}"
        )

    descriptor_embeddings = get_graft_text_encoder().encode(descriptor_texts)
    similarity_features = compute_similarity_features(
        features_modalities["sat"], descriptor_embeddings, descriptor_scores
    )
    features_modalities["sat"] = similarity_features

    naive_detection_prob_train = (
        np.mean(np.nanmean(y_train, axis=1)[y_train_naive])
        if np.any(y_train_naive)
        else np.nan
    )
    naive_detection_prob_test = (
        np.mean(np.nanmean(y_test, axis=1)[y_test_naive])
        if np.any(y_test_naive)
        else np.nan
    )

    modalities_scaler = {
        modality: StandardScaler().fit(features_modalities[modality][mask_train])
        for modality in modalities_list
    }

    modalities_pca = {
        modality: PCA(
            n_components=min(
                pca_dim, mask_train.sum(), features_modalities[modality].shape[1]
            )
        ).fit(
            modalities_scaler[modality].transform(
                features_modalities[modality][mask_train]
            )
        )
        for modality in modalities_list
    }

    features_train = np.concatenate(
        [
            modalities_pca[modality].transform(
                modalities_scaler[modality].transform(
                    features_modalities[modality][mask_train]
                )
            )
            for modality in modalities_list
        ],
        axis=1,
    )

    features_test = np.concatenate(
        [
            modalities_pca[modality].transform(
                modalities_scaler[modality].transform(
                    features_modalities[modality][mask_test]
                )
            )
            for modality in modalities_list
        ],
        axis=1,
    )

    features_dims = {
        modality: modalities_pca[modality].n_components_ for modality in modalities_list
    }

    sat_backbone_label = DESCRIPTOR_LABELS[descriptor_source]
    species_results = dict(
        taxon_id=taxon_id,
        scientific_name=scientific_name,
        common_name=common_name,
        modalities=modalities_list,
        sat_backbone=sat_backbone_label,
        sat_backbone_data=sat_backbone_data,
        descriptor_source=descriptor_source,
        limit_to_range=limit_to_range,
        modalities_scaler=modalities_scaler,
        modalities_pca=modalities_pca,
        mean_naive_occupancy_train=y_train_naive.mean().item(),
        mean_naive_occupancy_test=y_test_naive.mean().item(),
        naive_detection_prob_train=naive_detection_prob_train,
        naive_detection_prob_test=naive_detection_prob_test,
        mean_num_observations_train=np.nanmean(y_train).item(),
        mean_num_observations_test=np.nanmean(y_test).item(),
        mean_num_nonnan_train=np.sum(np.isfinite(y_train)).item(),
        mean_num_nonnan_test=np.sum(np.isfinite(y_test)).item(),
        test_sites_too_close=too_close.sum().item(),
        graft_descriptor_texts=descriptor_texts,
        graft_descriptor_scores=descriptor_scores.tolist(),
    )

    lr_results = fit_logistic(
        features_train,
        features_test,
        y_train_naive,
        y_test_naive,
        features_dims,
        modalities_list,
    )
    species_results.update({f"lr_{k}": v for k, v in lr_results.items()})

    gpu_env_vars = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("CUDA_", "SLURM_", "GPU_"))
    }
    regularization = "l1"
    regressor_name = "LinearRegression"
    q = Queue()
    numpyro_args = (
        gpu_env_vars,
        features_train,
        features_test,
        y_train,
        y_test,
        regressor_name,
        modalities_list,
        features_dims,
        regularization,
    )
    p = Process(target=run_biolith_in_process, daemon=False, args=(q, *numpyro_args))
    p.start()
    try:
        biolith_results = q.get()
    except Exception as exc:
        p.terminate()
        raise RuntimeError("Biolith fitting process timed out.") from exc
    finally:
        p.join()

    species_results.update(biolith_results)

    filename = experiment_to_filename(
        taxon_id, modalities_list, None, sat_backbone_label, "pkl"
    )
    fit_results_path = cache_path / "fit_results"
    fit_results_path.mkdir(parents=True, exist_ok=True)
    with open(fit_results_path / filename, "wb") as f:
        pickle.dump(species_results, f)


def parse_sources(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return ["visdiff"]
    if isinstance(value, str):
        candidates = [v.strip() for v in value.split(",") if v.strip()]
    else:
        candidates = list(value)
    unknown = [v for v in candidates if v not in DESCRIPTOR_FILES]
    if unknown:
        raise ValueError(f"Unknown descriptor sources requested: {unknown}")
    return sorted(set(candidates))


def parse_modalities(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return ["sat", "covariates"]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


def parse_species(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return get_focal_species_ids()
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v) for v in value]


def main(
    skip_existing: bool = True,
    visdiff_limit: int | None = num_habitat_descriptions,
    expert_limit: int | None = num_habitat_descriptions,
    descriptor_sources: Sequence[str] | str | None = None,
    modalities: Sequence[str] | str | None = None,
    sat_backbone_data: str = GRAFT_SAT_BACKBONE,
    species: Sequence[str] | str | None = None,
    max_species: int | None = None,
):
    descriptor_settings = {
        "visdiff": DescriptorSettings(max_entries=visdiff_limit),
        "expert": DescriptorSettings(max_entries=expert_limit),
    }
    requested_sources = parse_sources(descriptor_sources)
    modalities_list = parse_modalities(modalities)
    if "sat" not in modalities_list:
        raise ValueError("Modalities must include 'sat' for GRAFT refits.")

    species_ids = parse_species(species)
    if max_species is not None:
        species_ids = species_ids[:max_species]

    descriptor_species = {
        source: set(
            load_descriptor_table(source, require_valid=True)[
                "taxon_id"
            ].unique().tolist()
        )
        for source in requested_sources
    }

    executor = get_submitit_executor("refit_graft")
    jobs = []
    with executor.batch():
        for taxon_id in species_ids:
            for source in requested_sources:
                if taxon_id not in descriptor_species[source]:
                    continue

                sat_backbone_label = DESCRIPTOR_LABELS[source]
                if skip_existing:
                    filename = experiment_to_filename(
                        taxon_id,
                        set(modalities_list),
                        None,
                        sat_backbone_label,
                        "pkl",
                    )
                    if (cache_path / "fit_results" / filename).exists():
                        continue

                jobs.append(
                    executor.submit(
                        fit,
                        taxon_id,
                        set(modalities_list),
                        source,
                        descriptor_settings[source],
                        sat_backbone_data,
                    )
                )

    for job in tqdm(
        submitit.helpers.as_completed(jobs), total=len(jobs), desc="Jobs", leave=False
    ):
        try:
            job.result()
        except Exception as exc:
            print(f"Job {job.job_id} failed with error: {exc}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    fire.Fire(main)
