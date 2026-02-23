#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "geopandas",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Quantify train/test domain shift per species across modalities and detection
rates. Uses satellite remote sensing features."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import fire
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mmocc import utils as mm_utils
from mmocc.config import (
    cache_path,
    default_sat_backbone,
    limit_to_range,
    pca_dim,
    wildlife_insights_test_project_ids,
)

OUTPUT_DIR = cache_path / "domain_shift"


@dataclass(frozen=True)
class ShiftMetrics:
    """Container for modality-specific shift metrics."""

    modality: str
    train_count: int
    test_count: int
    feature_dim: int
    pca_variance: float
    mean_distance: float
    cov_frobenius: float
    frechet_distance: float
    mmd_rbf: float


@dataclass(frozen=True)
class DetectionMetrics:
    """Summary of detection rate differences between train and test splits."""

    train_detection_rate: float
    test_detection_rate: float
    detection_rate_gap: float
    detection_wasserstein: float
    train_positive_site_fraction: float
    test_positive_site_fraction: float


@dataclass(frozen=True)
class BaseData:
    """Shared arrays and metadata used across species."""

    result_array: np.memmap
    species_map_df: pd.DataFrame
    taxon_map: dict[str, str]
    site_idx_all: np.ndarray
    mask_train_base: np.ndarray
    mask_test_base: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    features_modalities: dict[str, np.ndarray]
    conus_boundary: gpd.GeoDataFrame | None


def _parse_species_ids(species_ids: str | Sequence[str] | None) -> list[str]:
    """Convert CLI species ids argument into a list."""
    if species_ids is None:
        return mm_utils.get_focal_species_ids()
    if isinstance(species_ids, str):
        try:
            maybe_path = Path(species_ids)
            if not maybe_path.exists():
                raise FileNotFoundError()
            df = pd.read_csv(maybe_path)
            if "taxon_id" not in df.columns:
                raise ValueError(
                    f"Provided species file {maybe_path} missing required column 'taxon_id'"
                )
            return df["taxon_id"].astype(str).tolist()
        except:
            species_ids = [s.strip() for s in species_ids.split(",") if s.strip()]
        return species_ids
    return [str(sid) for sid in species_ids]


def _sample_features(
    train: np.ndarray, test: np.ndarray, max_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample train/test features to keep pairwise computations tractable."""
    if max_samples <= 0:
        return train, test
    if train.shape[0] > max_samples:
        train = train[rng.choice(train.shape[0], max_samples, replace=False)]
    if test.shape[0] > max_samples:
        test = test[rng.choice(test.shape[0], max_samples, replace=False)]
    return train, test


def _fill_missing(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace missing values with column means computed across train+test."""
    combined = np.vstack([train, test])
    col_means = np.nanmean(combined, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    if np.isnan(train).any():
        train = np.where(np.isfinite(train), train, col_means)
    if np.isnan(test).any():
        test = np.where(np.isfinite(test), test, col_means)
    return train, test


def _prepare_base_data(
    modalities: set[str],
    sat_backbone: str,
    apply_range_filter: bool,
) -> BaseData:
    """Load shared arrays and compute the train/test split once."""
    shape = tuple(
        np.loadtxt(cache_path / "wi_db_computer_vision_occupancy_y_shape.txt")
        .round()
        .astype(np.int64)
    )
    result_array = np.memmap(
        cache_path / "wi_db_computer_vision_occupancy_y.npy",
        dtype=np.float32,
        mode="r",
        shape=shape,
    )
    location_map_df = pd.read_csv(
        cache_path / "wi_db_computer_vision_location_map.csv",
        index_col="Project_Location",
    )
    species_map_df = pd.read_csv(cache_path / "wi_db_computer_vision_species_map.csv")
    taxon_map = mm_utils.get_taxon_map()

    feature_path = cache_path / "features"
    ids_all = np.load(
        feature_path / f"wi_blank_sat_features_{sat_backbone}_ids.npy",
        allow_pickle=True,
    )
    sat_features = np.load(feature_path / f"wi_blank_sat_features_{sat_backbone}.npy", allow_pickle=True)
    covariates = np.load(
        feature_path / f"wi_blank_sat_features_{sat_backbone}_covariates.npy",
        allow_pickle=True,
    )
    locs = np.load(feature_path / f"wi_blank_sat_features_{sat_backbone}_locs.npy", allow_pickle=True)

    latitudes_all = locs[:, 0]
    longitudes_all = locs[:, 1]
    project_ids = np.array([e.split("___")[0] for e in ids_all])
    mask_test = np.array(
        [pid in wildlife_insights_test_project_ids for pid in project_ids]
    )
    mask_train = ~mask_test

    train_coords = np.column_stack(
        [latitudes_all[mask_train], longitudes_all[mask_train]]
    )
    test_coords = np.column_stack([latitudes_all[mask_test], longitudes_all[mask_test]])
    dist_matrix = mm_utils.get_dist_matrix(train_coords, test_coords)
    too_close = dist_matrix.min(axis=0) < 10
    mask_train[mask_train] &= ~too_close

    site_idx_all = location_map_df["Location_Index"][ids_all].to_numpy()

    feature_bank = dict(
        sat=sat_features,
        covariates=covariates,
    )
    features_modalities = {
        modality: feature_bank[modality] for modality in sorted(modalities)
    }
    conus_boundary = mm_utils.get_conus_boundary() if apply_range_filter else None

    return BaseData(
        result_array=result_array,
        species_map_df=species_map_df,
        taxon_map=taxon_map,
        site_idx_all=site_idx_all,
        mask_train_base=mask_train,
        mask_test_base=mask_test,
        latitudes=latitudes_all,
        longitudes=longitudes_all,
        features_modalities=features_modalities,
        conus_boundary=conus_boundary,
    )


def _apply_range_filter(
    base: BaseData, scientific_name: str, common_name: str | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test masks clipped to a species range if available."""
    mask_train = base.mask_train_base.copy()
    mask_test = base.mask_test_base.copy()
    if base.conus_boundary is None:
        return mask_train, mask_test

    name_final = common_name if common_name else scientific_name
    try:
        range_map = mm_utils.get_species_range_cached(
            name_final, admin_level="admin1"  # type: ignore[arg-type]
        )
        if range_map.crs != base.conus_boundary.crs:  # type: ignore[union-attr]
            range_map = range_map.to_crs(base.conus_boundary.crs)  # type: ignore[union-attr]
        range_map = gpd.overlay(range_map, base.conus_boundary, how="intersection")  # type: ignore[arg-type]
    except ValueError as exc:
        print(
            f"[WARN] Could not obtain range map for {name_final}: {exc}. Using CONUS boundary."
        )
        range_map = base.conus_boundary

    train_points = gpd.points_from_xy(
        base.longitudes[mask_train], base.latitudes[mask_train]
    )
    train_gdf = gpd.GeoDataFrame(geometry=train_points, crs="EPSG:4326")
    if range_map.crs != train_gdf.crs:  # type: ignore[union-attr]
        range_map = range_map.to_crs(train_gdf.crs)  # type: ignore[union-attr]
    range_mask_train = train_gdf.intersects(range_map.union_all()).values  # type: ignore[arg-type]
    mask_train[mask_train] &= range_mask_train

    test_points = gpd.points_from_xy(
        base.longitudes[mask_test], base.latitudes[mask_test]
    )
    test_gdf = gpd.GeoDataFrame(geometry=test_points, crs="EPSG:4326")
    if range_map.crs != test_gdf.crs:  # type: ignore[union-attr]
        range_map = range_map.to_crs(test_gdf.crs)  # type: ignore[union-attr]
    range_mask_test = test_gdf.intersects(range_map.union_all()).values  # type: ignore[arg-type]
    mask_test[mask_test] &= range_mask_test
    return mask_train, mask_test


def _project_features(
    train: np.ndarray,
    test: np.ndarray,
    n_components: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Standardize and optionally reduce dimensionality via PCA."""
    combined = np.vstack([train, test])
    scaler = StandardScaler()
    combined = scaler.fit_transform(combined)

    if n_components <= 0:
        return combined[: train.shape[0]], combined[train.shape[0] :], np.nan

    n_components = int(
        min(n_components, combined.shape[1], max(2, combined.shape[0] - 1))
    )
    pca = PCA(n_components=n_components, random_state=rng.integers(0, 1_000_000))
    reduced = pca.fit_transform(combined)
    explained_variance = float(np.sum(pca.explained_variance_ratio_))
    return (
        reduced[: train.shape[0]].astype(np.float32, copy=False),
        reduced[train.shape[0] :].astype(np.float32, copy=False),
        explained_variance,
    )


def _compute_frechet_distance(train: np.ndarray, test: np.ndarray) -> float:
    """Fréchet distance between Gaussian fits to train/test embeddings."""
    if train.size == 0 or test.size == 0:
        return float("nan")
    mu1 = np.mean(train, axis=0)
    mu2 = np.mean(test, axis=0)
    sigma1 = np.cov(train, rowvar=False)
    sigma2 = np.cov(test, rowvar=False)
    # numerical stability
    eps = 1e-6
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
    cov_prod = sigma1 @ sigma2
    covmean = linalg.sqrtm(cov_prod)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    trace_term = np.trace(sigma1 + sigma2 - 2 * covmean)
    diff = mu1 - mu2
    return float(np.dot(diff, diff) + trace_term)


def _compute_mmd_rbf(
    train: np.ndarray,
    test: np.ndarray,
    rng: np.random.Generator,
    bandwidth_samples: int = 512,
) -> float:
    """RBF kernel MMD with bandwidth from the median heuristic."""
    if train.size == 0 or test.size == 0:
        return float("nan")
    if train.shape[0] == 1 or test.shape[0] == 1:
        return float("nan")

    def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
        dists = linalg.norm(x[:, None, :] - y[None, :, :], axis=2) ** 2
        return np.exp(-gamma * dists)

    sample_for_bw = np.vstack([train, test])
    if sample_for_bw.shape[0] > bandwidth_samples:
        sample_for_bw = sample_for_bw[
            rng.choice(sample_for_bw.shape[0], bandwidth_samples, replace=False)
        ]

    pairwise = (
        linalg.norm(sample_for_bw[:, None, :] - sample_for_bw[None, :, :], axis=2) ** 2
    )
    upper_tri = pairwise[np.triu_indices_from(pairwise, k=1)]
    median_dist = np.median(upper_tri[upper_tri > 0])
    if not np.isfinite(median_dist) or median_dist <= 0:
        gamma = 1.0
    else:
        gamma = 1.0 / (2.0 * median_dist)

    k_xx = _rbf_kernel(train, train, gamma)
    k_yy = _rbf_kernel(test, test, gamma)
    k_xy = _rbf_kernel(train, test, gamma)

    m = train.shape[0]
    n = test.shape[0]
    mmd = (
        (np.sum(k_xx) - np.trace(k_xx)) / (m * (m - 1))
        + (np.sum(k_yy) - np.trace(k_yy)) / (n * (n - 1))
        - 2 * np.mean(k_xy)
    )
    return float(max(mmd, 0.0))


def _compute_modality_shift(
    train: np.ndarray,
    test: np.ndarray,
    modality: str,
    pca_components: int,
    max_samples: int,
    rng: np.random.Generator,
) -> ShiftMetrics:
    """Compute domain shift metrics for a single modality."""
    train = np.asarray(train, dtype=np.float32, order="C")
    test = np.asarray(test, dtype=np.float32, order="C")
    train, test = _fill_missing(train, test)

    train_sampled, test_sampled = _sample_features(train, test, max_samples, rng)
    if train_sampled.shape[0] == 0 or test_sampled.shape[0] == 0:
        return ShiftMetrics(
            modality=modality,
            train_count=train_sampled.shape[0],
            test_count=test_sampled.shape[0],
            feature_dim=0,
            pca_variance=float("nan"),
            mean_distance=float("nan"),
            cov_frobenius=float("nan"),
            frechet_distance=float("nan"),
            mmd_rbf=float("nan"),
        )
    projected_train, projected_test, explained = _project_features(
        train_sampled, test_sampled, pca_components, rng
    )

    mean_distance = float(
        np.linalg.norm(
            np.mean(projected_train, axis=0) - np.mean(projected_test, axis=0)
        )
    )
    cov_frobenius = float(
        np.linalg.norm(
            np.cov(projected_train, rowvar=False)
            - np.cov(projected_test, rowvar=False),
            ord="fro",
        )
    )
    frechet = _compute_frechet_distance(projected_train, projected_test)
    mmd = _compute_mmd_rbf(projected_train, projected_test, rng)

    return ShiftMetrics(
        modality=modality,
        train_count=train_sampled.shape[0],
        test_count=test_sampled.shape[0],
        feature_dim=projected_train.shape[1],
        pca_variance=explained,
        mean_distance=mean_distance,
        cov_frobenius=cov_frobenius,
        frechet_distance=frechet,
        mmd_rbf=mmd,
    )


def _compute_detection_metrics(
    y_train: np.ndarray, y_test: np.ndarray
) -> DetectionMetrics:
    """Quantify differences in label distributions."""
    train_detection_rate = (
        float(np.nanmean(y_train)) if np.isfinite(y_train).any() else float("nan")
    )
    test_detection_rate = (
        float(np.nanmean(y_test)) if np.isfinite(y_test).any() else float("nan")
    )
    detection_rate_gap = (
        test_detection_rate - train_detection_rate
        if np.isfinite(train_detection_rate) and np.isfinite(test_detection_rate)
        else float("nan")
    )

    train_site_rates = np.nanmean(y_train, axis=1) if y_train.ndim > 1 else y_train
    test_site_rates = np.nanmean(y_test, axis=1) if y_test.ndim > 1 else y_test
    train_site_rates = train_site_rates[np.isfinite(train_site_rates)]
    test_site_rates = test_site_rates[np.isfinite(test_site_rates)]
    if train_site_rates.size and test_site_rates.size:
        detection_wasserstein = float(
            wasserstein_distance(train_site_rates, test_site_rates)
        )
    else:
        detection_wasserstein = float("nan")

    train_positive_site_fraction = float(
        np.mean(train_site_rates > 0) if train_site_rates.size else np.nan
    )
    test_positive_site_fraction = float(
        np.mean(test_site_rates > 0) if test_site_rates.size else np.nan
    )

    return DetectionMetrics(
        train_detection_rate=train_detection_rate,
        test_detection_rate=test_detection_rate,
        detection_rate_gap=detection_rate_gap,
        detection_wasserstein=detection_wasserstein,
        train_positive_site_fraction=train_positive_site_fraction,
        test_positive_site_fraction=test_positive_site_fraction,
    )


def _iter_records(
    species_ids: Iterable[str],
    base_data: BaseData,
    pca_components: int,
    max_samples: int,
    seed: int,
    apply_range_filter: bool,
) -> list[dict]:
    """Compute shift metrics for every species."""
    records: list[dict] = []
    rng = np.random.default_rng(seed)

    for taxon_id in tqdm(list(species_ids), desc="species"):
        species_row = base_data.species_map_df.loc[
            base_data.species_map_df["WI_taxon_id"] == taxon_id
        ]
        if species_row.empty:
            print(f"[WARN] Taxon id {taxon_id} not found in species map. Skipping.")
            continue
        taxon_idx = int(species_row["Species_Index"].values[0])
        scientific_name = species_row["Scientific_Name"].values[0]
        common_name = base_data.taxon_map.get(taxon_id)

        mask_train, mask_test = (
            _apply_range_filter(base_data, scientific_name, common_name)
            if apply_range_filter
            else (
                base_data.mask_train_base.copy(),
                base_data.mask_test_base.copy(),
            )
        )

        y_all = base_data.result_array[taxon_idx, base_data.site_idx_all]
        y_train_raw = y_all[mask_train]
        y_test_raw = y_all[mask_test]
        y_train = np.where(
            np.isnan(y_train_raw), np.nan, np.where(y_train_raw >= 1, 1.0, 0.0)
        )
        y_test = np.where(
            np.isnan(y_test_raw), np.nan, np.where(y_test_raw >= 1, 1.0, 0.0)
        )

        detection_metrics = _compute_detection_metrics(y_train, y_test)
        for modality in sorted(base_data.features_modalities):
            train_features = base_data.features_modalities[modality][mask_train]
            test_features = base_data.features_modalities[modality][mask_test]
            shift_metrics = _compute_modality_shift(
                train=train_features,
                test=test_features,
                modality=modality,
                pca_components=pca_components,
                max_samples=max_samples,
                rng=rng,
            )
            records.append(
                dict(
                    taxon_id=str(taxon_id),
                    taxon_idx=int(taxon_idx),
                    scientific_name=str(scientific_name),
                    common_name=common_name,
                    modality=shift_metrics.modality,
                    train_count=shift_metrics.train_count,
                    test_count=shift_metrics.test_count,
                    feature_dim=shift_metrics.feature_dim,
                    pca_variance=shift_metrics.pca_variance,
                    mean_distance=shift_metrics.mean_distance,
                    cov_frobenius=shift_metrics.cov_frobenius,
                    frechet_distance=shift_metrics.frechet_distance,
                    mmd_rbf=shift_metrics.mmd_rbf,
                    train_detection_rate=detection_metrics.train_detection_rate,
                    test_detection_rate=detection_metrics.test_detection_rate,
                    detection_rate_gap=detection_metrics.detection_rate_gap,
                    detection_wasserstein=detection_metrics.detection_wasserstein,
                    train_positive_site_fraction=(
                        detection_metrics.train_positive_site_fraction
                    ),
                    test_positive_site_fraction=(
                        detection_metrics.test_positive_site_fraction
                    ),
                    range_filter_applied=apply_range_filter,
                )
            )
    return records


def main(
    species_ids: str | Sequence[str] | None = None,
    modalities: Sequence[str] = ("sat", "covariates"),
    sat_backbone: str | None = None,
    pca_components: int = pca_dim,
    max_samples: int = 1500,
    seed: int = 0,
    output_path: str | None = None,
    apply_range_filter: bool | None = None,
):
    """Run domain shift quantification across species."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = (
        Path(output_path)
        if output_path is not None
        else OUTPUT_DIR / "domain_shift_metrics.csv"
    )

    species_list = _parse_species_ids(species_ids)
    if not species_list:
        raise ValueError("No species IDs provided.")
    modality_set = set(modalities)
    if not modality_set:
        raise ValueError("At least one modality must be specified.")

    sat_backbone_final = sat_backbone or default_sat_backbone
    range_filter = (
        limit_to_range if apply_range_filter is None else bool(apply_range_filter)
    )

    base_data = _prepare_base_data(
        modalities=modality_set,
        sat_backbone=sat_backbone_final,
        apply_range_filter=range_filter,
    )

    records = _iter_records(
        species_ids=species_list,
        base_data=base_data,
        pca_components=pca_components,
        max_samples=max_samples,
        seed=seed,
        apply_range_filter=range_filter,
    )

    df = pd.DataFrame.from_records(records)
    df.sort_values(["taxon_id", "modality"], inplace=True)
    df.to_csv(output_file, index=False)
    print(f"Saved domain shift metrics to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
