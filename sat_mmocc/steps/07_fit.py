#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Fit naive and hierarchical occupancy models for each species, modality, and backbone
combination."""

import os
import pickle
from multiprocessing import Process, Queue

import fire
import numpy as np
import submitit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mmocc.config import (
    cache_path,
    default_image_backbone,
    default_sat_backbone,
    image_feature_dims,
    limit_to_range,
    pca_dim,
    sat_feature_dims,
)
from mmocc.solvers.logistic import fit_logistic
from mmocc.utils import (
    experiment_to_filename,
    get_focal_species_ids,
    get_submitit_executor,
    load_data,
    powerset,
    run_biolith_in_process,
)


def fit(
    taxon_id: str,
    modalities: set[str],
    image_backbone_name: str | None = None,
    sat_backbone_name: str | None = None,
):

    modalities_list = sorted(list(modalities))

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
    ) = load_data(taxon_id, modalities, image_backbone_name, sat_backbone_name)

    # compute naive "detection probability" as the ratio of presence observations at occupied sites
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

    # Standardize and PCA each modality
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

    species_results = dict(
        taxon_id=taxon_id,
        scientific_name=scientific_name,
        common_name=common_name,
        modalities=modalities_list,
        image_backbone=image_backbone_name,
        sat_backbone=sat_backbone_name,
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

    # Preserve GPU environment variables for spawned process
    gpu_env_vars = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("CUDA_", "SLURM_", "GPU_"))
    }
    regularization = "l1"  # "l1" or "l2"
    regressor_name = (
        "LinearRegression"  # "LinearRegression", "MLPRegression", or "BARTRegression"
    )
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
    except:
        p.terminate()
        raise RuntimeError("Biolith fitting process timed out.")
    finally:
        p.join()

    species_results.update(biolith_results)

    filename = experiment_to_filename(
        taxon_id, modalities_list, image_backbone_name, sat_backbone_name, "pkl"
    )
    fit_results_path = cache_path / "fit_results"
    fit_results_path.mkdir(parents=True, exist_ok=True)
    with open(fit_results_path / filename, "wb") as f:
        pickle.dump(species_results, f)


def main(skip_existing: bool = True):

    image_backbones = sorted(list(image_feature_dims.keys()))
    sat_backbones = sorted(list(sat_feature_dims.keys()))
    modalities = ["image", "sat", "covariates"]
    modalities_subsets = [set(m) for m in powerset(modalities) if len(m) > 0]

    executor = get_submitit_executor("fit")
    jobs = []
    with executor.batch():

        for taxon_id in get_focal_species_ids():
            for modalities_subset in modalities_subsets:

                modality_image_backbones = (
                    image_backbones if "image" in modalities_subset else [None]
                )
                modality_sat_backbones = (
                    sat_backbones if "sat" in modalities_subset else [None]
                )

                for image_backbone in modality_image_backbones:
                    for sat_backbone in modality_sat_backbones:

                        if (
                            image_backbone is not None
                            and image_backbone != default_image_backbone
                        ) and (
                            sat_backbone is not None
                            and sat_backbone != default_sat_backbone
                        ):
                            continue  # only experiment with one non-default backbone at a time

                        if skip_existing:
                            # skip if results already exist
                            filename = experiment_to_filename(
                                taxon_id,
                                modalities_subset,
                                image_backbone,
                                sat_backbone,
                                "pkl",
                            )
                            if (cache_path / "fit_results" / filename).exists():
                                print(
                                    f"Skipping existing results for taxon {taxon_id} with modalities {modalities_subset}, image_backbone {image_backbone}, sat_backbone {sat_backbone}"
                                )
                                continue

                        try:
                            jobs.append(
                                executor.submit(
                                    fit,
                                    taxon_id,
                                    modalities_subset,
                                    image_backbone_name=image_backbone,
                                    sat_backbone_name=sat_backbone,
                                )
                            )
                        except Exception as e:
                            print(
                                f"Error fitting taxon {taxon_id} with modalities {modalities}, image_backbone {image_backbone}, sat_backbone {sat_backbone}, limit_to_range {limit_to_range}: {e}"
                            )

    for job in tqdm(
        submitit.helpers.as_completed(jobs), total=len(jobs), desc="Jobs", leave=False
    ):
        try:
            job.result()
            print(f"Got results for job {job.job_id}")
        except Exception as e:
            print(f"Job {job.job_id} failed with error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    fire.Fire(main)
