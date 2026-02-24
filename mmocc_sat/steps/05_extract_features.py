#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "geogist",
#     "PyTorchWildlife==1.2.4.2",
#     "onnx2torch==1.5.15",
#     "open-clip-torch==3.2.0",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# geogist = { git = "https://github.com/timmh/geogist.git" }
# ///
"""Extract image features from blank camera trap images and satellite features from
corresponding locations."""

from datetime import datetime

import fire
import geogist
import numpy as np
import submitit
import torch
from tqdm import tqdm

from mmocc.config import (
    cache_path,
    image_batch_size,
    image_feature_dims,
    rs_model_kwargs,
    rs_scale,
    sat_batch_size,
    sat_feature_dims,
)
from mmocc.datasets.wildlifeinsights import WildlifeInsightsDataset
from mmocc.train_utils import get_backbone
from mmocc.utils import cpu_count, get_submitit_executor, memory

extract_embeddings_cached = memory.cache(geogist.extract_embeddings)


def extract_image_features(backbone_name: str):
    backbone = get_backbone(backbone_name)
    backbone.eval()

    dataset = WildlifeInsightsDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=image_batch_size,
        shuffle=False,
        num_workers=cpu_count(),
    )

    all_features = []
    loc_ids = []
    locs = []
    all_covariates = []
    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"]
            features = backbone(images)
            all_features.append(features.detach().cpu().numpy())
            loc_ids.append(batch["loc_id"])
            locs.append(
                np.stack(
                    [
                        batch["latitude"].flatten().cpu().numpy(),
                        batch["longitude"].flatten().cpu().numpy(),
                    ],
                    axis=1,
                )
            )
            all_covariates.append(batch["covariates"].detach().cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    loc_ids = np.concatenate(loc_ids, axis=0)
    locs = np.concatenate(locs, axis=0)
    all_covariates = np.concatenate(all_covariates, axis=0)
    output_path = cache_path / "features"
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / f"wi_blank_image_features_{backbone_name}.npy", all_features)
    np.save(output_path / f"wi_blank_image_features_{backbone_name}_ids.npy", loc_ids)
    np.save(output_path / f"wi_blank_image_features_{backbone_name}_locs.npy", locs)
    np.save(
        output_path / f"wi_blank_image_features_{backbone_name}_covariates.npy",
        all_covariates,
    )


def extract_sat_features(backbone_name: str):
    dataset = WildlifeInsightsDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=sat_batch_size,
        shuffle=False,
        num_workers=cpu_count(),
    )

    all_features = []
    loc_ids = []
    with torch.inference_mode():
        for batch in dataloader:
            datetimes = [
                datetime.fromtimestamp(dt)
                for dt in batch["datetime"].flatten().tolist()
            ]
            latitudes = batch["latitude"].flatten().tolist()
            longitudes = batch["longitude"].flatten().tolist()

            features = extract_embeddings_cached(
                latitudes=latitudes,  # type: ignore
                longitudes=longitudes,  # type: ignore
                datetimes=datetimes,  # type: ignore
                model=backbone_name,  # type: ignore
                scale=rs_scale,  # type: ignore
                cache_dir=cache_path / "geogist",  # type: ignore
                model_kwargs=rs_model_kwargs.get(backbone_name, {}),  # type: ignore
            )
            if len(features) != len(latitudes):  # type: ignore
                raise ValueError(
                    f"Expected {len(latitudes)} features but got {len(features)}"  # type: ignore
                )
            all_features.append(features)
            loc_ids.append(batch["loc_id"])

    all_features = np.concatenate(all_features, axis=0)
    loc_ids = np.concatenate(loc_ids, axis=0)
    output_path = cache_path / "features"
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / f"wi_blank_sat_features_{backbone_name}.npy", all_features)
    np.save(output_path / f"wi_blank_sat_features_{backbone_name}_ids.npy", loc_ids)


def main(skip_existing: bool = True):
    image_backbones = sorted(list(image_feature_dims.keys()))
    sat_backbones = sorted(list(sat_feature_dims.keys()))

    executor = get_submitit_executor("extract_features")
    executor.update_parameters(
        slurm_mem="256G",
        slurm_additional_parameters=dict(gpus=1),
    )
    jobs = []
    with executor.batch():
        for backbone_name in image_backbones:
            if (
                skip_existing
                and (
                    cache_path
                    / "features"
                    / f"wi_blank_image_features_{backbone_name}.npy"
                ).exists()
            ):
                print(
                    f"Image features for backbone {backbone_name} already exist, skipping."
                )
                continue
            jobs.append(executor.submit(extract_image_features, backbone_name))

        for backbone_name in sat_backbones:
            if (
                skip_existing
                and (
                    cache_path
                    / "features"
                    / f"wi_blank_sat_features_{backbone_name}.npy"
                ).exists()
            ):
                print(
                    f"Sat features for backbone {backbone_name} already exist, skipping."
                )
                continue
            jobs.append(executor.submit(extract_sat_features, backbone_name))

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
