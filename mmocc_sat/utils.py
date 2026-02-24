import json
import os
import re
import sys
from itertools import chain, combinations
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import submitit
import torch
from joblib import Memory
from PIL import Image
from rangepy import get_species_range  # type: ignore
from scipy.spatial.distance import cdist
from sklearn.impute import KNNImputer
from torchvision.transforms import v2

from mmocc_sat.config import (
    cache_path,
    default_image_backbone,
    default_sat_backbone,
    denylist_species_ids,
    focal_species_ids_v1,
    interesting_species_ids,
    limit_to_range,
    log_path,
    weights_path,
    wildlife_insights_test_project_ids,
)

memory = Memory(cache_path / "joblib")


get_species_range_cached = memory.cache(get_species_range)


def cpu_count():
    try:
        return os.process_cpu_count()
    except:
        return os.cpu_count() or 0


def unset_slurm_env_vars():
    for key in os.environ.keys():
        if key.startswith("SLURM_"):
            os.environ.pop(key)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


def get_submitit_executor(job_name: str = "mmocc") -> submitit.AutoExecutor:

    # remove SLURM environment variables to avoid issues with nested submitit jobs
    unset_slurm_env_vars()

    # run in-process if slurm is unconfigured or if debugging interactively
    debug = (
        (
            os.getenv("SUBMITIT_DEBUG") is not None
            and os.getenv("SUBMITIT_DEBUG", "").upper() not in {"", "0", "FALSE"}
        )
        or (os.getenv("SUBMITIT_SLURM_PARTITION") is None)
        or (sys.gettrace() is not None)
    )

    executor = submitit.AutoExecutor(
        folder=log_path, cluster="debug" if debug else None
    )
    executor.update_parameters(
        timeout_min=int(os.getenv("SUBMITIT_TIMEOUT_MIN", "10080")),
        slurm_job_name=job_name,
        slurm_partition=os.getenv("SUBMITIT_SLURM_PARTITION"),
        slurm_account=os.getenv("SUBMITIT_SLURM_ACCOUNT"),
        slurm_qos=os.getenv("SUBMITIT_SLURM_QOS"),
        slurm_mem=os.getenv("SUBMITIT_SLURM_MEM"),
        cpus_per_task=int(os.getenv("SUBMITIT_CPUS_PER_TASK", 1)),
        slurm_additional_parameters=dict(
            gpus=os.getenv("SUBMITIT_SLURM_GPUS", "1"),
            requeue=os.getenv("SUBMITIT_SLURM_REQUEUE", "1").upper()
            not in {"0", "FALSE"},
        ),
    )
    return executor


metadata_datetime_format = "%Y-%m-%dT%H:%M:%SZ"


def batch_collate_ignore_none(batch):
    return torch.utils.data.default_collate(
        [{k: v for k, v in b.items() if v is not None} for b in batch if b is not None]
    )


default_image_transform = v2.Compose(
    [
        v2.Resize(224),
        v2.CenterCrop(224),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

de_normalize_transform = v2.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def load_image(path, transform=default_image_transform):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if transform is not None:
            image = transform(image)
        return image


def load_sat(path, transform=default_image_transform):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if transform is not None:
            image = transform(image)
        return image


def maybe_to_item(x):
    return x.item() if isinstance(x, torch.Tensor) else x


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def parse_site_label(site_label):
    pattern = r"^([a-zA-Z]+)[^A-Za-z0-9]*([0-9]+)[^A-Za-z0-9]*([a-zA-Z0-9_-]*)$"
    matches = re.search(pattern, site_label)
    if matches is None:
        raise ValueError(f"Site label '{site_label}' does not match expected pattern")
    survey = matches.group(1)
    gridcell_idx = matches.group(2)
    modifier = matches.group(3) if len(matches.group(3)) > 0 else None
    if len(survey) == 0 or len(gridcell_idx) == 0:
        raise ValueError(f"Site label '{site_label}' does not match expected pattern")
    survey = re.sub(r"[^A-Z]", "", survey.upper())
    gridcell_idx = int(re.sub(r"[^0-9]", "", gridcell_idx))
    modifier = re.sub(r"[^a-z]", "", modifier.lower()) if modifier is not None else None
    return survey, gridcell_idx, modifier


def harmonize_site_label(site_label, include_modifier=True):
    survey, gridcell_idx, modifier = parse_site_label(site_label)
    gridcell_idx = f"{gridcell_idx:03d}"
    return "".join(
        [survey, gridcell_idx, modifier] if include_modifier else [survey, gridcell_idx]
    )


def harmonize_site_label_bais(site_label, include_modifier=True):
    audio_to_image_bais = {
        "AMBE": "Ambere",
        "CAPI": "Capitale",
        "IMBA": "Imbalanga",
        "LANGA": "Lango",
        "LOKO": "Lokoue",
        "MBOUEBE": "MbouEbe",
        "MOBAL": "Moba",
        "MOBAM": "Moba",
        "MOBAP": "Moba",
        "MOBAS": "Moba",
        "MOND": "Mondo",
        "MOUN": "Moungali",
    }
    if site_label.endswith("_Audiomoth"):
        return audio_to_image_bais[site_label.replace("_Audiomoth", "")]
    else:
        return site_label.split(" ")[0]


def get_taxon_map():
    taxonomy = pd.read_csv(
        weights_path / "speciesnet_4.0.1b" / "taxonomy_release.txt",
        sep=";",
        header=None,
        names=[
            "taxon_id",
            "class",
            "order",
            "family",
            "tribe",
            "species",
            "common_name",
        ],
    )
    taxon_map = taxonomy.set_index("taxon_id")["common_name"].to_dict()
    return taxon_map


def get_scientific_taxon_map() -> dict[str, str]:
    """Return a WI taxon_id -> scientific name map from the cached species map."""
    species_map = pd.read_csv(cache_path / "wi_db_computer_vision_species_map.csv")
    species_map = species_map.loc[:, ["WI_taxon_id", "Scientific_Name"]]
    species_map["WI_taxon_id"] = species_map["WI_taxon_id"].astype(str)
    species_map = species_map[species_map["Scientific_Name"].notna()]
    species_map["Scientific_Name"] = (
        species_map["Scientific_Name"].astype(str).str.strip()
    )
    species_map = species_map[species_map["Scientific_Name"] != ""]
    species_map = species_map.drop_duplicates(subset=["WI_taxon_id"])
    return species_map.set_index("WI_taxon_id")["Scientific_Name"].to_dict()


def experiment_to_filename(
    taxon_id, modalities, image_backbone_name, sat_backbone_name, suffix=None
):
    modalities_str = "_".join(sorted(list(modalities)))
    filename = f"{taxon_id}_modalities_{modalities_str}_image_{image_backbone_name}_sat_{sat_backbone_name}"
    if suffix is not None:
        filename += f".{suffix}"
    return filename


def filename_to_experiment(filename):
    pattern = r"^(?P<taxon_id>.+)_modalities_(?P<modalities>.+)_image_(?P<image_backbone_name>.+)_sat_(?P<sat_backbone_name>.+)$"
    matches = re.match(pattern, os.path.splitext(filename)[0])
    if matches is None:
        raise ValueError(f"Filename '{filename}' does not match expected pattern")
    taxon_id = matches.group("taxon_id")
    modalities = set(matches.group("modalities").split("_"))
    image_backbone_name = matches.group("image_backbone_name")
    sat_backbone_name = matches.group("sat_backbone_name")
    return taxon_id, modalities, image_backbone_name, sat_backbone_name


def validate_image(image_path_local):
    if not os.path.exists(image_path_local):
        print(f"Image not found: {image_path_local}")
        return False
    try:
        with Image.open(image_path_local) as image:
            image = np.array(image.convert("RGB"))
    except Exception:
        print(f"Error loading image: {image_path_local}")
        return False

    if np.mean(image) < 10:
        print(f"Image too dark: {image_path_local}")
        return False

    blue_ratio = np.mean(
        (image[:, :, 2] > 150) & (image[:, :, 0] < 150) & (image[:, :, 1] < 150)
    )
    if blue_ratio > 0.5:
        print(f"Image likely sky: {image_path_local}")
        return False

    return True


@memory.cache
def get_conus_boundary():
    conus_url = (
        "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"
    )
    excluded_states = {"AK", "HI", "PR", "GU", "VI", "AS", "MP"}
    conus_boundary = gpd.read_file(conus_url)
    conus_boundary = conus_boundary[
        ~conus_boundary["STUSPS"].isin(excluded_states)
    ].copy()
    if conus_boundary.crs is None:
        raise ValueError("CONUS boundary shapefile has no CRS defined")
    conus_boundary = conus_boundary.to_crs("EPSG:4326")
    return conus_boundary


@memory.cache
def get_dist_matrix(train_coords, test_coords):
    return cdist(
        test_coords, train_coords, lambda a, b: haversine_km(a[0], a[1], b[0], b[1])
    )


@memory.cache
def load_data(
    taxon_id: str,
    modalities: set[str],
    image_backbone_name: str | None = None,
    sat_backbone_name: str | None = None,
    impute: bool = True,
    limit_to_range_override: bool | None = None,
):
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

    taxon_map = get_taxon_map()
    taxon_idx = species_map_df.loc[
        species_map_df["WI_taxon_id"] == taxon_id, "Species_Index"
    ].values[0]
    scientific_name = species_map_df.loc[
        species_map_df["WI_taxon_id"] == taxon_id, "Scientific_Name"
    ].values[0]
    common_name = taxon_map.get(taxon_id)

    modalities_list = sorted(list(modalities))

    image_backbone_name_data = (
        image_backbone_name
        if image_backbone_name is not None
        else default_image_backbone
    )
    sat_backbone_name_data = (
        sat_backbone_name if sat_backbone_name is not None else default_sat_backbone
    )
    feature_path = cache_path / "features"
    ids_all = np.load(
        feature_path / f"wi_blank_image_features_{image_backbone_name_data}_ids.npy",
        allow_pickle=True,
    )
    image_features = np.load(
        feature_path / f"wi_blank_image_features_{image_backbone_name_data}.npy",
        allow_pickle=True,
    )
    sat_features = np.load(
        feature_path / f"wi_blank_sat_features_{sat_backbone_name_data}.npy",
        allow_pickle=True,
    )
    image_loc_ids = np.load(
        feature_path / f"wi_blank_image_features_{image_backbone_name_data}_ids.npy",
        allow_pickle=True,
    )
    sat_loc_ids = np.load(
        feature_path / f"wi_blank_sat_features_{sat_backbone_name_data}_ids.npy",
        allow_pickle=True,
    )
    locs = np.load(
        feature_path / f"wi_blank_image_features_{image_backbone_name_data}_locs.npy",
        allow_pickle=True,
    )
    covariates = np.load(
        feature_path
        / f"wi_blank_image_features_{image_backbone_name_data}_covariates.npy",
        allow_pickle=True,
    )
    assert (
        len(image_loc_ids) == image_features.shape[0]
    ), "Mismatch in image features and IDs"
    assert len(sat_loc_ids) == sat_features.shape[0], "Mismatch in sat features and IDs"
    assert len(locs) == image_features.shape[0], "Mismatch in locs and image features"
    assert (
        len(covariates) == image_features.shape[0]
    ), "Mismatch in covariates and image features"
    assert np.all(
        image_loc_ids == sat_loc_ids
    ), "Mismatch in location IDs between image and sat features"

    features_modalities = dict(
        image=image_features,
        sat=sat_features,
        covariates=covariates,
    )

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

    dist_matrix = get_dist_matrix(train_coords, test_coords)
    too_close = dist_matrix.min(axis=0) < 10
    mask_train[mask_train] &= ~too_close

    site_idx_all = location_map_df["Location_Index"][ids_all]
    y_all = result_array[taxon_idx, site_idx_all]

    apply_range_filter = (
        limit_to_range if limit_to_range_override is None else limit_to_range_override
    )

    if apply_range_filter:
        conus_boundary = get_conus_boundary()

        if common_name is None:
            if pd.isna(scientific_name):
                raise ValueError(f"Scientific name is NaN for taxon_id {taxon_id}")
            else:
                name_final = scientific_name
        else:
            name_final = common_name

        try:
            range_map = get_species_range_cached(name_final, admin_level="admin1")  # type: ignore

            # intersect with CONUS boundary to limit to US only
            if range_map.crs != conus_boundary.crs:  # type: ignore
                range_map = range_map.to_crs(conus_boundary.crs)  # type: ignore
            range_map = gpd.overlay(range_map, conus_boundary, how="intersection")  # type: ignore

        except ValueError as e:
            print(
                f"Could not get range map for species '{name_final}': {e}. Filtering to CONUS only."
            )
            range_map = conus_boundary

        # filter observations based on range map GeoPandas dataframe
        train_points = gpd.points_from_xy(
            longitudes_all[mask_train], latitudes_all[mask_train]
        )
        train_gdf = gpd.GeoDataFrame(geometry=train_points, crs="EPSG:4326")

        # Ensure range_map has the same CRS
        if range_map.crs != train_gdf.crs:
            range_map = range_map.to_crs(train_gdf.crs)  # type: ignore

        # Create spatial index for efficient intersection
        range_mask_train = train_gdf.intersects(range_map.unary_union).values

        # filter observations based on range map GeoPandas dataframe
        test_points = gpd.points_from_xy(
            longitudes_all[mask_test], latitudes_all[mask_test]
        )
        test_gdf = gpd.GeoDataFrame(geometry=test_points, crs="EPSG:4326")

        # Ensure range_map has the same CRS
        if range_map.crs != test_gdf.crs:
            range_map = range_map.to_crs(test_gdf.crs)  # type: ignore

        # Create spatial index for efficient intersection
        range_mask_test = test_gdf.intersects(range_map.unary_union).values

        mask_train[mask_train] &= range_mask_train
        mask_test[mask_test] &= range_mask_test

    y_train = y_all[mask_train]
    y_test = y_all[mask_test]

    y_train = np.where(np.isnan(y_train), np.nan, np.where(y_train >= 1, 1.0, 0.0))
    y_test = np.where(np.isnan(y_test), np.nan, np.where(y_test >= 1, 1.0, 0.0))

    y_train_naive = (y_train == 1).any(axis=1)
    y_test_naive = (y_test == 1).any(axis=1)

    # only keep requested modalities
    features_modalities = {
        modality: features_modalities[modality] for modality in modalities_list
    }

    # impute missing values in in all modalities
    if impute:
        for modality in modalities_list:
            features_modalities[modality][mask_train] = KNNImputer().fit_transform(
                features_modalities[modality][mask_train]
            )
            features_modalities[modality][mask_test] = KNNImputer().fit_transform(
                features_modalities[modality][mask_test]
            )

    return (
        shape,
        result_array,
        location_map_df,
        species_map_df,
        taxon_map,
        taxon_idx,
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
    )


def get_focal_species_ids() -> list[str]:
    df = pd.read_csv(cache_path / "species_stats.csv")
    feasible_species_ids = (
        df.loc[
            (
                (
                    df["train_num_sites_with_observations"] / df["train_num_sites"]
                    >= 0.05
                )
                & (
                    df["test_num_sites_with_observations"] / df["test_num_sites"]
                    >= 0.05
                )
            ),
            "taxon_id",
        ]
        .astype(str)
        .tolist()
    )

    # make sure v1 species are included and denylist species are excluded
    species_ids = sorted(
        list(
            (
                set(feasible_species_ids)
                | set(focal_species_ids_v1)
                | set(interesting_species_ids)
            )
            - set(denylist_species_ids)
        )
    )

    return species_ids


def summarize_split(labels: np.ndarray) -> Dict[str, float]:
    num_sites = labels.shape[0]
    mask = np.isfinite(labels) & (labels > 0)
    num_observations = int(mask.sum())
    if num_observations == 0:
        return dict(
            num_sites=num_sites,
            num_observations=0,
            num_sites_with_observations=0,
            naive_detection_prob=np.nan,
        )

    sites_with_obs = int(mask.any(axis=1).sum())
    positives = float(np.nansum(labels))
    detection_prob = positives / num_observations if num_observations > 0 else np.nan
    return dict(
        num_sites=num_sites,
        num_observations=num_observations,
        num_sites_with_observations=sites_with_obs,
        naive_detection_prob=detection_prob,
    )


def compute_species_stats(taxon_id: str) -> Optional[Dict]:
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
            _,
            _,
            _,
            y_train,
            y_test,
            _,
            _,
            _,
        ) = load_data(taxon_id, set())
    except Exception as exc:
        print(f"[WARN] Failed to process species {taxon_id}: {exc}")
        return None

    train_stats = summarize_split(y_train)
    test_stats = summarize_split(y_test)

    return dict(
        taxon_id=taxon_id,
        scientific_name=scientific_name,
        common_name=common_name,
        train_num_sites=train_stats["num_sites"],
        train_num_observations=train_stats["num_observations"],
        train_num_sites_with_observations=train_stats["num_sites_with_observations"],
        train_naive_detection_prob=train_stats["naive_detection_prob"],
        test_num_sites=test_stats["num_sites"],
        test_num_observations=test_stats["num_observations"],
        test_num_sites_with_observations=test_stats["num_sites_with_observations"],
        test_naive_detection_prob=test_stats["naive_detection_prob"],
    )


def run_biolith_in_process(queue, gpu_env_vars, *args_for_numpyro):

    from mmocc_sat.solvers.biolith import fit_biolith

    # Set GPU environment variables in spawned process
    for key, value in gpu_env_vars.items():
        os.environ[key] = value

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    result = fit_biolith(*args_for_numpyro)
    queue.put(result)
