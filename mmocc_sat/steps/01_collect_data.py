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
"""Collect and preprocess Wildlife Insights data."""

import gc
import glob

import fire
import pandas as pd
from tqdm import tqdm

from mmocc.config import (
    cache_path,
    species_id_blank,
    wi_metadata_path,
    wildlife_insights_exclude_project_ids,
)
from mmocc.utils import get_submitit_executor


def collect_data():
    df = None
    for f in tqdm(
        sorted(glob.glob(str(wi_metadata_path / "wi_db_computer_vision_*.csv")))
    ):
        if df is None:
            df = pd.read_csv(f)
        else:
            df = pd.concat([df, pd.read_csv(f)], ignore_index=True)
    gc.collect()

    if df is None:
        raise ValueError("No observation files found.")

    df["Date_Time"] = pd.to_datetime(
        df["Date_Time"].str.replace(" ", "T").str.replace("Z", "") + "Z",
        format="mixed",
        errors="raise",
    )

    df["loc_id"] = df["Project_Name"] + "___" + df["Deployment_Location_ID"].astype(str)

    # remove invalid geographic coordinates
    df = df[df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)]

    df.to_csv(cache_path / "wi_db_computer_vision.csv", index=False)

    # extract blank images
    df_blanks = df[df["WI_taxon_id"] == species_id_blank]

    # only keep non-excluded projects
    df_blanks = df_blanks[
        ~df_blanks["Project_Name"].isin(wildlife_insights_exclude_project_ids)
    ]

    # filter df_blanks to include only DateTimes between 9am and 3pm
    df_blanks = df_blanks[
        (df_blanks["Date_Time"].dt.hour >= 9) & (df_blanks["Date_Time"].dt.hour <= 15)
    ]

    # keep one image per location closest to noon and closest to the median date
    def time_distance_to_noon(dt):
        noon = dt.replace(hour=12, minute=0, second=0, microsecond=0)
        return abs((dt - noon).total_seconds())

    def date_distance_to_median(dt, median_date):
        median_dt = median_date.replace(hour=0, minute=0, second=0, microsecond=0)
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return abs((dt - median_dt).days)

    rows_to_keep = []
    for _, group in df_blanks.groupby("loc_id"):
        median_date = group["Date_Time"].median()
        group = group.copy()
        group["time_dist"] = group["Date_Time"].apply(time_distance_to_noon)
        group["date_dist"] = group["Date_Time"].apply(
            lambda dt: date_distance_to_median(dt, median_date)
        )
        group = group.sort_values(by=["date_dist", "time_dist"])
        rows_to_keep.append(group.iloc[0])

    df_blanks = pd.DataFrame(rows_to_keep)

    # keep only relevant columns
    df_blanks = df_blanks[
        [
            "Project_Name",
            "Deployment_Location_ID",
            "WI_taxon_id",
            "Date_Time",
            "FilePath",
            "Image_format",
            "Latitude",
            "Longitude",
            "Country",
            "loc_id",
        ]
    ]

    df_blanks.to_pickle(cache_path / "wi_blank_images_raw.pkl")


def main():
    executor = get_submitit_executor("collect_data")
    executor.update_parameters(
        slurm_mem="256G", slurm_additional_parameters=dict(gpus=0)
    )
    job = executor.submit(collect_data)
    print(job.result())


if __name__ == "__main__":
    fire.Fire(main)
