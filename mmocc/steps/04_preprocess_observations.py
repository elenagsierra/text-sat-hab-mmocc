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
"""Aggregate and prepare observations for occupancy modeling."""

import gc
import os

import fire
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from mmocc.config import (
    cache_path,
    denylist_species_ids,
    max_latitude,
    max_longitude,
    min_latitude,
    min_longitude,
)
from mmocc.utils import get_submitit_executor


def preprocess_observations():
    df = pl.read_csv(cache_path / "wi_db_computer_vision.csv", try_parse_dates=True)

    # filter out the rows with no geographic coordinates
    df = df.filter(pl.col("Latitude").is_not_null() & pl.col("Longitude").is_not_null())

    # create scientific name column
    df = df.with_columns(
        (pl.col("Genus") + " " + pl.col("Species").cast(pl.Utf8)).alias(
            "Scientific_Name"
        ),
    )

    # filter to geographic area
    df = df.filter(
        (pl.col("Latitude") >= min_latitude)
        & (pl.col("Latitude") <= max_latitude)
        & (pl.col("Longitude") >= min_longitude)
        & (pl.col("Longitude") <= max_longitude)
    )

    # calculate species counts
    species_counts = (
        df.filter(~pl.col("WI_taxon_id").is_in(denylist_species_ids))
        .group_by("WI_taxon_id")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    # get all species
    available_species = species_counts["WI_taxon_id"].to_list()

    # only keep one entry per sequence
    df = df.unique(
        subset=["Project_Name", "Deployment_Location_ID", "sequence_id"], keep="first"
    )

    # concatenate Project_Name and stringified Deployment_Location_ID
    df = df.with_columns(
        (
            pl.col("Project_Name")
            + "___"
            + pl.col("Deployment_Location_ID").cast(pl.Utf8)
        ).alias("Project_Location")
    )

    # define our aggregation window
    aggregation_window = (
        "1w"  # Polars uses "1w" for weekly, "1mo" for monthly, "1d" for daily
    )

    # Filter out rows where Date_Time parsing failed
    df = df.filter(pl.col("Date_Time").is_not_null())

    # Add truncated datetime column to both dataframes
    df = df.with_columns(
        pl.col("Date_Time")
        .dt.truncate(every=aggregation_window)
        .alias("Date_Time_Truncated")
    )

    # Get unique dimensions and create mappings
    # Ensure they are sorted for consistent indexing
    unique_species = sorted(available_species)

    # Use original df for all locations, including those potentially without species observations
    unique_locations = sorted(df["Project_Location"].unique().to_list())

    # Calculate unique truncated times per location
    print("Calculating unique time steps per location...")
    location_times_map = {}
    max_time_steps = 0
    for location in tqdm(unique_locations, desc="Processing locations for time steps"):
        location_df = df.filter(pl.col("Project_Location") == location)
        # Ensure Date_Time_Truncated is not null and get sorted unique times
        unique_times_for_loc = sorted(
            location_df["Date_Time_Truncated"].unique().drop_nulls().to_list()
        )
        location_times_map[location] = {
            time: i for i, time in enumerate(unique_times_for_loc)
        }
        max_time_steps = max(max_time_steps, len(unique_times_for_loc))

    species_map = {species: i for i, species in enumerate(unique_species)}

    species_scientific_names = {}
    for species in unique_species:
        species_name = (
            df.filter(pl.col("WI_taxon_id") == species)["Scientific_Name"]
            .unique()
            .to_list()
        )
        if species_name:
            species_scientific_names[species] = species_name[0]
        else:
            species_scientific_names[species] = species

    location_map = {location: i for i, location in enumerate(unique_locations)}
    # time_map is now location specific, stored in location_times_map

    n_species = len(unique_species)
    n_locations = len(unique_locations)
    n_times = (
        max_time_steps  # Use the maximum number of time steps across all locations
    )

    print(
        f"Dimensions: Species={n_species}, Locations={n_locations}, Max Times per Location={n_times}"
    )

    # Initialize result array with NaNs
    # Use float32 to accommodate NaNs
    output_filename = os.path.join(
        cache_path / f"wi_db_computer_vision_occupancy_y.npy"
    )
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Initialize result array as a memory-mapped file using the final path
    # Use float32 to accommodate NaNs. Mode 'w+' creates or overwrites the file.
    result_array = np.memmap(
        output_filename,
        dtype=np.float32,
        mode="w+",
        shape=(n_species, n_locations, n_times),
    )

    # Initialize with NaNs, as memmap doesn't guarantee initial values
    result_array[:] = np.nan

    # --- Calculate species-specific counts ---
    print("Calculating species counts...")
    # Group by species, location, and the truncated time
    # Use the main filtered df directly
    species_counts_agg = df.group_by(
        "WI_taxon_id", "Project_Location", "Date_Time_Truncated"
    ).agg(
        pl.len().alias("observation_count")  # Count rows in each group
    )

    # Fill the array with species counts
    # Entries corresponding to inactive time steps or padding will remain NaN
    print("Filling array with species counts...")
    for row_tuple in tqdm(
        species_counts_agg.iter_rows(),
        total=len(species_counts_agg),
        desc="Filling counts",
    ):
        species, location, time, count = row_tuple
        # Check if keys exist in maps (species should always exist now)
        if (
            species in species_map
            and location in location_map
            and location in location_times_map
            and time in location_times_map[location]
        ):
            species_idx = species_map[species]
            location_idx = location_map[location]
            # Use the location-specific time index
            time_idx = location_times_map[location][time]
            # Ensure time_idx is within bounds (should be, due to max_time_steps)
            if time_idx < n_times:
                result_array[species_idx, location_idx, time_idx] = count

    # --- Identify active (location, time) pairs and set unobserved species to 0 ---
    # An active (location, time) pair means *some* observation occurred (of one of the top 10 species).
    # If one of the top 10 species wasn't observed during an active slot, its count should be 0, not NaN.
    print(
        "Identifying active location-time pairs and setting unobserved species to 0..."
    )
    # Use the main filtered df
    active_pairs_agg = (
        df.group_by("Project_Location", "Date_Time_Truncated")
        .agg(pl.lit(1).alias("active"))  # Dummy aggregation to identify unique pairs
        .drop_nulls("Date_Time_Truncated")
    )  # Ensure we don't include null time groups

    active_location_time_indices_count = 0
    for row_tuple in tqdm(
        active_pairs_agg.iter_rows(),
        total=len(active_pairs_agg),
        desc="Setting zeros for active slots",
    ):
        location, time, _ = row_tuple
        # Check if keys exist before processing
        if (
            location in location_map
            and location in location_times_map
            and time in location_times_map[location]
        ):
            location_idx = location_map[location]
            time_idx = location_times_map[location][time]
            if time_idx < n_times:
                active_location_time_indices_count += 1
                # For this active (location, time_idx), set any remaining NaNs (unobserved top 10 species) to 0
                species_slice = result_array[:, location_idx, time_idx]
                species_slice[np.isnan(species_slice)] = 0
                result_array[:, location_idx, time_idx] = (
                    species_slice  # Write back potentially modified slice
                )

    print(
        f"Processed {active_location_time_indices_count} active location-time pairs to set unobserved species counts to 0."
    )
    # Padded entries and genuinely inactive time slots (where no observation occurred at all for the location/time) remain NaN.

    # Clean up intermediate dataframes to free memory
    del species_counts_agg, active_pairs_agg, location_times_map  # Free map memory
    # del df_species # Not needed if df_species = df
    gc.collect()

    # Display the shape of the final array
    print(f"Final result_array shape: {result_array.shape}")

    # save location_map and species_map to CSV
    location_map_df = pd.DataFrame(
        list(location_map.items()), columns=["Project_Location", "Location_Index"]
    )
    species_map_df = []
    for k, v in species_map.items():
        species_map_df.append((k, v, species_scientific_names[k]))
    species_map_df = pd.DataFrame(
        species_map_df, columns=["WI_taxon_id", "Species_Index", "Scientific_Name"]
    )
    location_map_df.to_csv(
        cache_path / f"wi_db_computer_vision_location_map.csv", index=False
    )
    species_map_df.to_csv(
        cache_path / f"wi_db_computer_vision_species_map.csv", index=False
    )
    np.savetxt(
        os.path.join(cache_path / f"wi_db_computer_vision_occupancy_y_shape.txt"),
        np.array([n_species, n_locations, n_times]),
    )
    result_array.flush()


def main():
    executor = get_submitit_executor("preprocess_observations")
    executor.update_parameters(
        slurm_mem="256G", slurm_additional_parameters=dict(gpus=0)
    )
    job = executor.submit(preprocess_observations)
    print(job.result())


if __name__ == "__main__":
    fire.Fire(main)
