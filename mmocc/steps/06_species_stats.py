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
"""Compute statistics for each species in the dataset to determine suitability for
modeling."""

from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import fire
import pandas as pd
from tqdm import tqdm

from mmocc.config import cache_path
from mmocc.utils import compute_species_stats, cpu_count, get_submitit_executor

SPECIES_MAP_PATH = cache_path / "wi_db_computer_vision_species_map.csv"
OUTPUT_FILENAME = "species_stats.csv"


def load_species_ids(limit: Optional[int]) -> list[str]:
    species_df = pd.read_csv(SPECIES_MAP_PATH)
    taxon_ids = species_df["WI_taxon_id"].dropna().astype(str).unique().tolist()
    if limit is not None:
        taxon_ids = taxon_ids[:limit]
    return taxon_ids


def species_stats(
    limit: Optional[int] = None,
    num_workers: Optional[int] = None,
    output_filename: str = OUTPUT_FILENAME,
) -> Path:
    if limit is not None:
        limit = int(limit)
    if num_workers is not None:
        num_workers = int(num_workers)

    taxon_ids = load_species_ids(limit)
    if len(taxon_ids) == 0:
        raise RuntimeError("No species found in species map.")

    workers = num_workers or cpu_count()
    results = []
    with (
        Pool(processes=workers) as pool,
        tqdm(
            total=len(taxon_ids),
            desc="Processing species",
        ) as pbar,
    ):
        for result in pool.imap_unordered(compute_species_stats, taxon_ids):
            if result is not None:
                results.append(result)
            pbar.update()

    if not results:
        raise RuntimeError("No species statistics could be computed.")

    stats_df = pd.DataFrame(results)
    stats_df = stats_df.sort_values(
        by=["train_num_observations", "test_num_observations"],
        ascending=False,
    )

    output_path = cache_path / output_filename
    stats_df.to_csv(output_path, index=False)
    print(f"Wrote species stats for {len(stats_df)} species to {output_path}")
    return output_path


def main(
    limit: Optional[int] = None,
    num_workers: Optional[int] = None,
    output_filename: str = OUTPUT_FILENAME,
):
    executor = get_submitit_executor("species_stats")
    executor.update_parameters(
        slurm_mem="256G",
        slurm_additional_parameters=dict(gpus=0),
    )
    job = executor.submit(species_stats, limit, num_workers, output_filename)
    print(job.result())


if __name__ == "__main__":
    fire.Fire(main)
