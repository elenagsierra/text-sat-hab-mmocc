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
"""Filter camera trap images to remove invalid images (e.g., blank images)."""

from multiprocessing import Pool

import fire
import numpy as np
import pandas as pd

from mmocc.config import cache_path, wi_image_path
from mmocc.utils import cpu_count, get_submitit_executor, validate_image


def filter_data():
    df_blanks = pd.read_pickle(cache_path / "wi_blank_images_raw.pkl")

    keep = []
    image_paths = (
        df_blanks["FilePath"].str.replace("gs://", str(wi_image_path) + "/").tolist()
    )

    with Pool(processes=cpu_count()) as pool:
        keep = pool.map(validate_image, image_paths)

    print(f"Keeping {np.sum(keep)} / {len(keep)} images after filtering.")
    df_filtered = df_blanks[keep]
    df_filtered.to_pickle(cache_path / "wi_blank_images.pkl")


def main():
    executor = get_submitit_executor("filter_data")
    executor.update_parameters(
        slurm_mem="256G", slurm_additional_parameters=dict(gpus=0)
    )
    job = executor.submit(filter_data)
    print(job.result())


if __name__ == "__main__":
    fire.Fire(main)
