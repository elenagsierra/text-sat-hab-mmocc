import numpy as np
import pandas as pd

from mmocc.config import cache_path

with open(cache_path / "final_species_ids.txt", "r") as f:
    taxon_ids = f.read().strip().split(",")

df = pd.read_csv(
    cache_path / "habitat_explainability" / "habitat_rs_explainability_summary.csv"
)

df = df[df["taxon_id"].isin(taxon_ids)]
print(f"Clipped mean: {np.nanmean(df['mean_r2_sat'].clip(lower=0)):.4f}")
print(f"Unclipped mean: {np.nanmean(df['mean_r2_sat']):.4f}")
