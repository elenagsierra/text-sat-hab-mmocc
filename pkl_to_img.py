import os
from pathlib import Path

import pandas as pd
from PIL import Image

from mmocc_sat.config import cache_path
from mmocc_sat.rs_graft import fetch_sentinel_patch

def to_iso(ts):
    if hasattr(ts, "to_pydatetime"):
        return ts.to_pydatetime().isoformat()
    return pd.to_datetime(ts, utc=True).isoformat()

df = pd.read_pickle(cache_path / "wi_blank_images.pkl").sort_values("Date_Time")
df = df.drop_duplicates("loc_id").reset_index(drop=True)

out_dir = cache_path / "rs_images"
out_dir.mkdir(parents=True, exist_ok=True)

written = missing = failed = skipped = 0
for _, row in df.iterrows():
    loc_id = str(row["loc_id"])
    out = out_dir / f"{loc_id}.png"
    if out.exists() and out.stat().st_size > 0:
        skipped += 1
        continue
    try:
        patch = fetch_sentinel_patch(
            latitude=float(row["Latitude"]),
            longitude=float(row["Longitude"]),
            timestamp=to_iso(row["Date_Time"]),
            window_days=60,
            size=224,
            pixel_size_meters=10.0,
            cloud_percent=20.0,
            project="multimodal-sdm-473820",
            dataset="COPERNICUS/S2_SR_HARMONIZED",
        )
        if patch is None:
            missing += 1
            continue
        Image.fromarray(patch.astype("uint8"), mode="RGB").save(out, format="PNG")
        written += 1
    except Exception:
        failed += 1

print(dict(written=written, missing=missing, failed=failed, skipped=skipped))
