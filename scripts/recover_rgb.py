#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "joblib",
#     "pandas",
#     "Pillow",
# ]
#
# [tool.uv.sources]
# mmocc = { path = ".." }
# ///
"""Recover Sentinel-2 RGB images from the joblib cache and save them as
    {loc_id}.png in cache_path/sat_images_png/ — the layout expected by
    sat_mmocc/steps/08_visdiff.py.

Each cached entry's metadata.json carries the exact latitude, longitude, and
timestamp used when fetching.  We join those back to wi_blank_images.pkl on
(lat, lon) rounded to 4 decimal places to recover the loc_id.  If a loc_id
cannot be resolved the image is saved under its hash as a fallback so nothing
is lost.
"""
import glob
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from PIL import Image

from mmocc.config import cache_path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GRAFT_CACHE_DIR = cache_path / "joblib_graft" / "mmocc" / "rs_graft" / "fetch_sentinel_patch"
OUT_DIR = cache_path / "sat_images_png_recovered"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Build (rounded_lat, rounded_lon) → loc_id lookup from wi_blank_images.pkl
# ---------------------------------------------------------------------------
pkl_path = cache_path / "wi_blank_images.pkl"
print(f"Loading {pkl_path} for loc_id lookup ...")
df = pd.read_pickle(pkl_path)

_PRECISION = 4  # ~11 m — matches rs_graft thumbnail precision


def _parse_date_str(value) -> str:
    """Return the YYYY-MM-DD date string from a timestamp value."""
    if hasattr(value, "date"):
        return value.date().isoformat()
    return datetime.fromisoformat(str(value).strip("'")).date().isoformat()


def _key(lat, lon, ts=None) -> tuple:
    """(rounded_lat, rounded_lon[, date]) lookup key."""
    base = (round(float(lat), _PRECISION), round(float(lon), _PRECISION))
    return base + (_parse_date_str(ts),) if ts is not None else base


# Primary lookup: (lat, lon, date) → loc_id  — unambiguous
loc_lookup: dict[tuple, str] = {}
# Fallback lookup: (lat, lon) → loc_id  — used when timestamp is missing
loc_lookup_latlon: dict[tuple, str] = {}
for _, row in df.iterrows():
    lid = str(row["loc_id"])
    loc_lookup[_key(row["Latitude"], row["Longitude"], row["Date_Time"])] = lid
    loc_lookup_latlon.setdefault(_key(row["Latitude"], row["Longitude"]), lid)
print(f"  {len(loc_lookup)} (lat, lon, date) keys and "
      f"{len(loc_lookup_latlon)} (lat, lon) fallback keys loaded.")

# ---------------------------------------------------------------------------
# Recover images
# ---------------------------------------------------------------------------
cache_files = glob.glob(str(GRAFT_CACHE_DIR / "*" / "output.pkl"))
print(f"Found {len(cache_files)} cached entries in {GRAFT_CACHE_DIR}. Recovering...")

saved = skipped = failed = unmatched = 0

for file_path in cache_files:
    try:
        img_array = joblib.load(file_path)
        if img_array is None:
            skipped += 1
            continue

        cache_folder = Path(file_path).parent
        hash_id = cache_folder.name

        # Parse metadata to get lat/lon
        metadata_path = cache_folder / "metadata.json"
        loc_id: str | None = None
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            args = meta.get("input_args", {})
            raw_lat = str(args.get("latitude", "")).strip("'")
            raw_lon = str(args.get("longitude", "")).strip("'")
            raw_ts = str(args.get("timestamp", "")).strip("'")
            if raw_lat and raw_lon:
                # Try precise (lat, lon, date) match first
                if raw_ts:
                    loc_id = loc_lookup.get(_key(raw_lat, raw_lon, raw_ts))
                # Fall back to (lat, lon) only
                if loc_id is None:
                    loc_id = loc_lookup_latlon.get(_key(raw_lat, raw_lon))

        if loc_id is not None:
            dest = OUT_DIR / f"{loc_id}.png"
        else:
            # Fallback: keep hash-based name in the same output dir so it's
            # still findable, but note it won't be picked up by step 8.
            dest = OUT_DIR / f"unmatched_{hash_id}.png"
            unmatched += 1
            print(f"  WARNING: could not resolve loc_id for {hash_id}, saving as fallback.")

        if dest.exists():
            skipped += 1
            continue

        Image.fromarray(img_array).save(dest)
        saved += 1

    except Exception as e:
        print(f"  ERROR recovering {file_path}: {e}")
        failed += 1

print(
    f"\nDone.  {saved} saved, {skipped} skipped (None or already exist), "
    f"{unmatched} unmatched (no loc_id found), {failed} errors."
)
print(f"Images in: {OUT_DIR}")