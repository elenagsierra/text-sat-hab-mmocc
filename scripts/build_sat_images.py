import os
from datetime import timedelta
from pathlib import Path

import ee
import geemap
import pandas as pd

CACHE_PATH = Path(os.environ["CACHE_PATH"])
OUT = CACHE_PATH / "sat_images"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 90
SCALE_METERS = 10
PATCH_SIZE_PX = 256
MAX_CLOUD = 20
MAX_SITES = None
EE_PROJECT = "multimodal-sdm-473820"

ee.Initialize(project=EE_PROJECT)


def pick_rgb_image(lat: float, lon: float, t: pd.Timestamp):
    pt = ee.Geometry.Point([lon, lat])

    # 1) Sentinel-2 strict window around timestamp
    start = (t - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    end = (t + timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    s2_strict = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(pt)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    if s2_strict.size().getInfo() > 0:
        img = ee.Image(s2_strict.first())
        return img.select(["B4", "B3", "B2"]).visualize(min=0, max=3000), "s2_strict"

    # 2) Sentinel-2 wide window (helps sparse regions/timestamps)
    start_wide = (t - timedelta(days=365)).strftime("%Y-%m-%d")
    end_wide = (t + timedelta(days=365)).strftime("%Y-%m-%d")
    s2_wide = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(pt)
        .filterDate(start_wide, end_wide)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    if s2_wide.size().getInfo() > 0:
        img = ee.Image(s2_wide.first())
        return img.select(["B4", "B3", "B2"]).visualize(min=0, max=3000), "s2_wide"

    # 3) Sentinel-2 median composite (no timestamp matching)
    s2_median = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(pt)
        .filterDate("2017-01-01", "2024-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
    )
    if s2_median.size().getInfo() > 0:
        img = s2_median.median()
        return img.select(["B4", "B3", "B2"]).visualize(min=0, max=3000), "s2_median"

    # 4) Landsat 8/9 fallback (covers 2013+)
    l89 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterBounds(pt)
        .filterDate(start_wide, end_wide)
        .filter(ee.Filter.lt("CLOUD_COVER", 80))
        .sort("CLOUD_COVER")
    )
    if l89.size().getInfo() > 0:
        img = ee.Image(l89.first())
        red = img.select("SR_B4").multiply(0.0000275).add(-0.2)
        green = img.select("SR_B3").multiply(0.0000275).add(-0.2)
        blue = img.select("SR_B2").multiply(0.0000275).add(-0.2)
        rgb = ee.Image.cat([red, green, blue]).visualize(min=0.03, max=0.35)
        return rgb, "landsat89"

    # 5) Landsat 5/7 fallback for older years (e.g., 2008)
    l57 = (
        ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"))
        .filterBounds(pt)
        .filterDate(start_wide, end_wide)
        .filter(ee.Filter.lt("CLOUD_COVER", 80))
        .sort("CLOUD_COVER")
    )
    if l57.size().getInfo() > 0:
        img = ee.Image(l57.first())
        red = img.select("SR_B3").multiply(0.0000275).add(-0.2)
        green = img.select("SR_B2").multiply(0.0000275).add(-0.2)
        blue = img.select("SR_B1").multiply(0.0000275).add(-0.2)
        rgb = ee.Image.cat([red, green, blue]).visualize(min=0.03, max=0.35)
        return rgb, "landsat57"

    return None, None


df = pd.read_pickle(CACHE_PATH / "wi_blank_images.pkl").sort_values("Date_Time")
df = df.drop_duplicates("loc_id")
if MAX_SITES is not None:
    df = df.head(MAX_SITES)

for _, row in df.iterrows():
    loc_id = str(row["loc_id"])
    out_tif = OUT / f"{loc_id}.tif"
    if out_tif.exists() and out_tif.stat().st_size > 0:
        continue

    lat = float(row["Latitude"])
    lon = float(row["Longitude"])
    t = pd.to_datetime(row["Date_Time"])

    pt = ee.Geometry.Point([lon, lat])
    region = pt.buffer((PATCH_SIZE_PX * SCALE_METERS) / 2).bounds()

    rgb, source = pick_rgb_image(lat, lon, t)
    if rgb is None:
        print("skip (no imagery in S2/Landsat fallbacks)", loc_id)
        continue

    try:
        geemap.ee_export_image(
            rgb,
            filename=str(out_tif),
            scale=SCALE_METERS,
            region=region,
            file_per_band=False,
        )
        if out_tif.exists() and out_tif.stat().st_size > 0:
            print(f"wrote {out_tif} via {source}")
        else:
            print("failed (empty output)", loc_id)
    except Exception as exc:
        print("failed", loc_id, exc)
