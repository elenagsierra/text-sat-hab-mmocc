# scripts/convert_sat_tif_to_png.py
import os
from pathlib import Path
from PIL import Image
import numpy as np

CACHE_PATH = Path(os.environ["CACHE_PATH"])
src_dir = CACHE_PATH / "sat_images"
dst_dir = CACHE_PATH / "sat_images_png"
dst_dir.mkdir(parents=True, exist_ok=True)

def scale_to_uint8(arr):
    arr = arr.astype(np.float32)
    lo = np.nanpercentile(arr, 2)
    hi = np.nanpercentile(arr, 98)
    if hi <= lo:
        hi = lo + 1e-6
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)

count = 0
for tif in src_dir.glob("*.tif"):
    loc_id = tif.stem
    png = dst_dir / f"{loc_id}.png"
    if png.exists():
        continue

    with Image.open(tif) as im:
        arr = np.array(im)

    # Handle grayscale or multi-band
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]

    arr8 = scale_to_uint8(arr)
    Image.fromarray(arr8, mode="RGB").save(png, compress_level=4)
    count += 1

print(f"wrote {count} png files to {dst_dir}")
