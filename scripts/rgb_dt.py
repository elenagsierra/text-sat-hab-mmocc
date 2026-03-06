import pandas as pd
from PIL import Image
from pathlib import Path
from datetime import datetime

from mmocc.config import cache_path
from mmocc.rs_graft import fetch_sentinel_patch, DEFAULT_TIME_WINDOW_DAYS

# 1. Load the dataset that contains the location IDs and coordinates
df = pd.read_pickle(cache_path / "wi_blank_images.pkl")

# 2. Create an output directory for the named images
output_dir = Path("sentinel_images_by_location")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Checking cache for {len(df)} locations...")

for idx, row in df.iterrows():
    loc_id = row["loc_id"]
    lat = float(row["Latitude"])
    lon = float(row["Longitude"])
    
    # Format the timestamp exactly as it was done during extraction
    timestamp = row["Date_Time"]
    if hasattr(timestamp, "to_pydatetime"):
        ts = timestamp.to_pydatetime().isoformat()
    else:
        ts = datetime.fromisoformat(str(timestamp)).isoformat()
        
    try:
        # 3. Call the cached function. 
        # Since this was already run, joblib intercepts it and instantly loads the array from disk.
        patch = fetch_sentinel_patch(
            latitude=lat,
            longitude=lon,
            timestamp=ts,
            window_days=DEFAULT_TIME_WINDOW_DAYS, 
            # Note: Ensure these arguments match EXACTLY what you used in 16_extract_graft_features.py
        )
        
        if patch is not None:
            # 4. Save the image with a meaningful name (e.g., the location ID)
            img = Image.fromarray(patch)
            
            # Example filename: ProjectName___DeploymentID_lat_lon.png
            safe_ts = ts.replace(":", "-")
            filename = f"{loc_id}_{lat:.4f}_{lon:.4f}_{safe_ts}.png"
            
            img.save(output_dir / filename)
            
    except Exception as e:
        print(f"Failed to process location {loc_id}: {e}")

print(f"Done! Images saved to {output_dir}")