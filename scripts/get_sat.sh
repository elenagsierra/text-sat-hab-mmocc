#!/bin/sh
export CACHE_PATH="/data/vision/beery/scratch/esierra/text-sat-hab-mmocc/.cache2/.cache"
python3 scripts/build_sat_images.py
python3 scripts/sat_tif_to_png.py