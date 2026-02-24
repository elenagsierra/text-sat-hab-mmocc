#!/usr/bin/env bash
# validate_cache_08_17.sh
set -euo pipefail

CACHE_PATH="${1:-${CACHE_PATH:-}}"
if [[ -z "${CACHE_PATH}" ]]; then
  echo "Usage: CACHE_PATH=/path/to/.cache $0"
  exit 2
fi

missing=0
check_file () {
  local p="$1"
  if [[ ! -f "$CACHE_PATH/$p" ]]; then
    echo "MISSING: $p"
    missing=1
  fi
  if [[ -f "$CACHE_PATH/$p" ]]; then
    echo "OK: $p"
  fi
}
check_glob () {
  local g="$1"
  shopt -s nullglob
  local arr=("$CACHE_PATH"/$g)
  shopt -u nullglob
  if [[ ${#arr[@]} -eq 0 ]]; then
    echo "MISSING: $g"
    missing=1
  fi
}

echo "Checking required inputs for steps 08-17 in: $CACHE_PATH"

# Core
check_file "wi_db_computer_vision_occupancy_y.npy"
check_file "wi_db_computer_vision_occupancy_y_shape.txt"
check_file "wi_db_computer_vision_location_map.csv"
check_file "wi_db_computer_vision_species_map.csv"
check_file "species_stats.csv"
check_file "wi_blank_images.pkl"

# Features needed by step 08 defaults
check_file "features/wi_blank_sat_features_alphaearth.npy"
check_file "features/wi_blank_sat_features_alphaearth_ids.npy"
check_file "features/wi_blank_sat_features_alphaearth_locs.npy"
check_file "features/wi_blank_sat_features_alphaearth_covariates.npy"
check_file "features/wi_blank_image_features_dinov2_vitb14.npy"
check_file "features/wi_blank_image_features_dinov2_vitb14_ids.npy"
check_file "features/wi_blank_image_features_dinov2_vitb14_locs.npy"
check_file "features/wi_blank_image_features_dinov2_vitb14_covariates.npy"

# Step 08 reads baseline fit results
check_glob "fit_results/*.pkl"

# Optional-but-needed later unless already generated
if [[ ! -f "$CACHE_PATH/visdiff_sat_descriptions.csv" ]]; then
  echo "NOTE: visdiff_sat_descriptions.csv missing (will be produced by step 08)"
fi
if [[ ! -f "$CACHE_PATH/expert_habitat_descriptions.csv" ]]; then
  echo "NOTE: expert_habitat_descriptions.csv missing (will be produced by step 09)"
fi

if [[ $missing -eq 0 ]]; then
  echo "OK: required files for starting steps 08-17 are present."
else
  echo "FAIL: missing required files."
  exit 1
fi
