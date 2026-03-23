# GLC_FCS30D Pixel-Level GRAFT

This experiment asks a narrower question than the main occupancy pipeline:
can pixel-level GRAFT patch tokens recover fine-grained land-use / land-cover
classes when the text side is only the raw GLC_FCS30D category names?

For now, it uses the already-downloaded sat_mmocc satellite PNGs and aligns
GLC_FCS30D labels to those image footprints. It does not attempt a continuous
map over larger extents.

## What It Reuses

- Cached sat_mmocc satellite PNGs under `CACHE_PATH`
- Shared GRAFT checkpoints and preprocessing from `mmocc.graft_utils`
- GLC_FCS30D imagery from the Google Earth Engine Community Catalog

## Assumptions

- Default imagery source is `sentinel`, since the cached Sentinel PNG footprint
  is the cleanest match to the 30 m GLC_FCS30D labels.
- Default land-cover year is `2022`.
- Text prompts are just the category names themselves, for example
  `"Open evergreen broadleaved forest"`, with no prompt template wrapped
  around them.
- Pixel-level comparison happens on the 14 x 14 GRAFT patch-token grid.

## Steps

1. Export aligned GLC_FCS30D labels:

```bash
uv run sat_mmocc/experiments/glc_fcs30d_graft/step_01_download_glc_fcs30d_labels.py   --imagery_source=sentinel   --year=2022   --max_sites=10   --project=YOUR_EE_PROJECT
```

2. Extract pixel-level GRAFT features from the cached satellite images:

```bash
uv run sat_mmocc/experiments/glc_fcs30d_graft/step_02_extract_pixel_graft_features.py   --imagery_source=sentinel   --year=2022   --max_sites=10
```

3. Score the land-cover class names against each patch token:

```bash
uv run sat_mmocc/experiments/glc_fcs30d_graft/step_03_score_lulc_text.py   --imagery_source=sentinel   --year=2022   --top_k=3
```

4. Build compact summaries and plots:

```bash
uv run sat_mmocc/experiments/glc_fcs30d_graft/step_04_analyze_results.py   --imagery_source=sentinel   --year=2022
```

## Outputs

Artifacts are written under:

```text
$CACHE_PATH/experiments/glc_fcs30d_graft/<imagery_source>/glc_year_<year>/graft_pixel/
```

Key outputs include:

- `glc_label_manifest.csv`
- `pixel_feature_manifest.csv`
- `text_scoring/site_metrics.csv`
- `text_scoring/confusion_matrix.csv`
- `text_scoring/per_class_metrics.csv`
- `text_scoring/*.png`

## Notes

- Per repo policy, run step 01 in the exploratory `nasa_roses` environment and
  run steps 02-04 in the locked `mmocc` environment.
- If you have local wrapper scripts for those environments, use them around the
  `uv run ...` commands above.
- The implementation is intentionally incremental and low-memory: most outputs
  are written per site instead of building a giant in-memory tensor.
