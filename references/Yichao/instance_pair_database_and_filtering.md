# Yichao Instance-Pair Database and Filtering Notes

This note documents the current instance-pair database, the resized `256x256` pair dataset metadata, the edge-padding flag, and the current size distributions for the full Yichao `1/2/3/4/5` dataset.


## Live Paths

Original instance-pair outputs:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs/database/instance_pairs.sqlite`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs/manifests/image_records.csv`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs/manifests/instance_records.csv`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs/manifests/summary.json`

Resized `256x256` pair dataset:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs_resized_256`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs_resized_256/metadata/pairs_manifest.csv`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs_resized_256/metadata/resized_pairs.sqlite`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs_resized_256/metadata/summary.json`

Updated size-analysis outputs:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_size_analysis`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_size_analysis/quantiles.json`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_size_analysis/square_crop_size_px_histogram_triptych.png`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_size_analysis/area_px_histogram_triptych.png`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_size_analysis/crop_area_px_histogram_triptych.png`

Older histogram kept for reference:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/outputs/yichao_instance_square_crop_size_histogram.png`


## Current Totals

Whole-dataset counts after Yichao-5:

- images: `29,072`
- instances: `127,435`

Per-dataset instance counts:

- `Data-Yichao-1`: `18`
- `Data-Yichao-2`: `4,401`
- `Data-Yichao-3`: `69,463`
- `Data-Yichao-4`: `28,258`
- `Data-Yichao-5`: `25,295`


## Channel Mapping Used Here

For the segmentation and instance-pair pipeline:

- `c1` = brightfield
- `c0` = fluorescence


## Database Fields Added for Filtering

The original instance database and the resized `256x256` metadata database now both include:

- `is_edge_padded`
- `source_image_width_px`
- `source_image_height_px`
- `crop_area_px`
- `area_px_percentile`
- `area_px_quantile_level_20`
- `area_px_within_middle_90`
- `area_px_within_middle_95`
- `square_crop_size_px_percentile`
- `square_crop_size_px_quantile_level_20`
- `square_crop_size_px_within_middle_90`
- `square_crop_size_px_within_middle_95`
- `crop_area_px_percentile`
- `crop_area_px_quantile_level_20`
- `crop_area_px_within_middle_90`
- `crop_area_px_within_middle_95`


## Meaning of `is_edge_padded`

`is_edge_padded = 1` means the square crop extends outside the original image boundary and had to be padded during crop extraction.

Equivalent logic:

- `crop_x < 0`
- or `crop_y < 0`
- or `crop_x + crop_w > source_image_width_px`
- or `crop_y + crop_h > source_image_height_px`

Current edge-padded count in the full database:

- edge-padded instances: `51,380`
- edge-padded fraction: `40.32%`


## Recommended Filtering Fields

If the goal is to remove likely debris and unstable border crops while keeping the database intact:

- remove border-padded crops: `is_edge_padded = 0`
- remove the smallest `5%` by segmented object size: `area_px_percentile >= 0.05`
- keep the middle `90%` by segmented object size: `area_px_within_middle_90 = 1`
- keep the middle `95%` by segmented object size: `area_px_within_middle_95 = 1`

For debris filtering, prefer:

- `area_px`

Do not use `square_crop_size_px` as the main debris criterion, because it includes context padding around the instance.


## Current Quantiles

### `area_px`

- `p00 = 1501`
- `p01 = 1839`
- `p025 = 2434`
- `p05 = 3198.7`
- `p10 = 4269`
- `p25 = 6898`
- `p50 = 10804`
- `p75 = 18048`
- `p90 = 28694.6`
- `p95 = 40952.9`
- `p975 = 48953`
- `p99 = 66024.96`
- `p100 = 267055`

Practical ranges:

- middle `90%`: `3198.7` to `40952.9`
- middle `95%`: `2434` to `48953`


### `square_crop_size_px`

- `p00 = 68`
- `p01 = 81`
- `p025 = 91`
- `p05 = 100`
- `p10 = 112`
- `p25 = 126`
- `p50 = 157`
- `p75 = 197`
- `p90 = 244`
- `p95 = 288`
- `p975 = 313`
- `p99 = 365`
- `p100 = 768`

Practical ranges:

- middle `90%`: `100` to `288`
- middle `95%`: `91` to `313`


### `crop_area_px = crop_w * crop_h`

- `p00 = 4624`
- `p01 = 6561`
- `p025 = 8281`
- `p05 = 10000`
- `p10 = 12544`
- `p25 = 15876`
- `p50 = 24649`
- `p75 = 38809`
- `p90 = 59536`
- `p95 = 82944`
- `p975 = 97969`
- `p99 = 133225`
- `p100 = 589824`

Practical ranges:

- middle `90%`: `10000` to `82944`
- middle `95%`: `8281` to `97969`


## Peak Values

Using the full upgraded instance database:

### Peak `area_px`

- exact mode: `7155 px`
- histogram peak bin: about `5926.9` to `8139.85 px`


### Peak `square_crop_size_px`

- exact mode: `123 px`
- histogram peak bin: about `120.5` to `126.33 px`

Interpretation:

- the most typical segmented object area is around `7.1k px`
- the most typical square crop size is around `123 px`


## Histograms

The new histogram folder contains three triptych plots:

- full distribution
- lower-tail zoom
- upper-tail zoom

Files:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_size_analysis/square_crop_size_px_histogram_triptych.png`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_size_analysis/area_px_histogram_triptych.png`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_size_analysis/crop_area_px_histogram_triptych.png`

The `crop_area_px` plot is only for inspection right now. It is not yet used as the main filter.


## Maintenance Scripts

Backfill the current databases and metadata in place:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-tools/yichao_instance_pairs/backfill_yichao_instance_metadata.sh`

Plot the updated histograms:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-tools/yichao_instance_pairs/plot_yichao_instance_size_histograms.py`

The future incremental pipeline will automatically keep these new fields in new Yichao data:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-tools/yichao_instance_pairs/run_yichao_instance_pair_extraction.py`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-tools/yichao_instance_pairs/build_yichao_instance_pair_database.py`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/differentiation_prediction/yichao_instance_pairs_256/prepare_dataset.py`
