# Yichao Instance Pairs

This pipeline segments every Yichao brightfield frame, saves per-image segmentation intermediates, and exports paired instance crops for later dataset packing.

Brightfield / fluorescence mapping used here:

- `c1` = brightfield
- `c0` = fluorescence

Default output root:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs`

Main scripts:

- `run_yichao_instance_pair_extraction.py`
- `build_yichao_instance_pair_database.py`
- `backfill_yichao_instance_metadata.sh`
- `plot_yichao_instance_size_histograms.py`
- `resume_yichao_instance_pairs_full_tmux.sh`
- `run_yichao_dataset_incremental_pipeline.sh`
- `resume_yichao_dataset_incremental_pipeline_tmux.sh`

Output layout:

- `images/<dataset>/<object>/<image_stem>/`
- `instances/<dataset>/<object>/<image_stem>/instance_0001/`
- `manifests/image_records.csv`
- `manifests/instance_records.csv`
- `manifests/summary.json`
- `database/instance_pairs.sqlite`

The rebuilt instance database now includes filter-ready fields such as:

- `is_edge_padded`
- `source_image_width_px`
- `source_image_height_px`
- `crop_area_px`
- `area_px_percentile`
- `area_px_quantile_level_20`
- `area_px_within_middle_90`
- `area_px_within_middle_95`
- `square_crop_size_px_percentile`
- `crop_area_px_percentile`

Each image folder contains:

- `brightfield_input.jpg`
- `fluorescence_reference.jpg`
- `debug_signal.png`
- `support.png`
- `multiscale_mask_16bit.png`
- `multiscale_instance_rgb.png`
- `multiscale_overlay_on_brightfield.png`
- `multiscale_overlay_on_fluorescence.png`
- `comparison_panel.png`
- `image_record.json`

Each instance folder contains:

- `brightfield_crop.png`
- `fluorescence_crop.png`
- `mask_crop.png`
- `overlay_on_brightfield_crop.png`
- `overlay_on_fluorescence_crop.png`
- `instance_rgb_crop.png`
- `instance_record.json`

The segmentation policy is intentionally simple for Yichao:

- run multiscale Cellpose on the brightfield image
- merge overlapping Cellpose masks across diameters
- only use the threshold/signal recovery branch as a fallback when Cellpose finds no candidates

Example:

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
bash analysis-tools/yichao_instance_pairs/run_yichao_instance_pair_extraction.sh --gpu true
bash analysis-tools/yichao_instance_pairs/build_yichao_instance_pair_database.sh
```

Resume in `tmux` after an interruption:

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
bash analysis-tools/yichao_instance_pairs/resume_yichao_instance_pairs_full_tmux.sh
```

Process one new Yichao LIF end to end and append it into the combined Yichao outputs:

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
bash analysis-tools/yichao_instance_pairs/run_yichao_dataset_incremental_pipeline.sh \
  --dataset-name Data-Yichao-5 \
  --lif-path /home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1/Data-Yichao-5/N39_TriRep_DF_3.lif
```

Run that incremental append flow in `tmux`:

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
bash analysis-tools/yichao_instance_pairs/resume_yichao_dataset_incremental_pipeline_tmux.sh \
  --dataset-name Data-Yichao-5 \
  --lif-path /home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1/Data-Yichao-5/N39_TriRep_DF_3.lif
```

Resume behavior:

- existing images are only skipped if the image record and all expected instance crops are present
- partial image folders are deleted and reprocessed
- `run_progress.json` is updated during the run
- the combined database and CSV manifests are rebuilt after extraction completes

Backfill the current instance database and resized metadata without touching the existing image files:

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
bash analysis-tools/yichao_instance_pairs/backfill_yichao_instance_metadata.sh
```
