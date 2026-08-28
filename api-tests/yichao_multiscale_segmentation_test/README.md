# Yichao Multiscale Segmentation Test

This test runs the transplanted Zhengyu multiscale Cellpose pipeline on one selected brightfield `c1` image from each Yichao dataset:

- `Data-Yichao-1`
- `Data-Yichao-2`
- `Data-Yichao-3`
- `Data-Yichao-4`

For Yichao, the test now uses a simpler policy:

- run multiscale Cellpose on the brightfield image
- merge overlapping Cellpose masks across diameters
- only use the threshold/signal recovery branch as a fallback when Cellpose finds no candidates

The runner reuses the multiscale merge logic from:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-tools/app80_first_replicate_multiscale_cellpose/run_multiscale_dateaware_cellpose.py`

Default output root:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_multiscale_segmentation_test`

Run it with:

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
api-tests/yichao_multiscale_segmentation_test/run_yichao_multiscale_segmentation_test.sh
```

Per-dataset outputs include:

- `brightfield_input.png`
- `fluorescence_reference.png`
- `source.png`
- `signal.png`
- `support.png`
- `multiscale_mask_16bit.png`
- `multiscale_instance_rgb.png`
- `multiscale_overlay.png`
- `multiscale_overlay_on_brightfield.png`
- `multiscale_overlay_on_fluorescence.png`
- `comparison_panel.png`
- `multiscale_stats.json`

The root run folder also contains:

- `yichao_test_summary.json`
- `yichao_multiscale_gallery.png`
