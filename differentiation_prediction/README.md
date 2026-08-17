# Differentiation Prediction

This folder contains a manifest-first pix2pix pipeline for the four Yichao datasets:

- `Data-Yichao-1`
- `Data-Yichao-2`
- `Data-Yichao-3`
- `Data-Yichao-4`

The goal is to predict fluorescence (`c1`) from brightfield (`c0`) without leaking duplicated samples or splitting correlated planes across train and test.

## Folder Layout

- `PLAN.md`: detailed experiment and data-usage plan.
- `build_yichao_manifest.py`: scans the grouped JPEG folders and writes manifest tables.
- `manifest.py`: dataset metadata, split policy, and summary logic.
- `data.py`: manifest reader, position-level subsampling, and tiling dataset.
- `model.py`: pix2pix generator and discriminator.
- `train_pix2pix.py`: training entrypoint for manifest-driven experiments.
- `configs/`: example JSON argument bundles for common runs.

## Recommended Use of the Four Datasets

- `Y1`: exclude from quantitative training and test. Its 5 static samples duplicate Y2.
- `Y2` static MUC2 folders: exclude from quantitative training and test.
- `Y2` dynamic Day-2 positions: keep as an external test first, because the physical sampling differs from Y3/Y4.
- `Y3` + `Y4` dynamic positions: use for the first real training run because both are 512 px dynamic monitoring datasets with the same extracted layout.

The manifest generator writes two split columns:

- `baseline_split`: `Y3` and `Y4` are split by position into `train/val/test`; `Y2` dynamic is labeled `external_test`; `Y1` and all static samples are `exclude`.
- `all_dynamic_split`: `Y2`, `Y3`, and `Y4` dynamic positions are all split by position into `train/val/test`; `Y1` and all static samples are `exclude`.

## Quickstart

Build the manifest bundle:

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
python -m differentiation_prediction.build_yichao_manifest \
  --output-dir results/differentiation_prediction/manifest
```

Baseline training on `Y3+Y4`, with `Y2` kept only for external evaluation:

```bash
python -m differentiation_prediction.train_pix2pix \
  --manifest results/differentiation_prediction/manifest/yichao_manifest.csv \
  --split-column baseline_split \
  --tile-mode quadrants \
  --tile-size 512 \
  --image-size 512 \
  --batch-size 8 \
  --epochs 50
```

Or use the provided config bundle:

```bash
python -m differentiation_prediction.train_pix2pix \
  --config differentiation_prediction/configs/baseline_y34_y4.json
```

All-dynamic training after harmonizing around 512 px tiles:

```bash
python -m differentiation_prediction.train_pix2pix \
  --manifest results/differentiation_prediction/manifest/yichao_manifest.csv \
  --split-column all_dynamic_split \
  --tile-mode quadrants \
  --tile-size 512 \
  --image-size 512 \
  --batch-size 8 \
  --epochs 50
```

## Notes

- Splits are position-level, not plane-level.
- `tile_mode=quadrants` keeps 512 px datasets untouched and expands 1024 px datasets into four 512 px tiles.
- Use `--time-stride` and `--z-stride` if you want a less redundant first pass.
- Use `--max-train-pairs-per-position` for smoke tests or quick ablations.
- The example JSON files in `configs/` can be passed directly with `--config`.
