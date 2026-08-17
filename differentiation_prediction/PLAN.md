# Yichao Differentiation Prediction Plan

## Objective

Train a pix2pix model that predicts fluorescence (`c1`) from brightfield (`c0`) using the four Yichao datasets without data leakage and with a split strategy that respects position, depth, and time structure.

## What Counts as One Sample

For this problem, the correct supervised unit is:

- one paired 2D plane at fixed `dataset`, `position`, `time`, and `z`

Do not split adjacent planes from the same position across train and test.

## What the Four Datasets Should Do

### `Data-Yichao-1`

- 5 static paired samples only
- duplicated inside `Data-Yichao-2`
- use only for manual inspection or parser checks
- do not use for training or evaluation

### `Data-Yichao-2`

- mixed file
- 5 static samples duplicated from `Y1`
- 6 dynamic Day-2 positions at `1024 x 1024`
- keep static folders excluded
- use dynamic folders in two ways:
  - first as `external_test` in the baseline split
  - later as part of `all_dynamic_split` after normalizing around 512 px tiles

### `Data-Yichao-3`

- main dynamic dataset
- 13 positions across Day 2/3/4
- `512 x 512`
- should be the backbone of the first train/val/test run

### `Data-Yichao-4`

- second dynamic dataset
- 9 positions across Day 2/3
- `512 x 512`
- should be merged with `Y3` for the first real baseline

## Recommended Training Stages

### Stage 0: Manifest and Sanity Checks

1. Scan all grouped JPEG roots.
2. Build one manifest row per `position/t/z` brightfield-fluorescence pair.
3. Write:
   - full manifest
   - per-position summary
   - split summary JSON
4. Verify counts:
   - `Y1`: 5 pairs
   - `Y2`: 965 pairs total, but only 960 dynamic pairs are unique/useful
   - `Y3`: 10019 pairs
   - `Y4`: 7940 pairs

### Stage 1: Baseline In-Domain Model

Use `baseline_split`.

Policy:

- exclude `Y1`
- exclude static folders from `Y2`
- train/val/test on `Y3 + Y4` only
- evaluate `Y2` dynamic as `external_test`

Why:

- avoids duplicate static data
- avoids mixing 1024 and 512 data before deciding how to normalize scale
- gives a cleaner first benchmark

### Stage 2: All-Dynamic Model

Use `all_dynamic_split`.

Policy:

- exclude `Y1`
- exclude static folders from `Y2`
- include dynamic positions from `Y2`, `Y3`, and `Y4`
- keep the split grouped by position

Normalization strategy:

- keep `tile_size=512`
- use `tile_mode=quadrants`
- this leaves `Y3/Y4` untouched
- this splits each `1024 x 1024` Y2 plane into four deterministic 512 tiles

This is the simplest way to combine all dynamic data without resizing the 512 px data upward.

### Stage 3: Redundancy Control

The raw stacks are heavily correlated across time and depth. For a first stable run:

- try `--time-stride 2`
- try `--z-stride 2`
- or cap with `--max-train-pairs-per-position`

Then remove those restrictions once the training pipeline is stable.

### Stage 4: Extensions

After the 2D pix2pix baseline works:

- add temporal context by stacking adjacent timepoints as channels
- add depth context by stacking adjacent z planes as channels
- compare `current plane only` vs `current + neighbors`
- keep evaluation grouped by position so leakage does not reappear

## Files Produced by This Folder

### Manifest bundle

- `yichao_manifest.csv`
- `yichao_manifest.jsonl`
- `yichao_positions.csv`
- `yichao_summary.json`

### Split columns

- `baseline_split`
- `all_dynamic_split`

### Training outputs

Written under `results/differentiation_prediction/`:

- run config
- checkpoints
- sample prediction grids
- metrics CSV

## Commands

Build the manifest:

```bash
python -m differentiation_prediction.build_yichao_manifest \
  --output-dir results/differentiation_prediction/manifest
```

Smoke-test the baseline pipeline:

```bash
python -m differentiation_prediction.train_pix2pix \
  --manifest results/differentiation_prediction/manifest/yichao_manifest.csv \
  --split-column baseline_split \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --max-train-steps 1 \
  --max-eval-steps 1 \
  --max-train-pairs-per-position 4 \
  --max-eval-pairs-per-position 2 \
  --cpu
```

Run the first real baseline:

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

Equivalent config-based command:

```bash
python -m differentiation_prediction.train_pix2pix \
  --config differentiation_prediction/configs/baseline_y34_y4.json
```

Run the all-dynamic experiment:

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
