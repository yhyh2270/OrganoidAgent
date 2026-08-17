# BioAgentUtils

Utilities for local microscopy data processing.

## LIF to JPEG Export

`lif_to_jpeg.py` exports Leica `.lif` images to JPEG files.
By default it exports all time points and all Z planes for each selected image/channel.

Install dependency:

```bash
pip install readlif
```

Example:

```bash
python BioAgentUtils/lif_to_jpeg.py \
  "Yichao-data-1/P11N&N39_Rep_DF.lif" \
  --output-dir "Yichao-data-1/P11N&N39_Rep_DF_jpeg"
```

Optional flags:

- `--image-indices 0 1` export specific image indices.
- `--channels 0` export specific channel(s).
- `--z-indices 0 1 2` export specific Z planes.
- `--time-indices 0 5 10` export specific time points.
- `--first-plane-only` export only `t=0, z=0` (legacy single-plane mode).
- `--overwrite` replace existing JPEGs.
- `--quality 95` set JPEG quality.

## Organize JPEG by Object

`organize_lif_jpegs.py` groups flat JPEG exports into one folder per object/series.

Example:

```bash
python BioAgentUtils/organize_lif_jpegs.py \
  "Data-Yichao-v1/Data-Yichao-2/P11N&N39_Rep_DF_jpeg_all" \
  --output-dir "Data-Yichao-v1/Data-Yichao-2/P11N&N39_Rep_DF_jpeg_all_by_object"
```

By default it uses hardlinks (`--mode link`) to avoid duplicating large files.
Use `--mode copy` or `--mode move` if needed.

## Train pix2pix (Yichao data)

`train_pix2pix_yichao.py` trains a brightfield-to-fluorescence pix2pix model.

Defaults:
- Train set: `Data-Yichao-v1/Data-Yichao-2/P11N&N39_Rep_DF_jpeg_all_by_object`
- Verification set: `Data-Yichao-v1/Data-Yichao-1/P11N&N39_Rep_DF_jpeg_all_by_object` (split into val/test)
- Output root: `results/`

Run full training (GPU-only):

```bash
./BioAgentUtils/run_train_pix2pix_yichao.sh
```

Current default prep mode in launcher:
- train set: first paired frame per top-level folder
- crop each selected 1024x1024 pair into 4 tiles of 512x512
- expected train samples for Yichao-2: `11 folders x 4 = 44`

Start in a tmux session (interactive; Ctrl+C works normally):

```bash
./BioAgentUtils/start_train_pix2pix_tmux.sh pix2pix-train --log-interval 10
tmux attach -t pix2pix-train
```

Live logs now include:
- startup config (device, pair counts, loader steps)
- scan progress while indexing dataset files
- preprocessing progress (selected objects and crop expansion)
- per-step progress with ETA and samples/sec
- epoch and validation summaries

Useful flags:
- `--log-interval 10` print every 10 train/eval steps
- `--epochs 20 --batch-size 4`
- `GPU_INDEX=0` (default; uses first GPU)
- `KILL_STALE=1` kill stale pix2pix python processes before training

Quick smoke run:

```bash
./BioAgentUtils/run_train_pix2pix_yichao.sh \
  --epochs 1 --batch-size 2 \
  --max-train-steps 2 --max-eval-steps 1 --log-interval 1
```

Notes:
- Script runs a torch/CUDA preflight check and prints stage logs before training.
- If it hangs at `[preflight] importing torch...`, clear stale GPU python jobs, e.g.:
  - `./BioAgentUtils/clear_gpu_python_processes.sh 0`
- Test evaluation triplet plots are saved under:
  - `results/pix2pix_npy_*/plots/test_triplets_epoch_XXX.png`
  - `results/pix2pix_npy_*/plots/test_triplets_best.png`

## Prepare Paired NPY (Yichao)

Prepare paired `c0 -> c1` arrays as `.npy` with progress logs:

```bash
./BioAgentUtils/run_prepare_yichao_pairs_to_npy.sh
```

Default prep behavior:
- train: first pair per top-level folder, then 4 crops of `512x512` per pair
- test: all pairs in `Data-Yichao-1`, 4 crops of `512x512` per pair

Default outputs:
- `results/yichao_paired_npy/train_input.npy`
- `results/yichao_paired_npy/train_target.npy`
- `results/yichao_paired_npy/test_input.npy`
- `results/yichao_paired_npy/test_target.npy`
- metadata JSON files and `summary.json`
