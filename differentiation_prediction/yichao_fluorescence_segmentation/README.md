# Yichao Context-Aware Fluorescence Segmentation

This folder builds cleaned fluorescence-positive masks from paired projected organoid crops and trains a context-aware segmentation model from brightfield input.

The goal is not plain pixel regression. The raw fluorescence channel often contains background, debris, saturated haze, or full-field pseudo-signal. The target builder therefore produces three regions:

- `positive`: confident fluorescence-positive cell signal inside a complete organoid crop.
- `ignore`: high or saturated fluorescence that is too ambiguous to punish as positive or negative.
- `negative`: valid organoid pixels not marked positive or ignore.

The trainer uses brightfield plus organoid mask plus distance-to-edge as input. The model is a U-Net with ASPP multi-scale context and an image-level expression gate, so it can learn both local morphology and whole-organoid context.

## Build Clean Targets

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid
python -m differentiation_prediction.yichao_fluorescence_segmentation.build_clean_targets --qc-count 120
```

Important outputs:

- `analysis-outputs/yichao_fluorescence_segmentation/manifests/segmentation_targets_manifest.csv`
- `analysis-outputs/yichao_fluorescence_segmentation/manifests/segmentation_targets_summary.json`
- `analysis-outputs/yichao_fluorescence_segmentation/qc/target_qc_gallery.png`

## Run Debug Checks

```bash
python -m differentiation_prediction.yichao_fluorescence_segmentation.build_clean_targets --self-test

python -m differentiation_prediction.yichao_fluorescence_segmentation.train_segmentation \
  --output-root analysis-outputs/yichao_fluorescence_segmentation/debug_memory_384 \
  --image-size 384 --epochs 1 --batch-size 8 --base-channels 32 \
  --num-workers 0 --balanced-sampler --limit-train-batches 1 --limit-eval-batches 1

python -m differentiation_prediction.yichao_fluorescence_segmentation.train_segmentation \
  --output-root analysis-outputs/yichao_fluorescence_segmentation/debug_overfit_noaug_128 \
  --image-size 128 --epochs 30 --batch-size 4 --base-channels 16 \
  --num-workers 0 --lr 0.001 --overfit-count 32 --eval-every 2 --panel-every 10
```

## Full Training

```bash
./differentiation_prediction/yichao_fluorescence_segmentation/resume_segmentation_tmux.sh
```

The default tmux session is `yichao_fluorescence_segmentation_v1`.

Important outputs:

- `analysis-outputs/yichao_fluorescence_segmentation/runs/global_gated_unet_v1/logs/train.log`
- `analysis-outputs/yichao_fluorescence_segmentation/runs/global_gated_unet_v1/metrics.jsonl`
- `analysis-outputs/yichao_fluorescence_segmentation/runs/global_gated_unet_v1/best_model.pt`
- `analysis-outputs/yichao_fluorescence_segmentation/runs/global_gated_unet_v1/predictions/`
- `analysis-outputs/yichao_fluorescence_segmentation/runs/global_gated_unet_v1/test_metrics.json`

