# Yichao Per-Z Brightfield-to-Fluorescence Performance Report

Date: 2026-05-20

This note documents the completed per-z brightfield-to-fluorescence (B2F) experiment. The goal is to predict the cleaned fluorescence signal from the corresponding brightfield organoid crop.

## Core Task

The model input is the brightfield channel, not fluorescence. In the Yichao data convention used here:

- `c1` is brightfield.
- `c0` is fluorescence.
- The prediction target is a one-channel, fluorescence-derived, noise-suppressed continuous signal in `[0, 1]`.
- The binary fluorescence support mask is derived from the cleaned fluorescence target and is used for support-style losses and metrics. It is not a brightfield mask.

Mathematically, for each organoid instance:

```text
B_i = brightfield crop
F_i = raw fluorescence crop
Y_i = cleaned continuous fluorescence target from F_i
S_i = binary/soft support derived from Y_i
G_theta(B_i, M_i, D_i) -> Y_hat_i
```

where `M_i` is the organoid segmentation mask and `D_i` is a distance-map auxiliary input.

## Data Policy

The main completed run uses every original z-plane as a separate example. No z projection is used.

- Train/validation/internal-test source: `Data-Yichao-1` through `Data-Yichao-10`.
- External holdout: `Data-Yichao-11`.
- Data-Yichao-11 was not used in training, validation, or model selection.

Counts:

- Per-z source image records: `79,405`
- Per-z instance records: `354,797`
- Filtered usable training-target rows: `214,311`
- Train/validation/internal-test split: `139,935 / 62,814 / 11,562`
- External Data-Yichao-11 target rows: `7,489`
- Target status counts in Data 1-10: `128,246 positive`, `83,098 negative`, `2,967 overexposure_suppressed`

## Training Method

The run used a 256 px hybrid U-Net style model with three input channels:

- brightfield crop,
- organoid mask,
- organoid distance map.

The output is one channel: the cleaned continuous fluorescence target.

Training configuration:

- epochs: `500`
- batch size: `12`
- samples per epoch: `20,000`
- optimizer learning rate: `1.5e-4`
- minimum learning rate: `1e-6`
- weight decay: `2e-4`
- base channels: `32`
- dropout: `0.05`
- balanced sampler: enabled
- early stopping: disabled
- device: CUDA

The objective combines continuous regression and fluorescence-support learning:

```text
L = L1_weighted
  + 0.25 MSE_weighted
  + 0.5 soft_dice_loss
  + focal_support_loss
  + 0.2 total_intensity_loss
  + 0.08 background_penalty
```

The important design choice is that the model does not learn a strict black/white mask only. It learns the continuous cleaned fluorescence signal, while support losses help it pay attention to sparse positive regions.

## Main Performance

### Data-Yichao-11 External Holdout

This is the easiest-to-understand result because Data-Yichao-11 was never used for training.

- Fluorescent support best F1: `0.454`
- Practical threshold F1 at threshold `0.2`: `0.418`
- Pixel-wise continuous Pearson: `0.333`
- Positive-pixel Pearson: `0.041`
- Continuous soft Dice: `0.206`
- Support AUPRC: `0.266`
- Total fluorescence ratio: `1.135x` truth
- Total fluorescence log-intensity Pearson: `0.580`
- Background false energy: `0.00965`

Plain interpretation:

- The model is not random. The positive support prevalence in Data-Yichao-11 is about `7.1%`, while the best F1 is `45.4%`, roughly `6.4x` the prevalence scale.
- It roughly locates fluorescent regions and estimates total fluorescence amount within about `13.5%` high overall.
- It is not pixel-perfect, and intensity inside positive regions remains weakly correlated.

### Validation and Internal Test

Best validation checkpoint:

- epoch: `210`
- validation selection score: `0.314`
- validation support best F1: `0.477`
- validation threshold F1: `0.447`
- validation total intensity ratio: `1.016x`

Last epoch:

- epoch: `500`
- validation support best F1: `0.476`
- validation threshold F1: `0.432`
- validation total intensity ratio: `0.976x`

Internal test:

- support best F1: `0.225`
- threshold F1: `0.129`
- total intensity ratio: `0.466x`

The internal test split is substantially harder than the Data-Yichao-11 holdout under these metrics. This should be kept in mind when using a single number as an accuracy summary.

## Projection Comparison

A separate max-brightfield/max-fluorescence projection run was also completed. It compresses z into a single projected organoid image.

On external Data-Yichao-11:

- Max-projection support best F1: `0.482`
- Per-z support best F1: `0.454`
- Max-projection threshold F1: `0.429`
- Per-z threshold F1: `0.418`
- Max-projection total intensity ratio: `0.828x`
- Per-z total intensity ratio: `1.135x`
- Max-projection total-intensity log Pearson: `0.355`
- Per-z total-intensity log Pearson: `0.580`

Interpretation:

- Max projection slightly improves binary-region F1 on Data-Yichao-11.
- Per-z keeps depth information and has better total-expression ranking on Data-Yichao-11.
- For future differentiation prediction, the per-z database is more informative because it preserves time/depth structure.

## Important Output Paths

- Per-z summary: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_per_z_b2f_pipeline/summary.md`
- Per-z model run root: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_fluorescence_continuous/runs/per_z_1to10_hybrid_unet_v1_500`
- Per-z training curves: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_fluorescence_continuous/runs/per_z_1to10_hybrid_unet_v1_500/plots/training_metrics.png`
- Per-z internal test panel: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_fluorescence_continuous/runs/per_z_1to10_hybrid_unet_v1_500/predictions/test_best.png`
- Data-Yichao-11 external panel: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_external_tests/Data-Yichao-11_per_z/evaluation/Data-Yichao-11_per_z_last_epoch_external_panel.png`
- Data-Yichao-11 external gallery: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_per_z_b2f_pipeline/visualizations/per_z_Data-Yichao-11_external_gallery.png`
- Publication report folder: `/home/lachlan/ProjectsLFS/OrganoidAgent/publication/yichao_per_z_b2f_performance_report`

## Bottom Line

The current per-z B2F model has learned real brightfield-to-fluorescence signal. On the external Data-Yichao-11 holdout, the most intuitive accuracy number is about `45%` F1 for fluorescent-region detection, with total fluorescence amount predicted at `1.13x` of truth. This is clearly above random, but it is not yet a solved pixel-perfect fluorescence reconstruction task.
