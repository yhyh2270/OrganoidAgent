# Yichao B2F Strong Training Strategy

## Current diagnosis

The previous future-expression prediction is not reliable yet. The first problem to fix is not the future model; it is to make same-time brightfield-to-fluorescence (B2F) reconstruction strong and stable.

The old B2F run had several avoidable weaknesses:

- It trained only 40 epochs.
- It used 256 x 256 crops only, which can compress thin peripheral fluorescent cell structures.
- It selected `best_model.pt` mainly by validation expression AUROC. That is unstable because expression examples are rare.
- Validation/test splits are highly imbalanced. The test split has only 14 expression-labeled B2F crops in the previous run.
- The fluorescence target is sparse relative to background, so ordinary image losses are dominated by easy dark/background pixels.
- The future task inherits additional problems: approximate track linking, rare future-expression examples, and uncertain target policy (`last_future` may not be the biologically best target).

## Root cause found on May 11, 2026

The failed strong B2F run was not actually training. In:

```text
/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_future_expression/stage1_b2f_strong_384_v2_long
```

`best_model.pt` and `last_model.pt` had identical model weights from epoch 1 to epoch 21:

```text
relative_l2_delta = 0.0
max_abs_delta = 0.0
changed_params = 0 / 127
optimizer state entries = 0
GradScaler scale = 0.0
```

The concrete failure was AMP, not model capacity: the CUDA AMP `GradScaler` collapsed to `0.0`, so `scaler.step(optimizer)` skipped optimizer updates while the loop kept logging epochs. This can make a run look alive while the model is frozen.

The trainer now records optimizer-step diagnostics and raises if AMP scale becomes non-positive or non-finite. The tmux helper now launches the long run without `--amp`.

## Correct priority

1. Train B2F properly first.
2. Use the B2F encoder as pretraining for future prediction.
3. Then train a joint temporal model for future expression.

Do not scale the future model before the B2F model and temporal labels are credible.

## Strong B2F changes

New code:

```text
/home/lachlan/ProjectsLFS/OrganoidAgent/differentiation_prediction/yichao_future_expression/train_b2f_strong.py
/home/lachlan/ProjectsLFS/OrganoidAgent/differentiation_prediction/yichao_future_expression/resume_strong_b2f_tmux.sh
/home/lachlan/ProjectsLFS/OrganoidAgent/differentiation_prediction/yichao_future_expression/debug_b2f_learning.py
```

Main changes:

- Reads original instance crops by default, not only the saved 256 crops.
- Defaults to 384 x 384 training.
- Uses a larger residual GroupNorm U-Net with squeeze-excitation.
- Also supports `--architecture pix2pix_unet`, an InstanceNorm + ELU pix2pix-style U-Net adapted from the older working fluorescence code.
- Uses pixel imbalance-aware training for sparse fluorescence.
- Uses continuous-intensity fluorescence loss instead of turning the target into a low-threshold binary mask.
- Uses Charbonnier intensity loss, SSIM-like structure loss, and Sobel edge loss.
- Uses scalar auxiliary targets for expression status, peak fluorescence, and total fluorescence.
- Supports image-level balanced sampling for rare expression examples.
- Selects the best checkpoint by reconstruction/signal quality, not AUROC alone.
- Saves `last_model.pt` every epoch and sparse periodic checkpoints every 20 epochs.
- Keeps only the most recent periodic checkpoints to avoid disk overload.
- Logs `optimizer_steps`, gradient norm, parameter norm changes, and a sentinel parameter delta.
- Fails loudly if AMP scale collapses instead of silently logging fake training.
- Supports resumable early stopping based on validation convergence.

## Real learning gate

Before trusting a long run, use the small overfit check:

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid
python -m differentiation_prediction.yichao_future_expression.debug_b2f_learning \
  --output-root /home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_future_expression/debug_b2f_pix2pix_overfit_128_real \
  --image-size 128 \
  --subset-size 32 \
  --steps 120 \
  --batch-size 4 \
  --base-channels 48 \
  --lr 0.001 \
  --panel-every 30
```

This script uses real projected Yichao instance pairs, selects high-signal training examples, and overfits them with the older pix2pix-style U-Net architecture from:

```text
/home/lachlan/ProjectsLFS/OrganoidAgent/pixel2pixel_fluorescent/fluorescent/cyclegan_modules.py
```

The May 11 overfit check succeeded:

```text
initial_loss = 0.4587736614
final_loss = 0.1359374505
loss_drop_fraction = 0.7036938650
final_param_delta_l2 = 15.1127531912
learned = true
```

Output panels:

```text
/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_future_expression/debug_b2f_pix2pix_overfit_128_real/panels
```

## Long-run command

The intended long run is:

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid
python -u -m differentiation_prediction.yichao_future_expression.train_b2f_strong \
  --output-root /home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_future_expression/stage1_b2f_pix2pix_384_v1_noamp_long \
  --image-size 384 \
  --path-mode original_crop \
  --epochs 1000 \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --architecture pix2pix_unet \
  --base-channels 64 \
  --dropout 0.5 \
  --lr 1.5e-4 \
  --min-lr 1e-6 \
  --balanced-sampler \
  --eval-every 5 \
  --panel-every 20 \
  --save-every 20 \
  --keep-periodic 8 \
  --early-stop \
  --early-stop-metric score \
  --early-stop-patience-evals 10 \
  --early-stop-min-delta 0.001 \
  --early-stop-min-epochs 40 \
  --resume
```

The tmux helper starts this command:

```bash
/home/lachlan/ProjectsLFS/OrganoidAgent/differentiation_prediction/yichao_future_expression/resume_strong_b2f_tmux.sh
```

Early stopping uses the same reconstruction-first validation score used for best-checkpoint selection:

```text
score = -val_loss + 0.05 * val_signal_f1 + 0.01 * val_peak_pearson
```

It stops after 10 validation checks without at least `0.001` score improvement, but only after 40 epochs. On resume, the trainer scans existing `metrics.jsonl`; it can therefore detect that a run has already converged and stop immediately rather than wasting GPU time.

## May 11 progress check and correction

The first long 384 run reached epoch 47, then stopped with:

```text
TypeError: silu() keywords must be strings
```

The run also showed flat validation metrics from epoch 1 through epoch 45. The validation panels showed that the model was producing a broad green brightfield-like reconstruction rather than sparse true fluorescence. That means the issue was not simply insufficient epoch count.

The fix is:

- Disable AMP for this trainer by default in the tmux helper. The previous AMP run had `GradScaler scale = 0.0` and no weight updates.
- Remove the channels-last training path from the tmux command because it likely triggered the PyTorch/SiLU runtime path.
- Replace the strong model's SiLU activations with GELU.
- Stop using low-threshold binary fluorescence BCE as the main pixel signal. A low threshold made most organoid pixels count as fluorescent and encouraged broad green predictions.
- Use continuous target-intensity weighting: bright fluorescence pixels get larger weight, but dark/background pixels still pull the prediction down.
- Reduce scalar auxiliary weight so image reconstruction drives the run.

## What success should look like

B2F should not be judged only by image MAE. A mostly dark prediction can get deceptively good MAE. The useful checks are:

- `val_masked_mae`: fluorescence error inside the organoid mask.
- `val_signal_mae`: error on true fluorescent signal pixels.
- `val_signal_f1`: whether the model localizes fluorescent pixels.
- `val_peak_pearson`: whether image-level fluorescence intensity is ranked correctly.
- Saved validation panels every 20 epochs.
- Held-out `test_best.png` after training.

## Next future model after B2F works

After B2F is credible, use its encoder as initialization for a future model:

```text
B1...Bk -> temporal encoder -> future heads
```

Recommended future heads:

- Current B2F reconstruction: `Bk -> Fk`
- Short-horizon fluorescence: `Bk -> F(k+1)` and `Bk -> F(k+2)`
- Peak future fluorescence: `B1...Bk -> F_peak`
- Future expression probability
- First-expression timing

Use horizon-specific targets rather than only `last_future`. The model should identify the earliest day `k` where morphology becomes predictive, not just produce one final-frame guess.
