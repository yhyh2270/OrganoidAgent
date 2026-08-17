# B2F Metric and Objective Improvement Research

## Purpose

This note documents the next improvement direction for the Yichao brightfield-to-fluorescence task.

Do not interrupt an active training run just to apply these ideas. Use this as the design note for the next run after the current run finishes or early-stops.

Current user request captured here:

```text
1, optimize the metrics in the future
2, optimize the objective function to make it learn better that on identifying the positive fluorescent signal
could you think a good objective that still can improve our result?
do a deep thinking and research
```

## Current Observation

The current continuous-target U-Net run shows an important mismatch:

```text
Visual output can become better while scalar validation score becomes worse.
```

This is plausible and expected for this dataset because the fluorescence target is sparse, discontinuous, noisy, and partly ambiguous. A model that starts drawing plausible fluorescence-like regions can look biologically better but still lose pixel-level Pearson or threshold F1 if the predicted signal is shifted, thicker, smoother, or slightly over/under-thresholded.

Therefore the next model should not be selected by a single fixed-threshold pixel score. The checkpoint metric must match the biological goal:

```text
identify true positive fluorescent signal from brightfield,
while suppressing background, debris, and overexposed pseudo-signal.
```

## Research Basis

The task should be treated as biomedical dense prediction, not ordinary image regression.

Key references:

- U-Net is the canonical encoder-decoder architecture for biomedical segmentation and localization with limited data: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597).
- Dice-style losses were introduced into neural medical segmentation to address overlap and foreground imbalance: [Milletari et al., 2016](https://arxiv.org/abs/1606.04797).
- Focal loss was designed for dense prediction where rare hard positives are overwhelmed by abundant easy negatives: [Lin et al., 2017](https://arxiv.org/abs/1708.02002).
- Tversky and focal Tversky losses are useful when false positives and false negatives need asymmetric weighting in small target segmentation: [Salehi et al., 2017](https://arxiv.org/abs/1706.05721), [Abraham and Khan, 2018](https://arxiv.org/abs/1810.07842).
- Boundary-aware losses can help highly imbalanced medical segmentation when overlap loss alone is not enough: [Kervadec et al., 2019](https://arxiv.org/abs/1812.07032).
- Direct IoU/Jaccard optimization is possible with Lovasz-style surrogate losses: [Berman et al., 2018](https://arxiv.org/abs/1705.08790).

The practical conclusion is:

```text
B -> F should not be trained as plain pixel regression.
B -> fluorescent signal support + continuous signal amplitude is the better formulation.
```

## Target Definition

Use the fluorescence-derived target only. Do not create a binary mask from brightfield.

For each organoid crop:

```text
B: brightfield crop
F: raw fluorescence crop
V: valid organoid/crop region
I: ignore region for ambiguous saturation, debris, or invalid padding
Y: noise-suppressed continuous fluorescence target in [0, 1]
M: positive-support mask derived from Y, not from B
S: soft positive support derived from Y, not from B
```

Recommended construction:

```text
F_bg = robust local background estimate
F_corr = max(F - F_bg, 0)
Y = normalize_clip(log(1 + scale * F_corr))
M = 1[Y >= tau_pos]
S = clamp((Y - tau_low) / (tau_high - tau_low), 0, 1)
```

Where:

```text
tau_low < tau_pos < tau_high
```

Interpretation:

- `Y` is the continuous target shown as the white/black fourth column in the current visualization.
- `M` is only used to focus the segmentation-style loss on positive support.
- `S` is a soft support map that avoids making the model overfit a hard 0/1 boundary.
- The model should learn positive fluorescence signal, not every weak green background pixel.

## Recommended Model

Use a gated two-head U-Net, even if the final output shown to the user is one fluorescence-like image.

```text
B -> shared U-Net encoder/decoder
       -> support head p(x) in [0, 1]
       -> amplitude head a(x) in [0, 1]

Y_hat(x) = p(x) * a(x)
```

This is more stable than one direct output because the model gets two simpler jobs:

```text
p(x): is this pixel/region part of true fluorescence-positive signal?
a(x): how strong is the signal if positive?
```

The final output is still a continuous one-channel fluorescence prediction:

```text
Y_hat = p * a
```

This keeps the model compatible with the current B2F visualization while giving the objective a clear way to learn sparse positives.

## Better Objective Function

The next loss should explicitly combine segmentation and continuous regression:

```text
L = lambda_support * L_support
  + lambda_continuous * L_continuous
  + lambda_positive * L_positive_reconstruction
  + lambda_background * L_background_suppression
  + lambda_total * L_total_intensity
  + lambda_calibration * L_area_calibration
```

### Support Loss

Use focal BCE plus focal Tversky or Dice:

```text
L_support = focal_BCE(logit_p, M, valid=V*(1-I))
          + focal_Tversky(p, M, valid=V*(1-I))
```

Why:

- Focal BCE prevents the many negative pixels from dominating the gradients.
- Tversky/Dice forces overlap on the sparse positive region.
- Tversky can be tuned depending on the failure mode.

Initial Tversky settings:

```text
alpha_fp = 0.40
beta_fn  = 0.60
```

This slightly favors not missing positive fluorescent signal. If the model paints too much signal, move toward:

```text
alpha_fp = 0.55
beta_fn  = 0.45
```

### Continuous Loss

Use the noise-suppressed continuous target `Y`, not raw fluorescence:

```text
L_continuous = mean( W(x) * SmoothL1(Y_hat(x), Y(x)) )
```

with:

```text
W(x) = 1 + w_soft * S(x) + w_hard * M(x)
```

Recommended start:

```text
w_soft = 4
w_hard = 8
```

Why:

- The model still learns continuous intensity, not only a mask.
- Positive signal pixels receive much more gradient than empty background.
- Weak but real signal can contribute through `S`, without requiring it to become a hard binary positive.

### Positive Reconstruction Loss

Inside positive-support regions, directly supervise the final prediction:

```text
L_positive_reconstruction = mean_{M=1}( SmoothL1(Y_hat, Y) )
```

This is the core term for "make the positive fluorescent signal correct." It prevents the model from winning by predicting only low-amplitude haze.

### Background Suppression Loss

Suppress false fluorescence in confident negatives:

```text
L_background_suppression = mean_{M=0, I=0}( Y_hat^2 )
```

Use a small but nonzero weight. Too high a weight makes the model blank; too low a weight makes it paint green everywhere.

Recommended start:

```text
lambda_background = 0.05 to 0.15
```

### Total Intensity Loss

Pixel alignment is not always exact, so also match organoid-level expression amount:

```text
T_hat = sum(Y_hat * V)
T     = sum(Y * V)
L_total_intensity = SmoothL1(log(1 + T_hat), log(1 + T))
```

Why:

- If the predicted signal is close but slightly shifted, pixel metrics may punish it.
- The biological question also cares whether the organoid is predicted as high-expression or low-expression.
- This helps checkpoint selection and training avoid visually plausible but quantitatively wrong predictions.

### Area Calibration Loss

Match the predicted support area to target support area:

```text
A_hat = mean(p * V)
A     = mean(M * V)
L_area_calibration = SmoothL1(log(1 + A_hat), log(1 + A))
```

This discourages both failure modes:

- all blank prediction
- over-fat filled positive masks

### Recommended Initial Weights

Start with:

```text
lambda_support    = 1.00
lambda_continuous = 0.75
lambda_positive   = 1.50
lambda_background = 0.10
lambda_total      = 0.25
lambda_calibration = 0.05
```

If output is too blank:

```text
increase lambda_positive
increase beta_fn in Tversky
decrease lambda_background
lower tau_pos for M
```

If output is too fat or too green:

```text
increase lambda_background
increase alpha_fp in Tversky
increase lambda_calibration
raise tau_pos slightly
```

If output localizes signal but intensity is flat:

```text
increase lambda_continuous
increase lambda_total
use positive-only intensity normalization diagnostics
```

## Better Metrics

The next validation should report several metrics and use a composite checkpoint score.

### Pixel Support Metrics

Compute threshold-swept metrics, not only fixed-threshold metrics:

```text
support_AUPRC
best_F1
best_F0.5
best_F2
precision_at_recall_50
recall_at_precision_50
soft_Dice
soft_IoU
```

Why:

- Sparse positives make ROC less informative than PR curves.
- Fixed thresholds can make a visually better model look numerically worse.
- `F2` rewards recall; `F0.5` rewards precision. We need both because the desired tradeoff may change.

### Continuous Signal Metrics

Use metrics weighted toward true signal:

```text
foreground_weighted_MAE
foreground_weighted_RMSE
positive_region_Pearson
positive_region_Spearman
background_false_energy
```

Recommended:

```text
foreground_weighted_MAE = mean((1 + 8*S) * abs(Y_hat - Y))
background_false_energy = mean_{M=0}(Y_hat^2)
```

### Organoid-Level Metrics

The biological result is also organoid-level:

```text
total_intensity_correlation
total_intensity_MAE_log
positive_area_correlation
image_level_expression_AUROC
image_level_expression_AUPRC
```

This handles cases where signal is visually and biologically close but not exactly pixel-aligned.

### Artifact Metrics

Specifically track cases Yichao warned about:

```text
overexposure_false_positive_rate
debris_like_false_positive_rate
negative_organoid_false_positive_area
background_energy_ratio
```

These should be calculated by image category if the dataset marks:

```text
positive
negative
overexposure_suppressed
ambiguous
```

### Visual Metrics

Every checkpoint should save fixed validation panels:

```text
B | raw F | target Y | predicted Y_hat | abs error | support p
```

Use the same fixed examples across epochs. Human visual review is necessary because some fluorescence labels are biologically ambiguous.

## Better Checkpoint Selection

Do not select best checkpoint by a single current scalar. Use a composite score:

```text
score =
  0.25 * normalized_support_AUPRC
+ 0.20 * normalized_best_F1
+ 0.20 * normalized_total_intensity_corr
+ 0.15 * normalized_positive_region_corr
- 0.10 * normalized_background_false_energy
- 0.10 * normalized_total_intensity_log_error
```

Also keep:

```text
best_visual_epoch
best_support_epoch
best_intensity_epoch
last_epoch
```

This avoids losing useful checkpoints just because a single thresholded metric moved down.

## Training Strategy

Use a staged schedule:

```text
Stage 1: train support head strongly
Stage 2: train support + amplitude jointly
Stage 3: fine-tune with total intensity and calibration losses
Stage 4: optional visual refinement only after biological metrics work
```

Recommended curriculum:

```text
epochs 1-30:
  emphasize support and positive reconstruction

epochs 31-100:
  balance support, continuous target, and background suppression

epochs 101+:
  lower LR, select by composite metric and fixed visual panel
```

Avoid adversarial/GAN loss until the supervised target is reliable. GAN loss can make fluorescence look plausible while making quantitative biology worse.

## Why This Should Improve the Current Result

The current run is useful because it shows the network can start producing fluorescence-like structure from brightfield. Its weakness is that the objective does not sufficiently force:

```text
correct positive support
correct sparse signal localization
correct intensity only where true signal exists
low false energy in negative/background regions
```

The proposed objective directly targets those points:

- Support loss handles "where is positive fluorescence?"
- Continuous loss handles "how strong is it?"
- Positive reconstruction prevents blank outputs.
- Background suppression prevents green haze.
- Total intensity loss protects biologically meaningful expression amount.
- Composite metrics prevent rejecting visually better models due to a bad fixed threshold.

## Implementation Checklist For The Next Run

1. Keep the current finished run as evidence; do not overwrite it.
2. Add two-head model output: support `p` and amplitude `a`.
3. Keep final continuous output as `Y_hat = p * a`.
4. Train against both `M/S` and `Y`.
5. Add threshold-swept support metrics.
6. Add organoid-level total expression metrics.
7. Save fixed validation panels every checkpoint.
8. Save checkpoint categories: best composite, best support, best intensity, latest.
9. Use early stop on composite score, not current fixed score.
10. Review visual panels before deciding whether a model is actually better.

