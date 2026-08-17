# Joint Mask and Intensity Fluorescence Prediction Strategy

## Short Answer

Yes. The current segmentation task means:

```text
brightfield B  ->  binary expression mask M
where M is derived from fluorescence F
```

This is not trying to reproduce raw fluorescence directly. It asks: given brightfield morphology, where are the fluorescence-positive cell-like regions?

The next better model should predict both:

```text
B -> M          positive / negative expression mask
B -> A          continuous fluorescence amount where expression is present
F_hat = M_hat * A_hat
```

This is a good divide-and-conquer formulation. The mask branch learns whether a region is truly expressing. The intensity branch learns how strong the expression is. This should be more stable than asking one network to directly regress raw fluorescence everywhere, because the raw image is sparse, noisy, imbalanced, and contaminated by debris/background pseudo-signal.

## Research Basis

Label-free fluorescence prediction from transmitted-light microscopy is a real and published task. Ounkomol et al. showed that transmitted-light images can be used to predict fluorescence structure directly, including 3D fluorescence, but their setting depends on clean paired labels and enough signal in the transmitted-light input [Ounkomol et al., Nature Methods 2018](https://www.nature.com/articles/s41592-018-0111-2).

Plain pix2pix is relevant because it learns paired image-to-image mappings with a conditional adversarial loss [Isola et al., CVPR 2017](https://arxiv.org/abs/1611.07004). However, for our data, raw B-to-F regression alone is weak because most pixels are negative and some bright fluorescence is biological artifact or overexposed background. A direct generator can minimize loss by producing weak blur, background haze, or conservative averages.

U-Net is the right base family for the mask branch because it combines context capture with precise localization, and it was designed for biomedical segmentation with limited data [Ronneberger et al., MICCAI 2015](https://arxiv.org/abs/1505.04597).

The positive signal is class-imbalanced, so focal/Tversky/Dice-style losses are appropriate. Focal loss was designed to stop abundant easy negatives from overwhelming dense prediction training [Lin et al., ICCV 2017](https://arxiv.org/abs/1708.02002). Medical segmentation literature also supports focal/Dice combinations for imbalanced masks [Yeung et al., 2021](https://arxiv.org/abs/2102.04525).

The mask-plus-intensity setup is a multi-task learning problem: a shared brightfield encoder predicts classification-like output and regression-like output. Loss balance matters. Kendall et al. showed that multi-task systems can jointly learn regression and classification objectives, and that weighting losses carefully is important [Kendall et al., CVPR 2018](https://arxiv.org/abs/1705.07115).

## Statistical View

Fluorescence is naturally zero-inflated:

```text
Most pixels: no real expression
Some pixels: true positive expression with a continuous intensity
Some pixels: debris/background/overexposure that should not be treated as true biology
```

A better formulation is therefore a hurdle model:

```text
P(Y_x > 0 | B)          expression probability at pixel x
E[log(1 + F_x) | Y_x > 0, B]   intensity if truly positive
```

Let:

```text
B: brightfield crop
F: fluorescence crop after robust background handling
V: valid organoid region
I: ignore region for ambiguous debris/saturation
M: cleaned binary positive mask derived from F
A: positive fluorescence amplitude target
p_theta(x) = P(M_x = 1 | B)
a_theta(x) = predicted positive intensity
```

Then:

```text
F_hat_x = p_theta(x) * a_theta(x)
```

or, for sharper visual output:

```text
F_hat_x = 1[p_theta(x) > tau] * a_theta(x)
```

The first form is differentiable and good for training. The second form is better for final visualization and quantification.

## Target Construction

The current target builder already creates:

```text
positive mask M
ignore mask I
valid region V
overexposure-suppressed status
```

For the joint model we should add one more target:

```text
A_x = clipped_log_background_corrected_fluorescence_x
```

Recommended intensity target:

```text
F_bg = robust background estimate outside/dilated-away from the organoid
F_corr = max(F_raw - F_bg, 0)
A = log(1 + scale * F_corr) / log(1 + scale)
```

Only train `A` strongly inside confident positive pixels:

```text
train intensity where M = 1 and I = 0
weakly suppress intensity where M = 0 and I = 0
ignore where I = 1
```

This is important. If we regress intensity everywhere, overexposed images and debris can dominate the loss. If we regress only where the cleaned mask says "true positive", the intensity branch learns degree of expression rather than background brightness.

## Model Design

Use one shared encoder and three heads:

```text
Brightfield B
  -> shared encoder
      -> mask head:       logit_m(x)
      -> intensity head:  logit_a(x) or raw_a(x)
      -> global head:     expression score / total expression
```

Outputs:

```text
p = sigmoid(logit_m)
a = softplus(raw_a) or sigmoid(logit_a)
F_hat = p * a
global_hat = predicted image/organoid-level expression
```

The global head matters because some images may contain no real positive cells. It helps the model avoid painting small false positive speckles when the whole organoid is likely negative.

The mask head should remain precision-first because false positives from debris are worse than missing a weak cell in early feasibility testing. Use F0.5 or precision-weighted selection for best checkpoint.

## Loss Function

Use a composite loss:

```text
L = lambda_mask * L_mask
  + lambda_int_pos * L_intensity_positive
  + lambda_recon * L_reconstruct
  + lambda_bg * L_background_suppression
  + lambda_global * L_global
  + lambda_area * L_area
```

Mask loss:

```text
L_mask = focal_BCE(logit_m, M, V * (1 - I)) + Tversky(p, M, V * (1 - I))
```

Positive intensity loss:

```text
L_intensity_positive =
  Huber(a, A) over pixels where M = 1 and I = 0
```

Reconstruction loss:

```text
L_reconstruct =
  Huber(p * a, A) over pixels where I = 0
```

Background suppression:

```text
L_background_suppression =
  mean((p * a)^2) over pixels where M = 0 and I = 0
```

Global expression loss:

```text
G = 1 if sum(M) is above a small threshold else 0
L_global = BCE(global_logit, G)
```

Area/intensity calibration:

```text
L_area = Huber(log(1 + mean(p)), log(1 + mean(M)))
L_total_intensity = Huber(log(1 + sum(p * a)), log(1 + sum(A * M)))
```

Initial practical weights:

```text
lambda_mask = 1.0
lambda_int_pos = 1.0
lambda_recon = 0.25
lambda_bg = 0.25
lambda_global = 0.30
lambda_area = 0.05
```

If the model paints too much background, increase `lambda_bg` and select checkpoints by F0.5. If the model detects masks but intensity is flat, increase `lambda_int_pos` and `lambda_total_intensity`.

## Training Schedule

Do not start directly with a GAN. The robust path should be:

1. Train mask-only model first.
2. Add intensity head and train with the mask loss still active.
3. Fine-tune jointly with `F_hat = p * a`.
4. Only after the supervised model works, optionally add a small PatchGAN or perceptual/local texture loss for visual realism.

Reason: adversarial loss can make outputs look plausible while harming biological calibration. Our first goal is not pretty fluorescence. It is correct expression localization and quantitative expression.

## Why This Should Work Better Than Raw Pix2Pix

Raw B-to-F regression has three failure modes:

```text
1. Sparse positives are overwhelmed by negative pixels.
2. Ambiguous bright debris/background is treated as signal.
3. The network learns average haze instead of confident expression.
```

The joint formulation fixes these:

```text
1. Mask loss explicitly learns the rare positive class.
2. Ignore masks remove ambiguous overexposed/debris pixels from the target.
3. Intensity regression is conditional on true expression.
4. Global expression head prevents positive hallucination in negative organoids.
5. Final F_hat remains continuous, so we keep expression degree, not only yes/no.
```

## Evaluation

We should report both mask and continuous fluorescence metrics.

Mask metrics:

```text
precision, recall, F1, F0.5, IoU, pixel AP
```

Global expression metrics:

```text
organoid-level AUROC
organoid-level AP
correlation between predicted and true total fluorescence
```

Continuous metrics:

```text
MAE / Huber on positive pixels
MAE / Huber on valid non-ignore pixels
Pearson/Spearman correlation of total expression per organoid
calibration plot: predicted total intensity vs measured total intensity
```

Visual checks:

```text
brightfield
raw fluorescence
clean target mask
predicted mask probability
predicted continuous fluorescence
overlay on brightfield
scatter of true vs predicted total expression
```

The key acceptance test is not just visual similarity. A useful model should:

```text
1. avoid whole-image pseudo-fluorescence,
2. avoid debris in the center when morphology is not cell-like,
3. localize the peripheral elongated cells,
4. preserve stronger vs weaker expression levels,
5. rank organoids correctly by total expression.
```

## Extension To Future Prediction

Once same-time `B_t -> M_t, A_t` works, future prediction becomes:

```text
B_1 ... B_k -> M_future, A_future
```

or:

```text
B_k -> M_future, A_future
```

The same output factorization should be used:

```text
future F_hat = future p * future a
```

This is better than predicting future raw fluorescence directly because the future signal is even more uncertain and sparse. First ask whether the organoid will express, then where, then how strongly.

Recommended future-prediction stages:

```text
Stage 1: Same-time B_t -> M_t, A_t
Stage 2: Late brightfield B_k -> late M_n, A_n
Stage 3: Early brightfield B_1 -> late M_n, A_n
Stage 4: Brightfield history B_1...B_k -> late M_n, A_n
```

Start with `B_k -> late F_n` where `k` is as late as possible but still before obvious fluorescence. Then move earlier only if the model is meaningfully above baseline.

## Implementation After Current Training Finishes

After the current mask-only tmux job finishes, the next code should:

```text
1. Reuse the cleaned mask target manifest.
2. Add continuous fluorescence target paths and summary statistics.
3. Add a joint dataset loader returning B, M, A, I, V.
4. Add an intensity head to the current GlobalGatedSegUNet.
5. Train with the composite loss above.
6. Save prediction panels with both mask and continuous F_hat.
7. Select best model by a mixed score:
   score = F0.5_mask + 0.1 * global_AUROC + 0.1 * Spearman(total_F_hat, total_F)
```

The first milestone should be simple and strict:

```text
Can B predict the cleaned fluorescence mask and rank total expression better than the old pix2pix model?
```

Only if yes, move to future prediction.

