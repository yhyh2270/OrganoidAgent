# Context-Aware Fluorescence Segmentation Strategy

## Purpose

The current B2F model is useful, but its main failure mode is clear: it can predict too much green. The result improved after switching to the pix2pix-style U-Net and fixing training, but it still behaves like a fluorescence image reconstruction model. The better formulation is:

```text
Brightfield organoid image -> fluorescence-positive segmentation mask
```

This is not just pixel-to-pixel regression. It is a context-aware dense classification problem:

```text
For every pixel x, estimate P(pixel x is truly fluorescent | the whole brightfield crop).
```

A pixel's fluorescence status is predicted at that pixel, but the decision should depend on local texture, neighboring pixels, organoid boundary geometry, peripheral cell morphology, and whole-organoid state. This is close to semantic segmentation, not simple per-pixel correlation.

## Current Evidence

Finished run:

```text
/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_future_expression/stage1_b2f_pix2pix_384_v1_noamp_long
```

Report:

```text
/home/lachlan/ProjectsLFS/OrganoidAgent/publication/yichao_b2f_pix2pix_results_report/main.pdf
```

Current held-out test result:

```text
signal_f1 = 0.387
signal_precision = 0.266
signal_recall = 0.710
expression_auc = 0.879
masked_mean_prediction_correlation = 0.692
```

Interpretation:

- The model has learned something real: image-level expression AUROC is strong.
- The model is over-inclusive at pixel level: recall is high, precision is low.
- Therefore the next method should not simply train longer or make a larger pix2pix generator.
- The next method should explicitly optimize fluorescence-positive segmentation and false-positive suppression.

## Terminology

This task is best described as:

```text
context-aware fluorescence-positive semantic segmentation
```

Alternative valid names:

- Dense binary classification with global context.
- Weakly/noisily supervised fluorescence segmentation.
- Brightfield-conditioned fluorescence-positive mask prediction.
- Multi-task segmentation with image-level expression gating.

It is not best described as ordinary multivariable correlation. The output is a structured image, and the target at one pixel can depend on features elsewhere in the same crop. Mathematically:

```text
p_x = P(Y_x = 1 | B_1, B_2, ..., B_n)
```

where `B` is the full brightfield crop, `Y_x` is the fluorescence-positive label at pixel `x`, and `p_x` is the predicted probability.

## Why Pix2Pix Reconstruction Is Not Enough

Pix2pix-style reconstruction learns:

```text
B -> F_intensity
```

This optimizes visual/intensity similarity. That is useful as a baseline, but it has three weaknesses for this project:

- Fluorescence is sparse, so ordinary reconstruction losses can be dominated by background.
- Raw fluorescence contains debris and strong artifacts that are not true cells.
- A visually plausible green prediction is not the same as a correct positive-cell classification.

The biological question is closer to:

```text
Which pixels or cell-like regions are truly fluorescence-positive?
```

So the target should be a cleaned positive mask, not only raw fluorescence intensity.

## Important Biological Constraint From Yichao

Yichao's note says that real positive cells have morphology:

- True cells are often elongated and peripheral.
- Many have clear boundaries.
- Some have small hollow regions, especially near edges.
- Strong green central blobs or debris can be non-cell material.
- Background/debris may have stronger fluorescence than cells but should be excluded if it lacks cell morphology.

This means the model must learn:

```text
fluorescence intensity + brightfield morphology -> true positive cell-like signal
```

not:

```text
all bright fluorescence -> positive
```

## The Killer Method: Global-Gated Context Segmentation

The recommended next model has one shared brightfield encoder and two heads:

```text
1. Pixel head:
   s(x) = P(pixel x is fluorescence-positive | local + context features)

2. Global head:
   g = P(the organoid crop contains true fluorescence-positive cells | whole crop)
```

The final pixel probability is:

```text
p(x) = g * s(x)
```

or equivalently in logit space:

```text
logit p(x) = z_pixel(x) + z_global
```

Why this helps:

- The pixel head can localize candidate fluorescent structures.
- The global head suppresses false positive masks when the entire organoid does not look positive.
- This directly attacks the current failure mode: too much green area.

## Input and Targets

### Inputs

Minimum input:

```text
B: brightfield crop, 1 channel
```

Recommended input:

```text
[B, M]
```

where `M` is the organoid instance mask. Adding `M` tells the model which pixels belong to the segmented organoid and reduces background confusion.

Optional input:

```text
[B, M, distance_to_boundary]
```

The distance-to-boundary map can help because true cells are often peripheral.

### Target Construction

Raw fluorescence should be converted into a training mask:

```text
F_raw -> Y_positive, Y_ignore, Y_negative
```

Recommended target rules:

1. Estimate local background from pixels outside or near the organoid mask.
2. Define candidate positive pixels by local corrected fluorescence:

```text
F_corrected(x) = F_raw(x) - background_local
Y_candidate(x) = 1 if F_corrected(x) > threshold
```

3. Restrict candidate positives to organoid/cell-like regions.
4. Mark ambiguous regions as `ignore`, not positive or negative.
5. Mark strong non-cell debris as negative or ignore, depending on confidence.

This tri-state target is important:

- Positive: confident true fluorescence-positive cell-like signal.
- Negative: confident non-positive pixels.
- Ignore: ambiguous debris, saturated artifacts, uncertain boundaries.

The loss should not punish the model on ignored pixels.

## Fluorescence Can Be Discontinuous

The positive mask may be spatially discontinuous:

- Expression can occur in scattered cell-like regions.
- Thin elongated structures may be separated.
- The central organoid material may be non-positive.
- Boundary cells may be positive while neighboring pixels are not.

Therefore, the model should not enforce excessive smoothness. Avoid using only L1/MSE intensity regression or strong smoothness penalties. Use segmentation losses that permit sparse/discontinuous regions:

- Focal loss for rare positive pixels.
- Dice/Tversky loss for overlap.
- Boundary-aware or edge-weighted terms only as secondary constraints.
- Ignore masks for uncertain pixels.

## Model Architecture

Recommended architecture:

```text
Input [B, M]
  -> encoder
  -> multi-scale context block
  -> decoder
  -> pixel logits z_pixel(x)
  -> global pooling head z_global
  -> final p(x)
```

### Encoder

Use a U-Net style encoder-decoder as the baseline. U-Net is still the practical standard for biomedical segmentation because it combines localization from skip connections with context from the encoder.

### Context Block

Add a multi-scale context block at the bottleneck:

```text
dilated conv rates: 1, 2, 4, 8
```

or a lightweight ASPP block. This allows each pixel prediction to use a broader field of view without needing a very deep model.

Why this matters:

- A pixel may not be classified correctly from its local texture alone.
- The model needs to know whether the organoid is peripheral, hollow, elongated, debris-like, or globally positive.

### Global Gate

The global head is:

```text
h = global_average_pool(encoder_features)
g = sigmoid(MLP(h))
```

Use it for:

- Image-level expression classification.
- Multiplying or biasing pixel-level probabilities.
- Calibration and false-positive suppression.

## Loss Function

Use a multi-task loss:

```text
L = L_pixel + lambda_g L_global + lambda_mil L_mil + lambda_area L_area
```

### Pixel Loss

Use focal BCE on confident pixels:

```text
L_focal = - alpha (1 - p)^gamma Y log(p)
          - (1 - alpha) p^gamma (1 - Y) log(1 - p)
```

This handles rare positives and discourages easy background pixels from dominating.

### Tversky / Dice Loss

Use Tversky loss to explicitly control false positives and false negatives:

```text
Tversky = TP / (TP + alpha FP + beta FN)
L_tversky = 1 - Tversky
```

For the current model, false positives are the main problem. Use:

```text
alpha > beta
```

Example:

```text
alpha = 0.7
beta = 0.3
```

This penalizes predicted green area that is not truly positive.

### Global Classification Loss

Define image-level label:

```text
y_global = 1 if positive mask area > threshold else 0
```

Then:

```text
L_global = BCE(g, y_global)
```

This trains the global gate directly.

### MIL Consistency Loss

The pixel map and global label should agree. Use soft max pooling:

```text
q = softmax_pool(p(x)) = sum_x p(x) exp(k p(x)) / sum_x exp(k p(x))
L_mil = BCE(q, y_global)
```

This says:

- If the image is globally negative, no pixel should be strongly positive.
- If the image is globally positive, at least some pixels should be positive.

### Area Prior / Calibration Loss

For examples with reliable fluorescence masks, regularize predicted positive area:

```text
A_pred = mean_x p(x)
A_true = mean_x Y(x)
L_area = SmoothL1(log(A_pred + eps), log(A_true + eps))
```

This discourages over-predicting green area.

## Thresholding and Calibration

Do not use a fixed threshold blindly. Select the probability threshold on validation data:

```text
t* = argmax_t F_beta(t)
```

Because false positives are problematic, use:

```text
F_0.5
```

or select the smallest threshold satisfying:

```text
precision >= target_precision
```

Example target:

```text
precision >= 0.60
```

Then report recall at that precision.

## Metrics

Main segmentation metrics:

- Pixel precision.
- Pixel recall.
- Pixel F1.
- Pixel IoU.
- Precision-recall curve.
- Average precision.
- Area calibration: correlation between predicted and true positive area.

Main image-level metrics:

- Expression AUROC.
- Expression average precision.
- Balanced accuracy.
- Recall at fixed false-positive rate.

Important reporting rule:

```text
Do not judge by image MAE alone.
```

Image MAE can look good even if the model predicts dark or broad blurry fluorescence.

## Validation Design

Split by experiment/position/track, not by random crop. Otherwise adjacent time points or z projections can leak into validation.

Recommended split key:

```text
dataset | experiment_design | position_label | track_id
```

Validation should include:

- High-expression examples.
- Low-expression examples.
- Strong-debris examples.
- Background/artifact examples.
- Early weak-expression examples.

The current per-dataset result shows a major issue:

```text
Data-Yichao-9: strong performance
Data-Yichao-3/5/6/7: weak precision / over-prediction
```

So future validation must report per-dataset and per-stage performance.

## Hard Negative Mining

This is likely essential.

After a first segmentation model is trained:

1. Run inference on validation/train data.
2. Find high-confidence false positives:

```text
p(x) high, Y(x)=0
```

3. Save those crops/regions as hard negatives.
4. Retrain with oversampling of hard-negative examples.

This directly teaches the model:

```text
strong-looking morphology or artifacts are not always true fluorescence-positive cells
```

Hard negatives should include:

- Central green debris.
- Background impurity.
- Non-cellular saturated fluorescence.
- Brightfield structures that resemble positive regions but have no true target.

## Recommended Training Stages

### Stage 0: Target Audit

Generate target masks from fluorescence and visualize:

```text
brightfield | raw fluorescence | cleaned target mask | ignored pixels
```

Do not train until these masks look biologically reasonable.

### Stage 1: Overfit Test

Overfit 16 to 32 examples:

```text
global-gated segmentation model
focal + Tversky + global BCE
```

Required result:

```text
training pixel F1 > 0.85
```

If it cannot overfit, the target construction or model implementation is wrong.

### Stage 2: Small Validation Run

Train on a subset and validate:

```text
target: improve precision at useful recall
```

Use early stopping on validation average precision or F0.5.

### Stage 3: Full Run

Train on all projected complete instances with:

- Hard-negative sampling.
- Image-level balanced sampling.
- Pixel-level class imbalance loss.
- Validation threshold calibration.

### Stage 4: Future Prediction

Only after same-time segmentation is credible:

```text
B_early -> future global expression
B_early -> future positive area
B_early history -> future expression timing
```

The B2F segmentation encoder can become the morphology encoder for future prediction.

## Minimal Implementation Plan

New folder:

```text
differentiation_prediction/yichao_fluorescence_segmentation/
```

Suggested files:

```text
build_clean_targets.py
datasets.py
models.py
losses.py
train_segmentation.py
evaluate_segmentation.py
make_report_artifacts.py
resume_segmentation_tmux.sh
```

Database additions:

```text
positive_mask_path
ignore_mask_path
target_positive_area_fraction
global_positive_label
hard_negative_score
threshold_used
target_generation_version
```

Output artifacts:

```text
analysis-outputs/yichao_fluorescence_segmentation/
  targets/
  runs/
  reports/
  databases/
```

## Recommended Model Defaults

```text
input_channels = 2  # brightfield + organoid mask
image_size = 384
encoder_channels = [32, 64, 128, 256, 384]
context = ASPP / dilated conv rates [1, 2, 4, 8]
decoder = U-Net skip decoder
pixel_head = 1 logit channel
global_head = 1 logit
final_probability = sigmoid(pixel_logit + global_logit)
```

Loss defaults:

```text
focal_alpha = 0.75
focal_gamma = 2.0
tversky_alpha = 0.7
tversky_beta = 0.3
lambda_focal = 1.0
lambda_tversky = 1.0
lambda_global = 0.3
lambda_mil = 0.2
lambda_area = 0.05
```

Training defaults:

```text
optimizer = AdamW
lr = 1e-4 to 2e-4
batch_size = 8 if 384 px fits GPU
epochs = 300 maximum
early_stop_metric = validation F0.5 or average precision
patience = 15 validation checks
no AMP until stability is proven
```

## Expected Improvement

Compared with the current pix2pix reconstruction model, this method should improve:

- Pixel precision.
- False-positive control.
- Interpretability.
- Image-level consistency.
- Robustness to low-expression and debris-heavy cases.

It may reduce recall at first. That is acceptable if precision improves. The correct target is not maximum green coverage; the target is biologically plausible fluorescence-positive cell regions.

## What Would Prove It Works

Minimum useful target:

```text
test signal precision > 0.50
test signal recall > 0.50
test signal F1 > 0.50
expression AUROC remains > 0.85
```

Better target:

```text
test signal precision > 0.65
test signal recall > 0.55
test signal F1 > 0.60
```

Most important qualitative check:

```text
The model should stop painting broad green regions on low-expression organoids.
```

## Why This Is Realistic

The current B2F result already shows image-level signal:

```text
expression_auc = 0.879
```

So brightfield morphology does contain information associated with fluorescence status. The problem is pixel localization and false positives, not total absence of signal. A segmentation/classification formulation directly optimizes the missing part.

## Primary Research References

- U-Net: biomedical encoder-decoder segmentation with skip connections. https://arxiv.org/abs/1505.04597
- pix2pix: conditional image-to-image translation baseline. https://arxiv.org/abs/1611.07004
- Focal Loss: class-imbalance-aware dense classification. https://arxiv.org/abs/1708.02002
- DeepLab / atrous convolution context: multi-scale context for dense prediction. https://arxiv.org/abs/1606.00915
- Generalized Dice loss: segmentation loss for highly imbalanced labels. https://arxiv.org/abs/1707.03237
- Tversky loss: false-positive/false-negative weighted segmentation loss. https://arxiv.org/abs/1706.05721
- Attention U-Net: attention gates for medical image segmentation. https://arxiv.org/abs/1804.03999

## Bottom Line

The next substantial step is:

```text
Do not train B -> raw F as the main task.
Train B -> cleaned fluorescence-positive mask with global gating.
```

Then keep raw fluorescence intensity as an auxiliary head only:

```text
Main head: true positive cell-like fluorescence mask
Auxiliary head: fluorescence intensity where positive
Global head: organoid-level expression status
```

This better matches the biology, the observed failure mode, and the final future-prediction objective.
