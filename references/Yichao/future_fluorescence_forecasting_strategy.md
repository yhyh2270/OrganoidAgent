# Future Fluorescence Forecasting From Early Brightfield

This note describes the real modeling goal for the Yichao differentiation data:

- not only same-time `brightfield -> fluorescence`
- but early `brightfield history -> future fluorescence expression`

Current segmentation/database convention for the instance-pair pipeline is:

- `c1`: brightfield input
- `c0`: fluorescence target

Always verify channel naming in any older note before using it.

## Core Idea

Train one forecasting model that accepts a variable-length early brightfield prefix and predicts future fluorescence outcomes:

- `M(D1) -> future expression`
- `M(D1, D2) -> future expression`
- `M(D1, D2, ..., Dk) -> future expression`

Use the same model for all prefix lengths by randomly sampling the prefix length during training. This is simpler and more robust than training one model per `k`.

The first target should be a robust future-expression summary, not a full image:

- future positive or negative expression
- future peak fluorescence
- future fluorescence AUC
- future onset time
- future positive-cell/organoid fraction

Only after these scalar or low-dimensional targets work should we train a future-fluorescence image generator.

Reason: exact future pixel location is noisy because organoids grow, move, deform, and change z-plane appearance. Predicting "will this organoid express, when, and how strongly" is the scientifically important first target and is much easier to validate.

## Required Data Unit

The correct forecasting unit is an organoid track, not an isolated image pair.

One training row should represent:

- dataset
- experiment/day group
- position
- organoid track id
- time index
- z index or z projection id
- early brightfield crop sequence
- future fluorescence labels

Use only complete organoids for the first pass:

- `is_edge_padded = 0`

Avoid edge-padded crops until the model is stable.

## Tracking Strategy

Use a simple tracking pipeline first:

1. Segment every brightfield frame, as already done.
2. Keep complete organoid instances only.
3. For each dataset and position, link instances over time by centroid distance and mask/box overlap.
4. Use the Hungarian algorithm or greedy nearest-neighbor matching with a maximum displacement threshold.
5. Break a track if no confident match exists.
6. Keep tracks with enough early frames and enough future frames.

For z-stacks, do not start with full 3D tracking. Start with one of these simpler choices:

- central z-plane per timepoint
- best-focus z-plane per timepoint
- max or mean brightfield projection per timepoint
- per-z independent tracks only if the z-plane identity is stable

The first robust model should use a single 2D representation per organoid per timepoint.

## Finding the Expression Onset `k`

For every organoid track, compute a background-corrected fluorescence signal:

```text
F_t = mean or p90 fluorescence inside organoid mask - local background fluorescence
```

Also compute:

- positive pixel fraction inside the organoid mask
- max fluorescence inside the organoid mask
- fluorescence AUC after smoothing

Define expression onset as the first timepoint where fluorescence becomes positive and stays positive:

```text
onset = first t where F_t > baseline + threshold
        and the condition holds for at least 2 consecutive future frames
```

Recommended threshold:

```text
threshold = max(global_negative_p95, local_background_median + 3 * local_background_MAD)
```

Then define the pre-expression prefix:

```text
usable early input = frames t <= onset - guard
```

Use a guard of 1 or 2 frames so the model cannot cheat from weak fluorescence already appearing.

If a track never expresses, keep it as a negative example with:

- `future_positive = 0`
- `peak_future_fluorescence = observed peak`
- `onset = censored`

## Choosing the Best `k`

Do not choose one fixed `k` manually at the beginning.

Train a variable-prefix model and evaluate performance by prefix length:

| Input prefix | Question |
| --- | --- |
| `D1` only | Can morphology at the earliest time predict future expression? |
| `D1..D2` | Does one extra day improve prediction? |
| `D1..D3` | Does the model need more growth history? |
| `D1..Dk before onset` | What is the latest non-cheating pre-expression prediction? |

Select the shortest prefix that reaches a useful validation performance, for example:

```text
shortest k whose AUROC / Spearman / MAE is within 95% of the best longer-prefix model
```

This gives a practical biological answer:

```text
"The earliest reliable prediction is possible from D1..D2."
```

or:

```text
"D1 alone is not enough; growth between D1 and D2 is required."
```

## Model Architecture

Use a simple two-stage model.

### Stage 1: Brightfield Encoder

Apply the same image encoder to every early brightfield crop:

```text
brightfield crop at time t -> CNN / ViT encoder -> feature vector e_t
```

Good first choices:

- ResNet/ConvNeXt encoder trained from scratch
- DINOv2 or self-supervised encoder if available
- the pix2pix encoder reused after same-time pretraining

Also concatenate explicit morphology features:

- area
- diameter
- circularity
- aspect ratio
- lumen or hole fraction
- edge sharpness
- texture statistics
- growth rate
- change in area
- change in circularity
- budding or protrusion score if available

This creates:

```text
x_t = [learned_image_embedding_t, morphology_features_t, delta_time_t]
```

### Stage 2: Temporal Aggregator

Use a small temporal model:

- GRU for the first implementation
- Temporal Transformer later if needed
- attention pooling for variable-length prefixes

Input:

```text
x_1, x_2, ..., x_k
```

Outputs:

- `future_positive`: binary classification
- `future_peak`: regression
- `future_auc`: regression
- `future_onset_bin`: ordinal classification
- optional `future_fluorescence_curve`: sequence regression

This is the recommended first model:

```text
Brightfield encoder + morphology features + GRU + multi-task heads
```

It is simple, variable-length, easy to debug, and does not require exact future image alignment.

## Training Objective

Use multi-task loss:

```text
loss =
  BCE(future_positive)
  + alpha * SmoothL1(log1p(future_peak))
  + beta  * SmoothL1(log1p(future_auc))
  + gamma * ordinal_loss(future_onset_bin)
```

Use `log1p` for fluorescence intensity targets because fluorescence often has a long tail.

Sample random prefixes during training:

```text
for each track:
    choose a prefix ending before onset - guard
    predict future targets after that prefix
```

This directly teaches:

- `M(D1)`
- `M(D1, D2)`
- `M(D1, ..., Dk)`

with one model.

## Optional Future Image Model

After the scalar forecasting model works, add future image prediction.

Recommended target:

- future fluorescence crop at a chosen horizon
- or future max-fluorescence projection
- or a coarse downsampled fluorescence heatmap

Architecture:

```text
early brightfield sequence encoder -> latent vector -> U-Net decoder -> future fluorescence image
```

Use the scalar model output as an auxiliary task. The image decoder should not be trained alone.

Image losses:

- mask-restricted L1
- SSIM
- positive-region Dice or focal loss after thresholding fluorescence
- correlation inside organoid mask

Do not judge this model only by PSNR. Fluorescence is sparse, so pixel-average metrics can look good while missing the biology.

## Evaluation Design

Splits must be track-level and position-level:

- all frames of one organoid track stay in the same split
- all tracks from a held-out position should preferably stay in the same split
- final test should be held-out positions or held-out LIF files

Do not split adjacent frames from the same position across train and test.

Metrics:

- future positive classification: AUROC, AUPRC, F1, sensitivity at fixed specificity
- future peak/AUC regression: Spearman, Pearson, MAE, R2
- onset prediction: mean absolute error in hours/days, early/late confusion matrix
- calibration: Brier score and reliability curve
- image prediction if used: mask-restricted MAE, SSIM, fluorescence-threshold Dice, within-mask Pearson

Always compare to simple baselines:

- predict by current brightfield area only
- predict by current growth rate only
- predict by previous fluorescence if fluorescence is allowed
- predict by dataset/position average
- random forest or XGBoost on explicit morphology features

The neural model is useful only if it beats these simple baselines on held-out positions.

## Feature Interpretation

Use two interpretation routes.

### Explicit Feature Route

Train a simple model on handcrafted features:

- area
- growth rate
- circularity
- lumen fraction
- edge sharpness
- texture
- budding score

Then inspect:

- feature importance
- SHAP values
- monotonic relation between feature and future expression

This answers which measurable morphology features predict expression.

### Image Model Route

For the image encoder:

- Grad-CAM
- occlusion sensitivity
- attention map visualization
- compare high-risk and low-risk organoids with matched size

The key biological question is:

```text
Which pre-expression morphology predicts later differentiation?
```

Candidate useful signals:

- growth rate before expression
- organoid size
- epithelial ring thickness
- lumen or hollow-core pattern
- edge smoothness or roughness
- protrusion or budding morphology
- texture heterogeneity
- whether the organoid is fused, cystic, or fragmented

## Practical Implementation Order

1. Finish the complete instance-pair database for all Yichao datasets.
2. Add an organoid-track table:
   - `track_id`
   - `dataset`
   - `position`
   - `time_index`
   - `z_index_or_projection`
   - `instance_id`
   - centroid and bbox
   - complete/non-edge flag
3. Add fluorescence summary fields per track/time:
   - background-corrected mean
   - p90
   - positive fraction
   - peak
4. Add onset labels:
   - `future_positive`
   - `onset_time_index`
   - `time_to_onset`
   - `future_peak`
   - `future_auc`
5. Train feature-only baselines first.
6. Train image-sequence model:
   - brightfield encoder
   - morphology features
   - GRU
   - multi-task heads
7. Evaluate by prefix length:
   - `D1`
   - `D1..D2`
   - `D1..D3`
   - all pre-expression frames
8. Only then train future image prediction.

## Minimal First Experiment

The first useful experiment should be:

```text
Input:
    complete organoid brightfield crops from early frames before fluorescence onset

Target:
    future_positive
    log1p(future_peak)
    log1p(future_auc)

Model:
    CNN encoder + morphology features + GRU

Split:
    held-out positions

Decision:
    find the shortest prefix length that predicts future expression above baseline
```

This directly addresses the real goal while staying simple enough to implement and debug.

## Main Risks

The largest risk is not model architecture. The largest risk is label construction.

Critical checks:

- Are the same organoids trackable across the requested days?
- Do `D1`, `D2`, `D3` refer to the same physical position or separate acquisitions?
- Is fluorescence truly absent in the input window?
- Does the target fluorescence include debris or background artifacts?
- Are positive and negative tracks balanced enough?
- Are train/test splits separated by position or LIF file?

If same-organoid tracking across days is not reliable, switch the first task to position-level or population-level forecasting:

```text
early brightfield population in one position -> later expression burden in that position
```

This is less precise but still biologically useful and more robust than false per-organoid labels.
