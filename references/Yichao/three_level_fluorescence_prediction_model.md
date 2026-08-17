# Three-Level Fluorescence Prediction Model

## Summary

The final useful model should operate at three linked levels:

```text
Level 1: image / organoid level
Level 2: segmented pixel / region level
Level 3: continuous fluorescence intensity level
```

This is a better framing than pure pix2pix regression because fluorescence expression is sparse, biologically structured, and contaminated by debris/background artifacts.

## Level 1: Image Or Organoid Level

Question:

```text
Does this organoid crop contain real fluorescence-positive cells?
```

Target:

```text
G = 1 if the crop contains enough confident positive fluorescence pixels
G = 0 if the crop contains no reliable positive fluorescence signal
```

This is the meaning of image/organoid-level labels such as:

```text
positive crop
negative crop
overexposure-suppressed crop
```

Example:

```text
positive crop:
  contains a small but reliable positive peripheral cell region

negative crop:
  no confident real positive signal

overexposure-suppressed crop:
  raw fluorescence is globally bright or suspicious, so dense pseudo-signal is suppressed
```

Why this level matters:

```text
1. It prevents the model from hallucinating tiny positive speckles in truly negative organoids.
2. It gives a simple expression yes/no readout for each organoid.
3. It supports future prediction: can early brightfield predict whether this organoid will express later?
```

Output:

```text
global_expression_probability = P(G = 1 | brightfield)
```

Metrics:

```text
organoid-level AUROC
organoid-level average precision
classification accuracy at selected threshold
correlation with total measured fluorescence
```

## Level 2: Segmented Pixel Or Region Level

Question:

```text
Where are the real fluorescence-positive cell-like regions?
```

Target:

```text
M_x = 1 for confident positive pixels
M_x = 0 for valid non-positive pixels
I_x = 1 for ambiguous pixels that should be ignored during training/evaluation
```

This is the current segmentation task:

```text
brightfield B -> cleaned fluorescence mask M
```

Important distinction:

```text
An image-level positive crop does not mean all pixels are positive.
It only means the crop contains enough positive pixels somewhere.
```

Example:

```text
one positive organoid crop:
  5% positive pixels
  90% negative pixels
  5% ignored pixels

image-level label:
  positive
```

Example:

```text
one negative organoid crop:
  0% positive pixels
  98% negative pixels
  2% ignored pixels

image-level label:
  negative
```

Why this level matters:

```text
1. It forces the model to localize real cell-like expression instead of producing global haze.
2. It handles the sparse-positive class imbalance explicitly.
3. It allows debris/background pseudo-signal to be excluded through ignore masks.
4. It is the bridge between whole-organoid expression and continuous fluorescence reconstruction.
```

Output:

```text
mask_probability_map p_x = P(M_x = 1 | brightfield)
```

Metrics:

```text
pixel precision
pixel recall
F1
F0.5 for precision-first selection
IoU
pixel average precision
region-level detection/count metrics later if needed
```

## Level 3: Continuous Fluorescence Intensity Level

Question:

```text
If a region is truly positive, how strong is the fluorescence expression?
```

Target:

```text
A_x = background-corrected, transformed fluorescence intensity
```

Recommended target:

```text
F_corr = max(F_raw - robust_background, 0)
A = log(1 + scale * F_corr) / log(1 + scale)
```

The intensity target should be trained mainly inside confident positive regions:

```text
strong intensity loss where M_x = 1 and I_x = 0
background suppression where M_x = 0 and I_x = 0
no loss where I_x = 1
```

Why this level matters:

```text
1. Binary masks only say yes/no, not expression degree.
2. Continuous intensity makes the prediction biologically more realistic.
3. It allows total-expression quantification per organoid.
4. It supports future prediction of expression strength, not just future expression presence.
```

Output:

```text
intensity_map a_x = predicted fluorescence strength if positive
```

Final reconstructed fluorescence:

```text
F_hat_x = p_x * a_x
```

or for thresholded visualization:

```text
F_hat_x = 1[p_x > threshold] * a_x
```

Metrics:

```text
MAE / Huber on positive pixels
MAE / Huber on valid non-ignore pixels
Pearson/Spearman correlation of total fluorescence per organoid
calibration plot of predicted total expression vs measured total expression
visual comparison of predicted and true fluorescence
```

## Combined Model

The recommended architecture is one shared brightfield encoder with three heads:

```text
brightfield crop B
  -> shared encoder
      -> global expression head: G_hat
      -> mask head: p_x
      -> intensity head: a_x
```

The final prediction is:

```text
G_hat = image/organoid expression probability
p_x   = pixel/region expression probability
a_x   = positive fluorescence intensity
F_hat = p_x * a_x
```

This creates a consistent hierarchy:

```text
image level says whether expression exists
mask level says where expression exists
intensity level says how strong expression is
```

## Training Loss

Use a composite loss:

```text
L_total =
  lambda_global * L_global
  + lambda_mask * L_mask
  + lambda_intensity * L_intensity_positive
  + lambda_reconstruction * L_reconstruct
  + lambda_background * L_background_suppression
```

Where:

```text
L_global:
  BCE for whole-organoid positive/negative label

L_mask:
  focal BCE + Tversky/Dice for pixel positive mask

L_intensity_positive:
  Huber or L1 intensity loss inside confident positive pixels

L_reconstruct:
  Huber loss between p_x * a_x and cleaned fluorescence target

L_background_suppression:
  penalty for predicted fluorescence in valid negative pixels
```

## Why This Is The Right Next Step

Pure `B -> F` regression asks one model to solve three problems at once:

```text
1. Is this organoid positive?
2. Where is expression?
3. How strong is expression?
```

That is why earlier B2F predictions tended to be weak or blurry.

The three-level model decomposes the problem:

```text
1. First decide if expression is present.
2. Then localize the expression.
3. Then estimate expression strength.
```

This should reduce false positives from debris and overexposure while preserving a continuous fluorescence-like output.

## Extension To Future Differentiation Prediction

For future prediction, keep the same three output levels:

```text
early brightfield history B_1...B_k
  -> future global expression G_n
  -> future expression mask M_n
  -> future intensity A_n
  -> future fluorescence F_n = M_n * A_n
```

This is more useful than only predicting a future fluorescence image because it gives:

```text
1. whether the organoid will express,
2. where expression will appear,
3. how strong expression will become.
```

