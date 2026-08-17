# Yichao v2 Temporal Brightfield-to-Fluorescence Research Plan

Date: 2026-06-27

## Goal

The immediate v2 goal is to reproduce the successful v1 brightfield-to-fluorescence behavior on the new `DATA-Yichao-v2` data, but with a stricter data model:

1. Prepare a v2 instance-pair database from segmented single organoids.
2. Train a same-day model:

```text
B(Day X) -> F(Day X)
```

3. Extend the model to early prediction:

```text
B(Day X - N) -> F(Day X)
```

The early-prediction objective should encourage useful prediction before strong fluorescence expression appears, while not forcing impossible early predictions to look perfect.

## Current v2 Data State

Local raw root:

```text
/home/lachlan/ProjectsLFS/OrganoidAgent/DATA-Yichao-v2/
```

Copied LIFs:

| Folder | LIF | Day hint | Current state |
|---|---|---|---|
| `1_N39Rep_Globet_DF_D3` | `N39Rep_Globet_DF_D3.lif` | D3 | already extracted and grouped |
| `2_N39Rep_Globet_DF_D4` | `N39Rep_Globet_DF_D4.lif` | D4 | already extracted and grouped |
| `3_N39Rep_Globet_DF_D2` | `N39Rep_Globet_DF_D2.lif` | D2 | copied, metadata inspected |
| `4_N39Rep_Globet_DF_D3_1` | `N39Rep_Globet_DF_D3_1.lif` | D3 | copied, metadata inspected |
| `5_N39Rep_Globet_DF_D3_2` | `N39Rep_Globet_DF_D3_2.lif` | D3 | copied, metadata inspected |

Metadata summary:

| Folder | Series | Fields/series | Z | Time | Channels |
|---|---:|---:|---|---:|---:|
| `1_N39Rep_Globet_DF_D3` | 40 | 10 fields x 4 named acquisitions | 20-49 | 1 | 2 |
| `2_N39Rep_Globet_DF_D4` | 24 | 6 fields x 4 named acquisitions | 11-38 | 1 | 2 |
| `3_N39Rep_Globet_DF_D2` | 15 | 15 generic `Series###` | 1 | 1 | 7 |
| `4_N39Rep_Globet_DF_D3_1` | 10 | 10 generic `Series###` | 4-13 | 1 | 8 |
| `5_N39Rep_Globet_DF_D3_2` | 12 | 12 generic `Series###` | 12-31 | 1 | 8 |

The first two files are already previewed and likely use:

```text
input brightfield = ALEXA488 c1
target fluorescence = ALEXA488 c0
```

The newly copied D2/D3_1/D3_2 files are different. They have generic series names and 7/8 internal channels, so channel mapping must be visually confirmed before segmentation/training.

## Stage 1: v2 Dataset Preparation

The preparation should be deliberately separated from training.

### 1. Extract and inspect

Use:

```bash
bash analysis-tools/yichao_v2/run_yichao_v2_lif_prepare.sh
```

This script:

- inspects LIF metadata,
- extracts all JPEG planes if missing,
- groups extracted planes by position/series,
- does not segment,
- does not train.

Metadata-only command:

```bash
/home/lachlan/miniconda3/envs/organoid/bin/python \
  analysis-tools/yichao_v2/inspect_v2_lif_metadata.py \
  --v2-root DATA-Yichao-v2 \
  --output-dir analysis-outputs/yichao_v2_metadata
```

### 2. Confirm channel mapping

For the already processed named-acquisition D3/D4 files, the current practical mapping is:

```text
ALEXA488 c1 -> brightfield
ALEXA488 c0 -> green fluorescence
```

For D2/D3_1/D3_2, do not assume `c1` is brightfield or `c0` is fluorescence. Make a channel grid across all internal channels and choose:

- one brightfield-like input channel,
- one biologically relevant green/Alexa488 fluorescence channel,
- optional non-green channels as controls only.

### 3. Segment brightfield, not fluorescence

The segmentation source should remain brightfield-like images.

Recommended first pass:

```text
segmentation input = projected brightfield image
cropping targets = same organoid mask applied to brightfield and fluorescence
```

This preserves one instance crop pair:

```text
organoid_i:
  B crop
  F crop
  mask
  bbox
  day
  field/series
  z policy
  channel mapping
```

### 4. Database fields

The v2 database should preserve at least:

- `dataset_version = v2`
- `dataset_folder`
- `source_lif`
- `day`
- `field_id`
- `series_index`
- `acquisition_name`
- `brightfield_channel`
- `fluorescence_channel`
- `z_index` or `z_projection_policy`
- `object_id`
- `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`
- `is_edge_padded`
- `original_width`, `original_height`
- `crop_width`, `crop_height`
- `mask_area`
- `fluorescence_positive_fraction`
- `split_group_id`

`split_group_id` is critical. It should bind all views from the same biological field/sample/day family so that train/val/test splitting cannot leak repeated acquisitions of the same object.

## Stage 2: Same-Day v2 B2F Model

The first real model should mimic the working v1 task:

```text
B_i,d -> F_i,d
```

where:

- `i` is organoid instance,
- `d` is day,
- `B_i,d` is brightfield crop,
- `F_i,d` is fluorescence target crop.

The model should initially reuse the robust v1 design:

- UNet-like segmentation/regression model,
- input size 256,
- target is a background-suppressed continuous fluorescence signal,
- optional auxiliary positive-mask loss,
- balanced sampling so positive examples are not drowned by negative/background examples.

Loss:

```text
L_same = L_continuous(F_hat, F_clean) + alpha * L_mask(M_hat, M_clean)
```

where:

- `F_clean` is the denoised continuous target,
- `M_clean` is a soft or binary fluorescence-positive mask derived from `F_clean`,
- `alpha` controls how much the model focuses on positive signal localization.

This establishes whether v2 has enough signal and correct channel mapping before temporal prediction is attempted.

## Stage 3: Early-to-Later Fluorescence Prediction

The proposed extension is:

```text
B_i,k -> F_i,d, where k < d
```

or, if history is available:

```text
B_i,1:k -> F_i,d
```

For v2, the most conservative initial version is snapshot-based:

```text
B_i,k, delta=(d-k) -> F_i,d
```

The model gets the brightfield crop and a known horizon token `delta`, but it does not get any future fluorescence or future brightfield.

## No Future Leakage Rules

The temporal task is easy to contaminate. These rules are mandatory:

1. Split by biological unit before making temporal pairs.
2. All days/acquisitions from the same field or organoid lineage stay in the same split.
3. Never split random crops from the same field into both train and validation/test.
4. Input for horizon `k -> d` can use only data observed at or before day `k`.
5. Target generation can use `F_d`, but no statistic from validation/test targets can affect training normalization or threshold tuning.
6. Channel mapping must be fixed using training/inspection data before evaluating held-out data.
7. Hyperparameters and early-useful-day decisions must be selected on validation groups, not the test groups.

Practical split unit:

```text
split_group_id = donor/experiment + LIF + field/position + biological replicate
```

If individual organoids cannot be reliably tracked across days, use field-level or replicate-level splitting instead of pretending there is an organoid lineage.

## "Encourage Early, But Forgive If Impossible"

The model should not be punished equally for all horizons. Earlier days may not contain enough morphological information.

Use a horizon-aware objective:

```text
L = L_same + sum_{k<d} w(d-k) * L_future(k,d)
```

with:

```text
L_future(k,d) =
  L_continuous(F_hat_{k->d}, F_clean_d)
  + alpha * L_mask(M_hat_{k->d}, M_clean_d)
  + beta * L_total_expression(E_hat_{k->d}, E_d)
```

where:

- `E_d = mean(F_clean_d)` or total positive fluorescence per organoid,
- `w(delta)` is smaller for very early horizons,
- `w(delta)` can be increased later if validation shows useful prediction.

Two robust options:

### Option A: fixed forgiveness

```text
w(delta) = exp(-gamma * delta)
```

This makes near-future targets more important and earlier predictions softer.

### Option B: learned uncertainty

```text
L_delta = L_raw_delta / (2 * sigma_delta^2) + log(sigma_delta)
```

The model learns which horizons are intrinsically uncertain. This is useful when early day morphology sometimes predicts future fluorescence and sometimes does not.

## Earliest Useful Prediction Day

Do not choose the early day by intuition. Estimate it from validation performance.

For every candidate horizon:

```text
k -> d
```

measure:

- positive-region F1 or Dice,
- total fluorescence correlation,
- AUROC/AUPRC for organoid-level positive status,
- calibration of predicted expression amount.

Define the earliest useful day as:

```text
min k such that validation metric >= threshold and remains stable across replicates
```

Example:

```text
earliest useful = earliest day with:
  positive-region F1 >= 0.30
  and total-expression Spearman >= 0.50
  and organoid-level AUROC >= 0.75
```

Thresholds should be adjusted after seeing v2 validation distributions.

## Model Architecture Recommendation

Start simple and robust:

```text
input:
  brightfield crop B
  optional horizon embedding delta

backbone:
  UNet / ResUNet encoder-decoder

outputs:
  clean fluorescence-like target F_hat
  optional positive mask M_hat
  optional total expression E_hat
```

For history input later:

```text
B_i,1:k -> temporal encoder -> UNet decoder
```

but do not start there. First prove that a single earlier snapshot has predictive value.

## Evaluation Hierarchy

Same-day:

- Does v2 reproduce v1-level B2F behavior?
- Does `ALEXA488 c1 -> ALEXA488 c0` work?
- Are the generic 7/8-channel files mapped correctly?

Early-to-later:

- Can D2 brightfield predict D3/D4 expression above baseline?
- Can D3 brightfield predict D4 expression better than D2?
- At which earliest day does prediction become useful?

Baselines:

- predict zero fluorescence,
- predict train-set mean fluorescence,
- predict same-day nearest morphology baseline,
- v1-trained model evaluated on v2 without finetuning.

## Current Decision

Do not train immediately. The next safe steps are:

1. Extract and preview all v2 channels.
2. Confirm channel mapping for D2/D3_1/D3_2.
3. Build v2 instance database only after mapping is confirmed.
4. Train same-day v2 B2F first.
5. Only then train horizon-aware early-to-later prediction.
