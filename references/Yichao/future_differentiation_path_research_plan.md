# Future Differentiation Path Prediction Research Plan

This note refines the goal from "predict same-time fluorescence from brightfield" to the biologically useful task:

```text
early brightfield morphology and growth history -> future differentiation / future fluorescence expression
```

The practical strategy is two-stage:

1. First test whether differentiated, fluorescent organoids have brightfield morphology that is distinguishable from non-fluorescent organoids at the same or late time point.
2. Only if that feasibility gate works, train models that predict future fluorescence or future expression outcomes from earlier brightfield frames before fluorescence is visible.

The first stage is not the final biological question, but it is the necessary sanity check. If current/late brightfield cannot distinguish fluorescent from non-fluorescent differentiated states, then future prediction from earlier brightfield is unlikely to be reliable without additional labels, higher-resolution imaging, or a clearer positive-control dataset.

## Current Yichao Data State

Use the projected-z database as the main starting point:

```text
/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_projected_instance_pairs/database/projected_instance_pairs.sqlite
```

This database was generated from z-projected brightfield and fluorescence pairs:

```text
c1 brightfield projection: min projection across z
c0 fluorescence projection: max projection across z
```

The relevant tables are:

```text
projected_images
projected_instances
```

Main counts:

| Quantity | Count |
| --- | ---: |
| Projected image pairs | 3432 |
| Segmented organoid pairs | 15619 |
| Complete non-edge organoid pairs | 9380 |
| Complete non-edge organoid pairs with time series | 9370 |

Per-dataset complete non-edge organoids:

| Dataset | Projected images | Time-aware images | Complete non-edge organoids | Time index range |
| --- | ---: | ---: | ---: | --- |
| Data-Yichao-1 | 5 | 0 | 5 | 0 |
| Data-Yichao-2 | 101 | 96 | 306 | 0-15 |
| Data-Yichao-3 | 528 | 528 | 1820 | 0-48 |
| Data-Yichao-4 | 369 | 369 | 886 | 0-48 |
| Data-Yichao-5 | 504 | 504 | 830 | 0-44 |
| Data-Yichao-6 | 135 | 135 | 429 | 0-44 |
| Data-Yichao-7 | 255 | 255 | 586 | 0-37 |
| Data-Yichao-8 | 654 | 654 | 1541 | 0-44 |
| Data-Yichao-9 | 348 | 348 | 946 | 0-115 |
| Data-Yichao-10 | 533 | 533 | 2031 | 0-40 |

Recommended first filters:

```sql
where projected_instances.is_edge_padded = '0'
```

For goblet/N39 differentiation modeling, treat `Data-Yichao-9` separately because it is the PDO28/Jurkat T-cell live-imaging task, not the same differentiation task.

## Literature Basis

Several lines of work support this staged approach.

Image-to-image translation is technically reasonable for same-time brightfield-to-fluorescence prediction. The original pix2pix paper introduced conditional GANs for paired image translation, combining a reconstruction term with an adversarial discriminator. This is useful when the input and target are spatially aligned image pairs. See Isola et al., CVPR 2017: <https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html>.

Label-free fluorescence prediction from transmitted-light microscopy is established, but it is not guaranteed for every marker or imaging setup. Ounkomol et al. showed 3D fluorescence prediction from transmitted-light microscopy, and Christiansen et al. showed in silico labeling of fluorescent labels from unlabeled microscopy images. These papers justify trying brightfield-to-fluorescence, but also imply the need for careful marker-specific validation. Sources: <https://www.nature.com/articles/s41592-018-0111-2> and <https://research.google/pubs/in-silico-labeling-predicting-fluorescent-labels-in-unlabeled-images/>.

The closest conceptual precedent is the PSC differentiation paper by Yang et al. They first used pix2pix to predict cTnT fluorescence from mature-stage brightfield images, then used weakly supervised learning on earlier brightfield images to recognize committed precursor regions and predict later differentiation efficiency. This is very close to the two-stage Yichao plan. Source: <https://www.nature.com/articles/s41421-023-00543-1>.

Organoid brightfield morphology can carry biologically relevant information, but it must be quantified carefully. OrganoSeg showed that brightfield morphometry of 3D organoids can stratify organoid phenotypes and link morphology to biological variation. OrganoID and OrganoidTracker are useful precedents for single-organoid detection and tracking in time-lapse experiments. Sources: <https://www.nature.com/articles/s41598-017-18815-8>, <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010584>, and <https://pubmed.ncbi.nlm.nih.gov/33091031/>.

For fluorescence reconstruction, Rittscher et al. emphasize that deep-learning fluorescence prediction has important limits and validation requirements. This matters here because an attractive synthetic fluorescence image can still be biologically wrong. Source: <https://www.nature.com/articles/s41592-019-0458-z>.

A recent retinal-organoid preprint is directly aligned with the long-term goal: predicting later tissue outcomes in organoids from earlier time-lapse images. It is useful as conceptual evidence, but because it is a preprint it should not be treated as final peer-reviewed evidence. Source: <https://doi.org/10.1101/2025.02.19.639061>.

## Stage 1: Feasibility Gate

Question:

```text
Given a brightfield crop B_t from a differentiated or late-stage organoid,
can we infer whether the corresponding fluorescence F_t is positive and where it appears?
```

This should be tested before future prediction. It answers whether the fluorescence-positive state has a visible brightfield phenotype.

Recommended labels from fluorescence:

```text
future_positive_t = 1 if corrected fluorescence exceeds threshold
F_peak_t = high percentile fluorescence inside organoid mask
F_auc_t = total corrected fluorescence inside organoid mask
F_positive_fraction_t = fraction of mask pixels above threshold
```

Recommended models, in increasing complexity:

| Model | Input | Target | Purpose |
| --- | --- | --- | --- |
| Logistic regression / XGBoost | morphology features | positive/negative | Check whether explicit morphology is enough |
| CNN classifier | brightfield crop | positive/negative | Check if image texture carries signal |
| U-Net regression | brightfield crop | fluorescence crop | Simple same-time B2F baseline |
| pix2pix | brightfield crop | fluorescence crop | Sharper same-time B2F prediction |
| Multi-task model | brightfield crop | positive + peak + fluorescence image | More robust than image-only prediction |

Do not judge feasibility only by visual similarity. Use object-level metrics:

```text
AUROC / AUPRC for positive vs negative organoids
Spearman or Pearson correlation for total fluorescence
MAE or RMSE for log fluorescence intensity
Calibration curve for predicted positivity probability
Pixel-level PCC / SSIM only as secondary metrics
```

Failure criteria:

```text
If a clean held-out-position split gives AUROC < 0.65
and predicted fluorescence collapses to average-looking fluorescence,
then brightfield morphology at the expressed stage is probably weak or noisy.
```

If Stage 1 fails, do not start future prediction yet. Instead:

```text
ask for a positive-control dataset with clear fluorescent cells
label debris/background examples explicitly
improve segmentation and mask filtering
use higher magnification or better focus if possible
```

## Fluorescence Histogram And Onset Analysis

Before future modeling, calculate fluorescence expression histograms over time. This determines when expression starts and defines the non-cheating early input window.

For every complete organoid instance:

```text
B_i,t = projected brightfield crop
F_i,t = projected fluorescence crop
M_i,t = organoid mask crop
```

Compute robust fluorescence features:

```text
inside_i,t = fluorescence pixels where M_i,t = 1
background_i,t = fluorescence pixels in an annulus around the organoid or crop pixels outside the mask

F_mean_i,t = mean(inside_i,t) - median(background_i,t)
F_p90_i,t = p90(inside_i,t) - median(background_i,t)
F_p99_i,t = p99(inside_i,t) - median(background_i,t)
F_total_i,t = sum(max(inside_i,t - median(background_i,t), 0))
F_pos_frac_i,t = fraction of mask pixels above background threshold
```

Use log scaling for histograms:

```text
S_i,t = log1p(max(F_p90_i,t, 0))
```

Recommended plots:

```text
histogram of S_i,t by dataset/day/time
ridge plot of fluorescence distributions across time
positive fraction over time
median and p90 fluorescence over time
onset-time histogram per organoid track
heatmap: dataset x time -> positive fraction
```

Threshold definition:

```text
negative_distribution = earliest time points and/or manually negative organoids
threshold = max(p99(negative_distribution), median(background) + 3 * MAD(background))
```

Per-organoid expression onset:

```text
onset_i = first t where
  F_pos_frac_i,t > positive_fraction_threshold
  and F_p90_i,t > threshold
  and the condition is sustained for at least 2 consecutive frames
```

Population critical date:

```text
T_5  = first time where >= 5% of tracked organoids are positive
T_10 = first time where >= 10% of tracked organoids are positive
T_25 = first time where >= 25% of tracked organoids are positive
```

The first modeling target should use:

```text
input prefix ends before onset_i - guard
guard = 1 or 2 frames
```

This prevents the model from seeing weak fluorescence-correlated morphology at the exact onset frame.

## Stage 2: Future Fluorescence Prediction

The real task is:

```text
M(B_i,1, B_i,2, ..., B_i,k) -> future fluorescence outcome
```

Start with outcome prediction, not future image synthesis. Future pixel-level fluorescence is hard because organoids move, grow, deform, and may not stay spatially aligned. A scalar or curve target is more biologically meaningful and easier to validate.

Recommended target order:

| Priority | Target | Definition |
| --- | --- | --- |
| 1 | future_positive | Does the organoid ever become positive after the prefix? |
| 2 | future_peak | Maximum corrected fluorescence after the prefix |
| 3 | future_auc | Area under corrected fluorescence curve |
| 4 | onset_time | First sustained positive time |
| 5 | future_curve | Fluorescence trajectory over future time |
| 6 | future_image | Fluorescence image at a future time point |

Training row:

```text
dataset
experiment/day
position
tracked_organoid_id
prefix_start_time
prefix_end_time
future_window
brightfield crop sequence
morphology feature sequence
future fluorescence labels
```

The model should support variable prefix length:

```text
M(B_1) -> future expression
M(B_1, B_2) -> future expression
M(B_1, ..., B_k) -> future expression
```

During training, randomly sample prefix length. During evaluation, report performance by prefix length:

```text
D1 only
D1-D2
D1-D3
all frames before onset - guard
```

Choose the earliest useful prediction point:

```text
shortest prefix whose validation metric is within 95% of the best longer-prefix model
```

## Tracking Requirement

Future prediction must be track-based. Individual crops without identity over time are not enough.

First-pass tracking can be simple:

```text
for each dataset / day / position:
  sort frames by time_index
  link organoid instances between t and t+1 by centroid distance and mask/bbox IoU
  use Hungarian matching or greedy nearest-neighbor
  break track if match confidence is low
  keep tracks with enough early frames and enough future frames
```

Track quality filters:

```text
minimum track length
maximum allowed centroid jump
minimum IoU or bbox overlap
no edge-padded crops
reasonable area change per frame
exclude tracks with obvious debris morphology
```

This is where Yichao's debris warning matters. Strong fluorescence is not automatically a cell. For labels, only count fluorescence that lies inside a morphology-valid organoid/cell mask or inside a validated peripheral-cell region. Fluorescent blobs without organoid/cell morphology should be marked debris/background and excluded from positive labels.

## Model Design

Recommended first future model:

```text
brightfield image encoder + morphology feature encoder + temporal GRU/Transformer + multi-task heads
```

Per-frame image features:

```text
e_i,t = CNN(B_i,t)
```

Per-frame morphology features:

```text
m_i,t = [
  area,
  equivalent_diameter,
  circularity,
  aspect_ratio,
  solidity,
  lumen/hole fraction,
  edge sharpness,
  texture statistics,
  growth rate,
  delta area,
  delta circularity
]
```

Temporal input:

```text
x_i,t = concat(e_i,t, m_i,t, delta_time_t, day_embedding, dataset_embedding)
```

Outputs:

```text
p_future_positive
future_peak_log
future_auc_log
onset_time_bin
optional future_curve
```

Loss:

```text
L = BCE(future_positive)
  + lambda_peak * Huber(log_future_peak)
  + lambda_auc * Huber(log_future_auc)
  + lambda_onset * ordinal_cross_entropy(onset_bin)
```

Only after this model works should we train:

```text
future fluorescence generator:
  input = early brightfield sequence + requested future time
  output = future fluorescence crop
```

For the future generator, use a conditional U-Net/diffusion/pix2pix-style model only after scalar outcomes are predictable. Otherwise image generation will mainly hallucinate plausible fluorescence.

## Validation Design

Avoid frame leakage. Do not randomly split individual crops.

Preferred splits:

```text
hold out full positions
hold out full days or experiments
hold out full datasets when testing generalization
```

At minimum:

```text
train: some positions from each dataset
validation: separate positions
test: held-out positions and ideally held-out dataset/day
```

Report:

```text
same-time B2F feasibility:
  AUROC/AUPRC for current positive state
  fluorescence intensity correlation
  calibration
  prediction examples

future prediction:
  AUROC/AUPRC for future_positive by prefix length
  Spearman/Pearson for future_peak and future_auc
  onset MAE or onset-bin accuracy
  performance vs days-before-onset
  ablation: image only vs morphology only vs combined
```

Critical controls:

```text
shuffled future labels
same-position leakage check
dataset-only baseline
time-only baseline
area-only baseline
fluorescence debris exclusion sensitivity
```

The model is only useful if it beats:

```text
dataset/day/position prior
current organoid size/growth-rate baseline
time-only baseline
```

## Interpretability

The scientific question is not just prediction. We also want to know what early features matter.

Recommended analyses:

```text
feature importance for explicit morphology features
ablation of area/growth/texture/lumen features
Grad-CAM or integrated gradients on brightfield crops
temporal attention or frame-removal sensitivity
compare high-risk and low-risk predicted tracks
cluster tracks in learned embedding space
```

Useful biological readouts:

```text
Does early area predict future fluorescence?
Does growth rate predict future fluorescence?
Does lumen formation predict future fluorescence?
Does peripheral texture predict future fluorescence?
Does edge sharpness or organoid compactness predict future differentiation?
```

## Practical Next Steps

1. Build a fluorescence-timeseries table from `projected_instance_pairs.sqlite`.

Suggested output:

```text
/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_future_expression/
  fluorescence_timeseries.sqlite
  histograms/
  onset/
```

2. Generate fluorescence histograms and onset plots.

Minimum plots:

```text
fluorescence_histogram_by_dataset_day_time.png
positive_fraction_over_time.png
onset_time_histogram.png
dataset_day_time_positive_heatmap.png
```

3. Train Stage 1 same-time feasibility models.

Recommended first datasets:

```text
Input: projected_instances where is_edge_padded='0'
Target: same-time fluorescence positivity and intensity
Exclude or isolate: Data-Yichao-9 if training goblet/N39 differentiation
```

4. If Stage 1 works, build organoid tracks.

Suggested output:

```text
/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_future_expression/tracks.sqlite
```

5. Train future outcome models.

Start with:

```text
M(B_early_sequence) -> future_positive, future_peak, future_auc, onset_bin
```

Do not start with:

```text
M(B_early_sequence) -> future fluorescence image
```

until scalar outcome prediction is validated.

## Decision Rule

Proceed from Stage 1 to Stage 2 only if:

```text
same-time current/late B -> fluorescence-positive classification works on held-out positions
and fluorescence onset is measurable with a stable threshold
and enough tracks have pre-onset brightfield frames plus future fluorescence labels
```

If these conditions are not met, the best next experimental request is:

```text
positive-control imaging set with many clean positive cells/organoids
negative-control imaging set with clean non-expressing organoids
explicit debris/background examples
consistent time-lapse positions from before expression through expression onset
```

## Bottom Line

The elegant and robust plan is not one pix2pix model. It is a ladder:

```text
1. quantify fluorescence over time and find onset
2. prove same-time brightfield contains expression information
3. build organoid tracks
4. predict future expression scalars from early brightfield
5. only then attempt future fluorescence image generation
```

This ladder prevents us from overfitting to attractive synthetic fluorescence images and keeps the project focused on the real biological goal: whether early brightfield morphology contains enough information to forecast future differentiation.
