# Joint B2F And Future Fluorescence Prediction Research Note

Date: 2026-05-11

## Saved Prompt

The current task prompt to preserve is:

> could you write the code to do the test and help me
>
> 1, show the feasibility at least b2f
>
> 2, show the possible explanation of the features
>
> 3, do the eary prediction of expression of f at eary stage bright
>
> 4, quantify and explain the feature
>
> with elegant and robust algorithm
>
> 1, prepare the dataset loader of the projected data and complete one and maybe zoom to the same size
>
> 2, train the model one by one
>
> 3, quantify and analyze the features
>
> 4, infer and test and validation
>
> could you start write the code and run in tmux session and write robust code and test and validate and debug to make sure it work
>
> and save all necessary intermediate artifacts and stage outputs predictions and analysis viaualizations
>
> don't just do mock test
>
> the training should be real in tmux with all the projected data either by loader or by pt file
>
> and the epochs also need to be realistic that can really work and train a valid model

Additional saved instruction from the follow-up:

> when you finish the training the future expression prediction , quantify , visualize it and possibly explain it
>
> btw, can you train the b2f and also the future expression jointly
>
> so we have a b2f(b) --> F and also B2F(FutureB(B0-k)) or FutureF(B2F(B0-k)) with indefinite priors. because we don't know how many days is enough and at which day it's okay to predict future. here we can use B0-k or Bk to predict Bn or Bp (k<p<n).
>
> you needn't make things complicated and think a most realistic pathway and solution . the bottom line is to predict Bn (most mature) at Bk (most late). you can decide either predice Bp (which is better (because not the final state)) at Bk or B0-Bk (no difference as we have the B0 to Bk. it's your choice use the all history or just a snapshot)
>
> do a deep research and save also our previous prompt . don't do mock test . do real test and write real code and check until it works .

Final instruction before implementation:

> first save and deep research
>
> save into markdown
>
> with all the idea and solution in math and logic
>
> with intuition and research and reasoning

## Current Real Baseline State

The real projected-instance pipeline has already created:

- Projected complete/non-edge instance database: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_future_expression/future_expression.sqlite`
- Instance manifest: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_future_expression/manifests/projected_instances_manifest.csv`
- Future prefix samples: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_future_expression/manifests/future_samples.csv`
- Complete projected instances: 9,380
- Linked time-series tracks: 363
- Early-prefix future samples: 6,088
- Future sample split: 4,277 train, 1,178 validation, 633 test

Existing real B2F result on the held-out test split:

- Test instances: 1,088
- Positive instances: 14
- Image MAE: 0.1077
- Image PSNR: 14.18 dB
- Positive AUROC: 0.728
- Average precision: 0.110
- Fluorescence peak Pearson: 0.685
- Evaluation output: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_future_expression/stage1_b2f_evaluation/`

Existing first future-expression baseline result on the held-out future-sample test split:

- Future samples: 633
- Future-positive AUROC: 0.228
- Average precision: 0.0156
- Future peak Pearson: 0.134
- Future AUC Pearson: 0.105
- Output: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_future_expression/stage2_future_expression/`

Interpretation: same-time B2F is feasible, but the first scalar-only early future model is not yet reliable. This is not surprising: the future-positive labels are rare, track matching is approximate, and the future model only used a compact embedding and scalar targets. The next realistic step is to train a joint image-and-scalar model so the future task receives a stronger dense supervision signal.

## Research Anchors

The design uses four research ideas:

1. Paired image-to-image translation is the right first feasibility test. Pix2pix formalized paired conditional image translation as learning `G: X -> Y` with image reconstruction and adversarial objectives; for this dataset the simpler U-Net reconstruction component is the safest starting point because the data are limited and fluorescence positives are rare. Source: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks", CVPR 2017, https://arxiv.org/abs/1611.07004.

2. Label-free fluorescence reconstruction from transmitted/brightfield microscopy is biologically plausible. The Allen Cell `pytorch_fnet` work explicitly targets label-free prediction of fluorescence from transmitted-light microscopy, and practical fluorescence reconstruction microscopy uses transmitted-light inputs with fluorescence ground truth to validate predicted fluorescent features. Sources: https://allencellmodeling.github.io/pytorch_fnet/ and Christiansen et al./FRM discussion at https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008443.

3. Future prediction from longitudinal images should be treated as spatiotemporal forecasting. ConvLSTM introduced the formulation of predicting future spatial signals from spatial-temporal input sequences; our model can be simpler than ConvLSTM by using a CNN encoder per frame plus a GRU over frame embeddings because each training item is a cropped organoid instance rather than a full moving field. Source: Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting", https://arxiv.org/abs/1506.04214.

4. A joint model is justified because same-time B2F is an auxiliary task that regularizes the representation needed for future prediction. Multi-task and auxiliary-task learning are commonly used to share representation across related objectives and improve data efficiency. Source: Vafaeikia et al., "A Brief Review of Deep Multi-task Learning and Auxiliary Task Learning", https://arxiv.org/abs/2007.01126.

5. The organoid outcome-prediction goal itself is realistic: recent retinal organoid work shows that longitudinal brightfield morphology can predict later tissue outcomes before visual emergence. The exact marker and biology are different, but the machine-learning structure is similar: organoid-level time series, early morphology, later outcome. Source: Afting et al., PLOS Biology 2026, https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003597.

## Intuition

There are two different questions:

1. Same-time B2F:
   Given a brightfield crop `B_t`, can a model predict the matching fluorescence crop `F_t`?

   This tests whether morphology and optical texture at a given stage contain enough information to infer fluorescence-correlated cell state.

2. Future expression:
   Given brightfield history `B_0, ..., B_k`, can a model predict a future fluorescence state `F_p`, where `p > k`?

   This tests whether morphology before or near the onset of expression contains predictive information about the later differentiation path.

The first question is easier and should be solved first because it proves the fluorescence target is at least learnable from brightfield morphology in the same time window. The second question is harder because the target is partly causal/biological and partly stochastic.

The practical path is not to ask the model for a single magic answer at day 1. Instead, create many prefix-to-future tasks:

- `B_0 -> F_p`
- `B_0, B_1 -> F_p`
- `B_0, B_1, B_2 -> F_p`
- ...
- `B_0, ..., B_k -> F_p`

Then evaluate performance as a function of `k` and `p-k`. This tells us the earliest useful prediction window rather than assuming it.

## Data Objects

For each projected organoid instance:

- Brightfield crop: `B_i in R^{1 x H x W}`
- Fluorescence crop: `F_i in R^{1 x H x W}`
- Mask: `M_i in {0,1}^{1 x H x W}`
- Metadata: dataset, experiment, replicate, sample, position, time index, day index, z-projection policy, size features
- Scalar fluorescence features:
  - `y_i^pos in {0,1}`
  - `y_i^peak = log(1 + max(0, corrected p90 fluorescence))`
  - `y_i^auc = log(1 + future corrected intensity sum)` for future samples

For each linked track:

`T = [(B_0, F_0, M_0), ..., (B_n, F_n, M_n)]`

The current track linking is approximate, so the model and evaluation should be robust to occasional mismatches. Dense image loss should use the instance crop and mask, not full-field position.

## Proposed Realistic Joint Training Task

Use one model with two supervised branches:

1. Same-time auxiliary B2F branch:

   `G_aux(B_k) -> F_k`

2. Future branch:

   `G_future(B_{max(0,k-L+1)}, ..., B_k, X_{max(0,k-L+1)}, ..., X_k) -> F_p`

Here:

- `L` is the maximum prefix length, default `L=5`.
- `X_t` is a small explicit feature vector for frame `t`: area, diameter, circularity, support ratio, edge strength, bbox size, crop size, day/time indices.
- `p` is the future target index.
- Use `p = n` first, the most mature available future frame in that track.
- Later add optional intermediate targets `p in {k+1, ..., n}` if final-state prediction is too hard or if we want learning curves by horizon.

This gives:

`B_0...B_k -> F_n`

with auxiliary:

`B_k -> F_k`

That is the simplest realistic answer to "use all history or just a snapshot": use history by default, but keep the final frame `B_k` strongly supervised through the B2F auxiliary head.

## Math

Let a training sample be:

`s = (P_k, B_k, F_k, M_k, F_p, M_p, y_p^pos, y_p^peak, y_p^auc)`

where:

- `P_k = {(B_j, X_j, v_j)}_{j=k-L+1}^{k}` is the padded prefix sequence.
- `v_j` is a valid-frame indicator.
- `p > k` is the chosen target frame, initially the last future frame in the track.

The model has:

- Frame encoder `E_img(B_j) = z_j`
- Feature encoder `E_feat(X_j) = u_j`
- Sequence encoder `H({z_j, u_j, v_j}) = h_k`
- Same-time decoder `D_aux(B_k) = \hat{F}_k`
- Future decoder `D_future(h_k) = \hat{F}_p`
- Future scalar head `C(h_k) = (\hat{a}_p, \hat{r}_p, \hat{q}_p)`

where:

- `\hat{a}_p` is the future-positive logit.
- `\hat{r}_p` predicts standardized future peak log.
- `\hat{q}_p` predicts standardized future AUC log.

The dense masked reconstruction loss:

`L_img(\hat{F}, F, M) = mean(|\hat{F}-F| * M) + lambda_bg mean(|\hat{F}-F| * (1-M))`

with `lambda_bg = 0.2`, because signal inside the organoid matters more than background but background should not become arbitrary.

The scalar loss:

`L_scalar = BCEWithLogits(\hat{a}_p, y_p^pos) + alpha SmoothL1(\hat{r}_p, r_p) + beta SmoothL1(\hat{q}_p, q_p)`

The joint objective:

`L = L_img(\hat{F}_p, F_p, M_p) + gamma L_img(\hat{F}_k, F_k, M_k) + eta L_scalar`

Default weights:

- `gamma = 0.5`
- `eta = 0.25`
- `alpha = 0.3`
- `beta = 0.2`

Reasoning:

- Future image loss forces the model to learn where fluorescence should appear later.
- Same-time B2F loss prevents the image encoder from drifting away from the known brightfield-to-fluorescence relation.
- Scalar loss gives a compact readout for downstream decisions and allows feature ablation/permutation analysis.

## Target Choices

Use two target policies, in this order:

1. `last_future`: target the last available future frame `F_n`.

   This matches the bottom line: predict the most mature outcome from a prior brightfield prefix.

2. `peak_future`: target the future frame with the highest corrected fluorescence peak.

   This may be more biologically meaningful if mature fluorescence is transient or if the last frame suffers photobleaching or tracking noise.

The first implementation should default to `last_future`, because it is simple and reproducible. The code should expose `--target-policy last_future|peak_future` so the second policy can be tested without rewriting the pipeline.

## Handling Indefinite Priors

The question "how many days are enough?" should be answered empirically:

For each future sample, record:

- `prefix_length`
- `prefix_end_time_index = k`
- `target_time_index = p`
- `horizon = p-k`
- `dataset`
- `position`
- `track_id`

Then evaluate metrics by groups:

- prefix length: 1, 2, 3, 4, 5
- prefix end time: early/middle/late bins
- horizon: short/medium/long
- dataset: Data-Yichao-2/3/4/5/6/7/8/10

The decision rule is:

- If performance is bad for short prefixes but improves for late `k`, then prediction requires later morphology.
- If performance is already useful with early `k`, then true early prediction is feasible.
- If only same-time B2F is good but future prediction stays weak, then brightfield contains fluorescence-correlated morphology but not enough future-lineage information at the current temporal resolution/tracking quality.

## Evaluation Metrics

Same-time B2F:

- Image MAE and MSE
- PSNR
- Pearson correlation between predicted and true fluorescence peak
- Positive AUROC and average precision
- Prediction panel: brightfield, predicted fluorescence, true fluorescence

Future prediction:

- Future image MAE/MSE/PSNR
- Masked future image MAE
- Future-positive AUROC and average precision
- Future peak Pearson and MAE
- Future AUC Pearson and MAE
- Prediction panel: last observed brightfield, auxiliary predicted `F_k`, true `F_k`, predicted future `F_p`, true future `F_p`
- Metrics grouped by `prefix_length`, `prefix_end_time_index`, `horizon`, and dataset

Feature explanation:

- Explicit feature baseline: MLP on morphology features only
- Permutation importance: drop in AUROC and peak correlation when one feature is permuted/zeroed
- Joint model ablation: zero one feature channel in the sequence and measure future metric drop
- Plot feature trends: feature quantile vs positive fraction and mean fluorescence peak

## Realistic Implementation Plan

1. Keep existing data preparation.

   Do not rebuild segmentation. Use the existing projected complete/non-edge instance manifest and future samples.

2. Add `JointFutureDataset`.

   It reads `future_samples.csv`, maps instance IDs to instance rows, loads resized brightfield/fluorescence/mask images, chooses a future target frame, pads prefix sequences to `L=5`, and returns all metadata needed for grouped evaluation.

3. Add `JointB2FFutureModel`.

   Minimal robust architecture:

   - CNN frame encoder for each prefix brightfield.
   - Feature MLP for morphology/time features.
   - GRU sequence encoder.
   - Future image decoder from hidden vector.
   - Same-time B2F U-Net branch on `B_k`.
   - Scalar future head for positive/peak/AUC.

   This is more useful than a pure scalar model because it learns a spatial fluorescence target.

4. Train real model.

   - Use all future samples.
   - Use GPU in `organoid`.
   - Use `tmux`.
   - Save `last_model.pt` every epoch and `best_model.pt` by validation AUROC, falling back to future image MAE if AUROC is unstable.
   - Default epochs: 40 to 60.
   - Use AMP.

5. Evaluate and visualize.

   - Write `test_metrics.json`.
   - Write `test_predictions.csv`.
   - Write grouped metrics CSV files.
   - Save training curves.
   - Save future prediction panels.
   - Save ROC/PR/scatter/histograms.
   - Save feature ablation CSV and plots.

## Expected Outcomes And Risks

Expected:

- B2F remains feasible because current results already show meaningful performance.
- Joint future image prediction should be easier to interpret than scalar-only future prediction.
- The auxiliary B2F branch should help the model learn fluorescence-related morphology before trying to extrapolate future expression.

Main risks:

- Future-positive labels are rare, so average precision will remain low unless the data contain enough positive future examples in the held-out split.
- Approximate track linking can introduce label noise.
- Some datasets are not identical biological tasks; Data-Yichao-9 is already excluded from future training by default because it is PDO28/Jurkat rather than the N39 differentiation task.
- Final-frame prediction can be too hard if the last frame is noisy or biologically stochastic; `peak_future` target policy should be tested if `last_future` underperforms.

## Immediate Next Code To Write

New code should go under:

`/home/lachlan/ProjectsLFS/OrganoidAgent/differentiation_prediction/yichao_future_expression/`

Files:

- `joint_dataset.py`
- `train_joint_b2f_future.py`
- `evaluate_joint_b2f_future.py` if not folded into the trainer
- `resume_joint_b2f_future_tmux.sh`

Output folder:

`/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_future_expression/stage3_joint_b2f_future/`

The job should be run in tmux:

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
bash differentiation_prediction/yichao_future_expression/resume_joint_b2f_future_tmux.sh
```

## References

- Isola P, Zhu JY, Zhou T, Efros AA. Image-to-Image Translation with Conditional Adversarial Networks. CVPR 2017. https://arxiv.org/abs/1611.07004
- Shi X, Chen Z, Wang H, Yeung DY, Wong WK, Woo WC. Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. https://arxiv.org/abs/1506.04214
- Allen Cell Modeling. Pytorch_fnet: label-free prediction of 3D fluorescence images from transmitted-light microscopy. https://allencellmodeling.github.io/pytorch_fnet/
- Christiansen EM et al. Practical fluorescence reconstruction microscopy for large samples and low-magnification imaging. PLOS Computational Biology. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008443
- Vafaeikia P, Namdar K, Khalvati F. A Brief Review of Deep Multi-task Learning and Auxiliary Task Learning. https://arxiv.org/abs/2007.01126
- Afting C et al. A deep learning-based computational pipeline predicts developmental outcome in retinal organoids. PLOS Biology 2026. https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003597
