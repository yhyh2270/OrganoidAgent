# Yichao WeChat Strategy: Virtual Staining, Brightfield Cell-Type Inference, and Early Fluorescence Prediction

Date: 2026-06-27

This note documents the visible WeChat messages from the screen-recorded discussion and turns them into a concrete research direction for the Yichao brightfield-to-fluorescence project. The original video is treated as private source material; this file records only the technical content needed for the project.

Private local source copy:

```text
/home/lachlan/ProjectsLFS/Private/yichao_wechat_video_20260627/1243979019.mp4
```

Sampled keyframes used for inspection:

```text
/home/lachlan/ProjectsLFS/Private/yichao_wechat_video_20260627/frames_1fps/
```

## Directly Observed Conversation Points

### If direct prediction is hard, use virtual staining / cell-type inference

Yichao's first main point is that if direct future fluorescence prediction is not good enough, a useful alternative direction is to use brightfield morphology to infer cell identity or composition. The target becomes:

```text
brightfield image -> virtual stain / inferred cell type / inferred cell-type proportion
```

The visible message states that this could avoid physical staining while still estimating what cell types are present in the organoid:

```text
如果预测不好做的话，可以考虑通过明场来判断这个细胞是什么细胞。
类似虚拟染色：只拍明场，但通过明场知道内器官里面有哪些细胞类型，
以及不同细胞类型的比例。
```

Interpretation:

- This is not only image-to-image regression.
- It is better framed as label-free / in-silico labeling.
- The useful output may be a cell-type map, a cell-type probability map, an organoid-level composition vector, or a virtual fluorescence image.

### Similar published work likely exists

Yichao said he remembers related published work, likely in tissue, and suggested checking whether this direction has precedent:

```text
印象中应该有人做过类似研究，也发表过文章，可能是对于组织。
你看看做这个方向会不会相对容易一点，有没有可行性。
```

Interpretation:

- This matches the established literature on in-silico labeling and virtual staining.
- The closest precedents are transmitted-light-to-fluorescence prediction, virtual histological staining, and label-free organoid tracking.

### Three candidate directions were explicitly listed

The chat shows the following project directions:

```text
1. 虚拟染色
2. 时序预测，早期明场预测后期荧光
3. 利用普通明场或者自己后相机，直接在明场上进行分类
```

Interpretation:

1. `B -> virtual F / cell-type map`: same-day virtual staining or fluorescence prediction.
2. `B_early -> F_later`: early temporal prediction of future differentiation/expression.
3. `B -> class/composition`: direct brightfield classification, possibly without pixel-perfect fluorescence synthesis.

These are not competing tasks. They should be staged:

```text
same-day virtual staining -> same-day cell-type classification -> early-to-later forecasting
```

### Start with an easy, visually distinguishable cell type

Yichao says the currently marked fluorescent cell type is likely only one type and is relatively easy to distinguish visually. The visible OCR is not perfect, but the technical meaning is Goblet/goblet-like cells:

```text
目前只有一种细胞类型，就是杯状/Goblet-like 细胞。
绿色荧光有发荧光的其实只有一种类型。
杯状/Goblet-like 细胞可以从肉眼辨别出来，因为它和别的细胞形态不一样，
它是粘液分泌，所以形状也可以看出来。
```

Interpretation:

- The first scientifically defensible target should be Goblet/high-goblet-like cells, not all intestinal epithelial subtypes.
- The model should first learn the easiest visible morphology-to-marker relationship.
- Success criterion should be localization and composition accuracy, not only pixel-level intensity correlation.

### Rarer cell types should come later

Yichao says rarer intestinal cells, such as Paneth cells or enteroendocrine/neuroendocrine cells, are more meaningful but harder because they are rare:

```text
其他比较少见的细胞，比如 Paneth cell 或神经内分泌细胞，
在肠内器官里面数量非常少。
如果需要很大的数据集训练，周期会比较长。
```

Interpretation:

- Do not begin with rare classes.
- Rare classes need either targeted imaging, marker-specific enrichment, or public/external pretraining.
- The data strategy should be active: use the current data for Goblet-like proof of concept, then ask for targeted rare-cell data only if the model and metrics justify it.

### External/public data may help

Yichao shared the Gut Cell Atlas link and suggested choosing cell types with more public data:

```text
https://www.gutcellatlas.org/spacetime/epithelium/

肠子里面有几种细胞类型，不可能都做。
看看哪些细胞公开的数据比较多，先做哪个。
```

Interpretation:

- Public atlas data is useful for biological cell-type prioritization, not directly for paired brightfield-to-fluorescence supervision unless paired images are available.
- Public imaging datasets can help if they include label-free microscopy paired with fluorescence labels.
- Single-cell RNA atlases can define which cell types are relevant and which marker labels should be requested experimentally.

### Organoid-reproducible cell types mentioned

The visible message lists organoid-reproducible cell types:

```text
BEST4
Enterocyte / Colonocyte
EEC 神经内分泌细胞
Goblet
Microfold / M cells
Paneth cells
TA
Tuft
stem cells
```

Interpretation:

- These are candidate biological target classes.
- The first target should still be the currently imaged fluorescent class, likely Goblet/high-goblet-like marker expression.
- Later phases can add markers for one or two rare but meaningful classes after the Goblet proof of concept works.

### Need for more imaging is conditional

The visible final exchange:

```text
师弟那需要再继续拍吗，还是这样就行了
我这几天出去了，我先看下新拍的数据效果怎么样
好，需要你再跟我说
```

Interpretation:

- Do not ask for more data blindly.
- First inspect the newly received v2 data, confirm channels, and test whether same-day virtual staining works.
- Only then request targeted extra imaging, e.g. more positive Goblet examples, more clean negatives, or specific rare-cell marker sets.

## Research Framing

The project should now be described with three related but distinct prediction levels.

### Level 1: Same-day virtual staining

Input:

```text
B_d
```

Output:

```text
F_d or clean fluorescence signal target
```

Goal:

```text
Can brightfield morphology predict same-day fluorescence expression?
```

This is the v1-style B2F task. It validates channel pairing, segmentation, crop quality, and whether there is visible morphological signal.

### Level 2: Same-day cell-type map / composition

Input:

```text
B_d
```

Output:

```text
P(cell type | pixel/region/organoid), or organoid-level composition vector
```

Goal:

```text
Can brightfield morphology distinguish fluorescent/Goblet-like cells from non-fluorescent cells?
```

This is closer to virtual staining / in-silico labeling. It may be more biologically useful than pixel-perfect intensity regression if intensity is noisy, saturated, or partly confounded by debris.

### Level 3: Early-to-later prediction

Input:

```text
B_k, k < d
```

Output:

```text
F_d, cell-type map at d, or expression score at d
```

Goal:

```text
Can early morphology predict later differentiation before strong fluorescence appears?
```

This should be attempted only after Level 1 and Level 2 are stable.

## Practical Modeling Implication

The model should not be optimized only for green pixel intensity. The target should be decomposed:

```text
brightfield -> positive-expression mask
brightfield -> clean continuous fluorescence intensity
brightfield -> organoid-level expression/composition score
```

A useful loss is:

```text
L = λ_mask L_mask + λ_int L_intensity_on_positive + λ_score L_organoid_score
```

where:

- `L_mask` handles the strong positive/negative pixel imbalance.
- `L_intensity_on_positive` prevents the output from becoming only binary.
- `L_organoid_score` makes the model preserve total expression even if pixel-perfect alignment is impossible.

For early prediction:

```text
L_future(k,d) = w(d-k) L(B_k -> target_d)
```

Use a horizon-aware or uncertainty-aware weight so the model is encouraged to predict early but not punished as if very early prediction is always possible.

## No-Future-Leakage Rules

Temporal prediction is scientifically invalid if split incorrectly. These rules are mandatory:

1. Split by biological field/replicate/position before creating temporal pairs.
2. Keep all days and acquisitions from the same physical sample lineage in the same split.
3. For prediction `B_k -> F_d`, the model input must contain only information available at or before day/time `k`.
4. Do not choose thresholds or normalization from validation/test target fluorescence.
5. Use validation only to choose earliest useful prediction day.
6. Hold out a full data batch, such as v2 or a future Yichao dataset, as an external test when possible.

## Immediate Decision For Current Data

For `DATA-Yichao-v2`, do not train yet until channels are confirmed.

Current known state:

```text
1_N39Rep_Globet_DF_D3: extracted/grouped, named acquisitions, likely usable after channel confirmation
2_N39Rep_Globet_DF_D4: extracted/grouped, named acquisitions, likely usable after channel confirmation
3_N39Rep_Globet_DF_D2: copied/metadata only, generic Series###, 7 channels
4_N39Rep_Globet_DF_D3_1: copied/metadata only, generic Series###, 8 channels
5_N39Rep_Globet_DF_D3_2: copied/metadata only, generic Series###, 8 channels
```

Required next step:

```text
extract all v2 files -> channel contact sheets -> manual channel mapping -> segmentation database
```

Only then:

```text
same-day v2 B2F / virtual staining -> same-day cell-type classification -> early-to-later prediction
```

## Related Research Materials

### In-silico labeling from transmitted light

Christiansen et al. introduced in-silico labeling, predicting fluorescent labels from unlabeled transmitted-light microscopy images. This is directly aligned with the same-day B2F task and supports the claim that some biological labels can be inferred from label-free morphology.

Reference:

```text
Christiansen et al. In Silico Labeling: Predicting Fluorescent Labels in Unlabeled Images. Cell, 2018.
https://pubmed.ncbi.nlm.nih.gov/29656897/
```

### 3D fluorescence prediction from transmitted-light microscopy

Ounkomol et al. predicted 3D fluorescence images directly from transmitted-light images, showing that brightfield/transmitted-light data can contain enough information to infer labeled structures.

Reference:

```text
Ounkomol et al. Label-free prediction of three-dimensional fluorescence images from transmitted-light microscopy. Nature Methods, 2018.
https://www.nature.com/articles/s41592-018-0111-2
```

### Virtual histological staining

Rivenson et al. showed virtual histological staining from label-free autofluorescence images. The modality differs from Yichao brightfield, but the conceptual frame is the same: digitally generate staining-like readouts from minimally invasive imaging.

Reference:

```text
Rivenson et al. Virtual histological staining of unlabelled tissue-autofluorescence images via deep learning. Nature Biomedical Engineering, 2019.
https://www.nature.com/articles/s41551-019-0362-y
```

### Label-free organoid tracking

Recent organoid-specific work uses U-Net-style models to infer nuclei/membrane information from brightfield images of 3D intestinal organoids, supporting the idea that organoid brightfield images can carry morphology useful for label-free analysis.

Reference:

```text
Label-free cell imaging and tracking in 3D organoids.
https://www.cell.com/cell-reports-physical-science/pdfExtended/S2666-3864%2825%2900121-3
```

### Label-free live-cell recognition and tracking

A recent review frames label-free live-cell microscopy as a practical route for segmentation, tracking, and biological readouts without perturbing labels.

Reference:

```text
Chen et al. Label-free live cell recognition and tracking for biological discoveries and translational applications. npj Imaging, 2024.
https://www.nature.com/articles/s44303-024-00046-y
```

### Cell-type biological reference

The Gut Cell Atlas and related Nature work provide a biological map of intestinal cell types across space and time, including BEST4 cells and other intestinal epithelial lineages.

References:

```text
Gut Cell Atlas:
https://www.gutcellatlas.org/

Elmentaite et al. Cells of the human intestinal tract mapped across space and time. Nature, 2021.
https://www.nature.com/articles/s41586-021-03852-1
```

### Model architecture and imbalance handling

U-Net remains the practical baseline for segmentation-like microscopy prediction. Focal loss is relevant because fluorescence-positive pixels are sparse relative to negative/background pixels.

References:

```text
Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI, 2015.
https://arxiv.org/abs/1505.04597

Lin et al. Focal Loss for Dense Object Detection. ICCV, 2017.
https://arxiv.org/abs/1708.02002
```

## Recommended Next Experiment

The next experiment should be deliberately simple:

1. Confirm v2 channel mapping.
2. Segment brightfield organoids.
3. Build a v2 instance-pair database.
4. Train same-day `B_d -> clean F_d` and `B_d -> positive mask_d`.
5. Evaluate:
   - visual predictions,
   - positive-region Dice/F1,
   - organoid-level expression correlation,
   - false positives from debris/background.
6. If same-day works, train `B_D2 -> F_D3/D4` and `B_D3 -> F_D4`.
7. Report the earliest day where prediction is useful.

The first biological claim should be conservative:

```text
Brightfield morphology can infer the currently labeled Goblet/high-expression phenotype better than chance.
```

Only after that should the project claim:

```text
Early brightfield can forecast future differentiation/expression.
```
