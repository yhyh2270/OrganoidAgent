# Yichao Dataset Structure for Brightfield-to-Fluorescence Pix2pix

Related future-prediction strategy:

- `references/Yichao/yichao_wechat_virtual_staining_strategy.md`
- `references/Yichao/yichao_virtual_staining_strategy_tex/main.tex`
- `references/Yichao/yichao_virtual_staining_strategy_tex/main.pdf`
- `references/Yichao/future_fluorescence_forecasting_strategy.md`
- `references/Yichao/future_fluorescence_forecasting_tex/main.tex`
- `references/Yichao/future_differentiation_path_research_plan.md`
- `references/Yichao/b2f_metric_and_objective_improvement_research.md`

This note documents the current Yichao dataset layout in:

- v1 root: `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1/Data-Yichao-1`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1/Data-Yichao-2`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1/Data-Yichao-3`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1/Data-Yichao-4`
- v2 root: `/home/lachlan/ProjectsLFS/OrganoidAgent/DATA-Yichao-v2`

The supervised mapping used in this repo is:

- input: `c1` brightfield
- target: `c0` fluorescence

Older helper scripts may still contain the legacy assumption and should be checked before reuse:

- `BioAgentUtils/prepare_yichao_pairs_to_npy.py`
- `BioAgentUtils/train_pix2pix_yichao.py`


## Current Split

The split is now done **by raw LIF file first**, not by day.

Current meaning of the folders:

- `Data-Yichao-3` = data from `N39_TriRep_DF.lif`
- `Data-Yichao-4` = data from `N39_TriRep_DF_2.lif`

Current on-disk state:

- `Data-Yichao-v1/Data-Yichao-3/N39_TriRep_DF.lif`
- `Data-Yichao-v1/Data-Yichao-3/N39_TriRep_DF_jpeg_all`
- `Data-Yichao-v1/Data-Yichao-3/N39_TriRep_DF_jpeg_all_by_position`
- `Data-Yichao-v1/Data-Yichao-4/N39_TriRep_DF_2.lif`
- `Data-Yichao-v1/Data-Yichao-4/N39_TriRep_DF_2_jpeg_all`
- `Data-Yichao-v1/Data-Yichao-4/N39_TriRep_DF_2_jpeg_all_by_position`

Important:

- `N39_TriRep_DF.lif` was successfully exported
- `N39_TriRep_DF_2.lif` was also successfully exported
- the `*_by_position` folders are regrouped views of the same JPEG planes, organized by monitored position


## Core Unit of Usable Data

For pix2pix, the usable supervised unit is:

- one paired 2D image plane at fixed `series/position`, `z`, and `t`
- brightfield `c1` paired with fluorescence `c0`

For one LIF series:

- usable paired samples = `z_count * time_count`
- exported JPEG files = `z_count * time_count * 2`

because each `(t, z)` plane is exported for both channels.


## Important Overlap Warning

`Data-Yichao-v1/Data-Yichao-1/P11N&N39_Rep_DF.lif` is not an independent evaluation-only dataset.

Its 5 static MUC2 samples are byte-identical to the first 5 static samples inside:

- `Data-Yichao-v1/Data-Yichao-2/P11N&N39_Rep_DF.lif`

So:

- train on `Data-Yichao-2` and test on `Data-Yichao-1` is a leaky split
- the current repo defaults are fine for smoke tests, but not for a valid final benchmark


## File-Level Summary

### Data-Yichao-1

LIF file:

- `Data-Yichao-v1/Data-Yichao-1/P11N&N39_Rep_DF.lif`

Contents:

- 5 series
- all are static single-plane samples
- names:
  - `N39_TriRep_MUC2_mNeon_20X_1`
  - `N39_TriRep_MUC2_mNeon_20X_2`
  - `N39_TriRep_MUC2_mNeon_20X_3`
  - `N39_TriReP_MUC2_mNeon_20X_4`
  - `N39_TriRep_MUC2_mNeon_20X_5`

Acquisition structure:

- XY size: `1024 x 1024`
- channels: `2`
- z-depth per sample: `1`
- timepoints per sample: `1`
- usable paired samples per series: `1`
- total usable paired samples: `5`
- approximate pixel size: `0.303 um/pixel`

Interpretation:

- 5 distinct static fields of view
- no z-stack
- no time-lapse
- all 5 are duplicated in `Data-Yichao-2`


### Data-Yichao-2

LIF file:

- `Data-Yichao-v1/Data-Yichao-2/P11N&N39_Rep_DF.lif`

Contents:

- 11 series total
- 5 static MUC2 series, duplicated from `Data-Yichao-1`
- 6 dynamic Day-2 positions

Series breakdown:

| Series group | Count | XY | Z | T | Usable pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static MUC2 | 5 | 1024x1024 | 1 | 1 | 5 |
| `N39_TriRep_DF_D2/Position001` | 1 | 1024x1024 | 11 | 16 | 176 |
| `N39_TriRep_DF_D2/Position002` | 1 | 1024x1024 | 9 | 16 | 144 |
| `N39_TriRep_DF_D2/Position003` | 1 | 1024x1024 | 11 | 16 | 176 |
| `N39_TriRep_DF_D2/Position004` | 1 | 1024x1024 | 9 | 16 | 144 |
| `N39_TriRep_DF_D2/Position005` | 1 | 1024x1024 | 9 | 16 | 144 |
| `N39_TriRep_DF_D2/Position006` | 1 | 1024x1024 | 11 | 16 | 176 |

Acquisition structure for the dynamic Day-2 part:

- channels: `2`
- approximate pixel size: `0.568 um/pixel`
- z step magnitude: about `2.469 um`
- time step: about `3622 s` or `60.4 min`

Totals:

- total usable paired samples including duplicated static images: `965`
- unique dynamic paired samples beyond `Data-Yichao-1`: `960`

Interpretation:

- this is a mixed file
- it contains both static MUC2 images and dynamic Day-2 stacks
- the unique part for training is mainly the 6 Day-2 positions


### Data-Yichao-3

Current role:

- first LIF-based split folder
- contains data from `N39_TriRep_DF.lif` only

Current files:

- `Data-Yichao-v1/Data-Yichao-3/N39_TriRep_DF.lif`
- `Data-Yichao-v1/Data-Yichao-3/N39_TriRep_DF_jpeg_all`
- `Data-Yichao-v1/Data-Yichao-3/N39_TriRep_DF_jpeg_all_by_position`

Current extracted totals:

- usable paired samples: `10019`
- JPEG files in `N39_TriRep_DF_jpeg_all`: `20038`
- position folders in `N39_TriRep_DF_jpeg_all_by_position`: `13`

Series breakdown:

| Series | XY | Z | T | Usable pairs |
| --- | ---: | ---: | ---: | ---: |
| `Experiment_1 Day_2/Position001` | 512x512 | 26 | 41 | 1066 |
| `Experiment_1 Day_2/Position002` | 512x512 | 10 | 41 | 410 |
| `Experiment_1 Day_2/Position003` | 512x512 | 20 | 41 | 820 |
| `Experiment_1 Day_3/Position001` | 512x512 | 16 | 49 | 784 |
| `Experiment_1 Day_3/Position002` | 512x512 | 24 | 49 | 1176 |
| `Experiment_1 Day_3/Position003` | 512x512 | 8 | 49 | 392 |
| `Experiment_1 Day_3/Position004` | 512x512 | 11 | 49 | 539 |
| `Experiment_1 Day_3/Position005` | 512x512 | 32 | 49 | 1568 |
| `Experiment_1 Day_4/Position001` | 512x512 | 18 | 32 | 576 |
| `Experiment_1 Day_4/Position002` | 512x512 | 25 | 32 | 800 |
| `Experiment_1 Day_4/Position003` | 512x512 | 18 | 32 | 576 |
| `Experiment_1 Day_4/Position004` | 512x512 | 23 | 32 | 736 |
| `Experiment_1 Day_4/Position005` | 512x512 | 18 | 32 | 576 |

Day-level totals:

| Day | Positions | Usable pairs |
| --- | ---: | ---: |
| Day 2 | 3 | 2296 |
| Day 3 | 5 | 4459 |
| Day 4 | 5 | 3264 |

Acquisition structure:

- channels: `2`
- approximate pixel size: `1.137 um/pixel`
- Day 2 z step: about `2.000 um`
- Day 3 z step: about `1.608 um`
- Day 4 z step: about `2.000 um`
- time step: about `1800-1805 s` or about `30 min`

Interpretation:

- this is the main dynamic monitoring dataset
- it contains time-lapse z-stacks across Day 2, Day 3, and Day 4


### Data-Yichao-4

Current role:

- second LIF-based split folder
- contains data from `N39_TriRep_DF_2.lif` only

Current files:

- `Data-Yichao-v1/Data-Yichao-4/N39_TriRep_DF_2.lif`
- `Data-Yichao-v1/Data-Yichao-4/N39_TriRep_DF_2_jpeg_all`
- `Data-Yichao-v1/Data-Yichao-4/N39_TriRep_DF_2_jpeg_all_by_position`

Current extracted totals:

- usable paired samples: `7940`
- JPEG files in `N39_TriRep_DF_2_jpeg_all`: `15880`
- position folders in `N39_TriRep_DF_2_jpeg_all_by_position`: `9`

Series breakdown:

| Series | XY | Z | T | Usable pairs |
| --- | ---: | ---: | ---: | ---: |
| `Experiment_1 Day_2/Position001` | 512x512 | 16 | 49 | 784 |
| `Experiment_1 Day_2/Position002` | 512x512 | 22 | 49 | 1078 |
| `Experiment_1 Day_2/Position003` | 512x512 | 22 | 49 | 1078 |
| `Experiment_1 Day_2/Position004` | 512x512 | 20 | 49 | 980 |
| `Experiment_1 Day_2/Position005` | 512x512 | 27 | 49 | 1323 |
| `Experiment_1 Day_3/Position001` | 512x512 | 38 | 31 | 1178 |
| `Experiment_1 Day_3/Position003` | 512x512 | 23 | 31 | 713 |
| `Experiment_1 Day_3/Position004` | 512x512 | 17 | 31 | 527 |
| `Experiment_1 Day_3/Position005` | 512x512 | 9 | 31 | 279 |

Day-level totals:

| Day | Positions | Usable pairs |
| --- | ---: | ---: |
| Day 2 | 5 | 5243 |
| Day 3 | 4 | 2697 |

Interpretation:

- this folder is correctly split by LIF file
- it contains a second dynamic monitoring dataset
- it includes Day 2 and Day 3 time-lapse z-stacks


## What Each Folder Means

### `N39_TriRep_DF_jpeg_all`

This is the flat export directory.

Each file is one 2D plane from one position at one timepoint and one z-depth.

Example:

```text
00_Experiment_1_Day_2_Position001_t040_z025_c1.jpg
```

Meaning:

- `00`: series index inside the LIF export
- `Experiment_1_Day_2_Position001`: one monitored position / field of view
- `t040`: timepoint 40
- `z025`: z-plane 25
- `c1`: channel 1, brightfield
- matching `c0`: channel 0, fluorescence


### `N39_TriRep_DF_jpeg_all_by_position`

This is the same export regrouped by LIF series name.

Here, “object” is better read as:

- one imaging position
- one fixed field of view
- one monitored sample stack

So one folder under `*_by_position` corresponds to one monitored position.


### `Data-Yichao-v1/Data-Yichao-2/P11N&N39_Rep_DF_jpeg_all_by_object`

This folder mixes two different data types:

- dynamic Day-2 monitoring positions: `N39_TriRep_DF_D2_Position001..006`
- static single-image MUC2 samples: `N39_TriRep_MUC2_mNeon_20X_1..5`

So not every folder there is a time-lapse sequence.


### `Data-Yichao-v1/Data-Yichao-1/P11N&N39_Rep_DF_jpeg`

This is a legacy export for the 5 static MUC2 samples.

It predates the fuller `t000_z000` naming style and is effectively:

- one c0/c1 pair per sample
- no time dimension
- no z-stack

The cleaner equivalent export is:

- `Data-Yichao-v1/Data-Yichao-1/P11N&N39_Rep_DF_jpeg_all`


## Unique Usable Data Across the Underlying Source Files

If duplicated content is removed:

- `Data-Yichao-1`: `5` paired samples, but all duplicated in `Data-Yichao-2`
- `Data-Yichao-2`: `960` unique dynamic paired samples
- `Data-Yichao-v1/Data-Yichao-3/N39_TriRep_DF.lif`: `10019` unique dynamic paired samples
- `Data-Yichao-v1/Data-Yichao-4/N39_TriRep_DF_2.lif`: `7940` dynamic paired samples

Total unique paired planes across the current non-empty usable source LIF files:

- `18924`

Unique series/positions across the usable source files:

- 5 static MUC2 series
- 6 dynamic Yichao-2 Day-2 positions
- 13 dynamic positions from `N39_TriRep_DF.lif`
- 9 dynamic positions from `N39_TriRep_DF_2.lif`
- total unique series/positions: `33`


## What "Replication" Means Here

The filenames contain strings like `TriRep`, but the LIF metadata examined here does not cleanly encode biological replicate labels.

Safe interpretation:

- one LIF series is one field of view / position / sample stack
- positions within the same day should be treated as distinct samples
- z-planes are repeated observations across depth
- timepoints are repeated observations across time
- do not assume that `TriRep` provides a machine-readable replicate grouping for evaluation


## Implications for Pix2pix Training

Recommended supervised sample definition:

- one paired plane at fixed `position`, `t`, and `z`

Recommended split rule:

- split by position, not by random planes

Why:

- adjacent z-slices are strongly correlated
- adjacent timepoints are strongly correlated
- random plane splitting would leak almost-identical data between train and validation/test


### Practical baseline

The cleanest current baseline is:

- train first on `Data-Yichao-3`
- treat `Data-Yichao-4` as unavailable until `N39_TriRep_DF_2.lif` is restored


### Should Yichao-2 be mixed with Data-Yichao-3?

Not immediately.

Yichao-2 dynamic data and `N39_TriRep_DF.lif` are at different spatial sampling:

- Yichao-2 dynamic: `1024 x 1024`, about `0.568 um/pixel`
- `N39_TriRep_DF.lif`: `512 x 512`, about `1.137 um/pixel`

That means:

- the physical field of view and effective scale differ
- mixing them without normalization adds a domain shift

Best practice:

- first build a clean baseline on `Data-Yichao-3`
- then add Yichao-2 dynamic positions only after deciding how to normalize physical scale


## Bottom Line

If the goal is to learn fluorescence from brightfield with pix2pix:

- `Data-Yichao-1` is static only and not an independent test set
- `Data-Yichao-2` contains useful dynamic Day-2 data plus duplicated static content
- `Data-Yichao-3` is now the first LIF-based split folder and contains the full export of `N39_TriRep_DF.lif`
- `Data-Yichao-4` is now the second LIF-based split folder and contains the exported outputs of `N39_TriRep_DF_2.lif`
