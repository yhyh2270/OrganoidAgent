# Organoid Time-Course Dataset Analysis (类器官)

## Data source
- Folder: `/home/lachlan/ProjectsLFS/OrganoidData/类器官`
- Focus: organoid culture time course (named days in filenames)

## File inventory (from filenames)
Total image files (tif/tiff/png/jpg): **2824**

Top-level folders (image counts):
- `App81 25.8.28`: 1348
- `App73 25.8.28`: 468
- `App65 25.8.28`: 321
- `App80 25.9.26`: 269
- `App61 25.8.28`: 234
- `App74 25.9.26`: 184

Passage labels in paths:
- `P7`: 590
- `P5`: 588
- `P6`: 490
- `P8`: 428
- `P9`: 229
- `P4`: 196
- `P10`: 169
- `P11`: 61
- `P12`: 41
- `P3`: 32

Day tokens in filenames (counts):
- `D0`: 427
- `D4`: 384
- `D5`: 292
- `D1`: 290
- `D6`: 274
- `D8`: 256
- `D7`: 208
- `D3`: 142
- `D2`: 134
- `D10`: 98
- `D9`: 76
- `D11`: 56
- `Day9`: 18
- `Day0`: 13
- `Day1`: 13
- `Day10`: 8
- `D13`: 8
- `D14`: 8
- `D12`: 4

## Observations
- The dataset spans multiple passages and collection dates, not just D1/D7/D10.
- There are repeated day tokens and both `D#` and `Day#` naming styles.
- Many images include `_RGB` suffixes and `2X` variants in filenames.

## Use-case: growth prediction with limited days
Potential tasks:
- Predict later-day morphology (e.g., D7/D10/D11) using early days (e.g., D0–D3).
- Estimate growth curves and classify fast vs slow growers.
- Learn cross-passage generalization by holding out passages (e.g., train on P5/P6, test on P7/P8).

Suggested outputs:
- Organoid size/area distributions per day.
- Growth rate trajectories (median size vs day).
- Predictive models that reduce required culture time.
