# PBMC + Organoid Project Analysis (sciadv.adn6331)

## Source paper context
- Paper: “MAIT cell activation and recruitment in inflammation and tissue damage in acute appendicitis” (Science Advances, 2024, doi:10.1126/sciadv.adn6331).
- Key excerpt (from PDF): the study establishes a patient-derived appendix organoid (PDAO) and MAIT cell co-culture system, showing MAIT cell recruitment/infiltration into infected organoids and organoid destruction in a CCR1/CCR2/CCR4-dependent manner.

## Data sources in OrganoidData
- `/home/lachlan/ProjectsLFS/OrganoidData/sciadv.adn6331.pdf`
- `/home/lachlan/ProjectsLFS/OrganoidData/PBMC+Organoid`
- PPTX decks in `/home/lachlan/ProjectsLFS/OrganoidData/*.pptx` (10 files)
- Additional media: `2-150ms.mp4`, `3-150ms.mp4`

## PBMC+Organoid dataset structure (from filenames)
Total image files (tif/tiff/png/jpg): **4893**

Top-level folder counts:
- `App61+P42`: 464
- `App61+P62`: 458
- `App61+P72`: 412
- `App61+P60`: 384
- `App61+P70`: 374
- `App61+P59`: 350
- `App61+P28`: 341
- `App61+P50`: 316
- `App61 new P10+P30`: 246
- `App61+P36`: 206
- `App61 new new+P44`: 206
- `App61+P33`: 190
- `App61+P56`: 188
- `App61 new P9+P40`: 188
- `App61 P8+PBMC32`: 173
- `APP61+P31 2`: 158
- `App61 new new+P53`: 152
- `MAIT+Organoid`: 32
- `App58+P4`: 31
- `APP61+P31`: 24

Condition tags found in filenames (counts):
- `PBMC`: 4510
- `Organoid`: 2878
- `Matrigel`: 1334
- `IL12+18`: 1185
- `IL1b+23`: 1173
- `Unstim`: 1138
- `IL2+7`: 1113
- `Organoid+EC`: 1030
- `Matrigel+EC`: 534

Day/time tokens (counts):
- `D2`: 1301
- `D1`: 1158
- `D0`: 1100
- `D3`: 884
- `D8`: 14
- `D5`: 11
- `D4`: 5
- `D10`: 4

Notes from filenames:
- Many files include `RGB` suffixes and `2x` (likely magnification variants).
- Conditions include immune stimulation cytokines (`IL12+18`, `IL1b+23`, `IL2+7`) and `Unstim`.
- Experimental material types include `Organoid`, `Organoid+EC`, `Matrigel`, `Matrigel+EC`.

## PPTX text extraction
Extracted text is stored in:
- `references/pbmc_organoid_pptx_text_extract.md`
The slide text content is sparse (most slides appear to be images), but the extracted text provides any labels/notes embedded in the decks.

## PDF text extraction
Full text extraction is stored in:
- `references/sciadv_adn6331_text_extract.md`

## Suggested analysis focus
Given the paper’s focus and dataset naming:
- Immune cell recruitment/infiltration toward infected organoids
- Condition-driven comparisons (IL12+18 vs IL1b+23 vs IL2+7 vs Unstim)
- Timepoint progression (D0/D1/D2/D3)
- Quantify organoid structural changes and immune interaction phenotypes

Suggested endpoints:
- Organoid boundary changes (size, solidity)
- Immune cell track density near boundary
- Boundary crossing events (infiltration score)
- Temporal response curves by condition
