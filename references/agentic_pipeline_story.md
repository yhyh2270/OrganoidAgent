# Agentic Pipeline Story for Organoid Immune Co-culture Analysis

Here is a complete, realistic story for an off-the-shelf, agentic analysis pipeline that turns brightfield organoid co-culture movies into quantitative immune-cell migration phenotypes. It emphasizes QC, reproducibility, and clear biological endpoints without claiming new foundation models.

## 1) One-sentence story
Build a reproducible “agentic” pipeline that turns raw brightfield organoid co-culture movies into validated, quantitative immune-cell migration phenotypes (QC → segmentation → tracking → event detection → interpretation), enabling comparisons across conditions that are difficult to do manually.

## 2) What to fix in the current story
### A. A biological question
Pick one concrete question:
- Recruitment efficiency: Which conditions increase immune cell recruitment?
- Infiltration phenotype: Do cells stay peripheral vs infiltrate?
- Motility mode shift: Do perturbations change speed/persistence/turning?
- Heterogeneity: Are phenotypes different across organoids in one condition?

### B. A quantitative endpoint
Use measurable metrics:
- Cell flux into a boundary ring (arrival rate, dwell time)
- Radial velocity toward organoid center
- Infiltration score: fraction of tracks crossing boundary and persisting inside
- Chemotaxis index: net displacement toward organoid / total path length
- Track persistence and speed distributions

### C. Validation strategy
- Human annotation subset for segmentation and tracking
- Optional fluorescence subset for immune-cell ground truth
- Reproducibility across batches, days, or microscopes

## 3) The realistic “agent”
The agent is a workflow conductor + self-checker, not a magical scientist.

### Module 0: Data Intake + Metadata
- Reads folders, frame rate, pixel size, condition labels, timepoints
- Outputs a standardized manifest (CSV/JSON)

### Module 1: QC
- Focus score (Tenengrad / Laplacian variance)
- Illumination statistics (background mean/variance)
- Motion blur estimate
- Optional lightweight artifact classifier

Outputs:
- QC score per frame and well
- Reasons: out-of-focus, drift, occlusion, lighting jump
- Curated subset for downstream processing

### Module 2: Organoid Segmentation
Off-the-shelf options:
- Cellpose
- SAM/SAM2 with prompts
- Classical threshold + morphology fallback

Outputs:
- Organoid masks
- Boundary contours
- Shape descriptors and confidence

### Module 3: Immune Cell Detection (brightfield)
Two-track approach:
1) Detection-first: background subtraction + bandpass + blob detection
2) Optical-flow-first: detect coherent moving objects

Outputs:
- Cell centroids or coarse masks
- Detection confidence

### Module 4: Tracking
- Kalman + Hungarian linking
- TrackMate-style logic or DeepSORT/ByteTrack for boxes

Outputs:
- Track IDs, ID-switch stats, track quality flags

### Module 5: Migration Phenotyping
Given boundaries + tracks:
- Speed, persistence, turning distributions
- Radial distance vs time, boundary crossings
- Condition-level aggregates + uncertainty (bootstrap)

Outputs:
- Per-well phenotype table
- Per-condition comparisons

### Module 6: Report Agent (LLM)
Inputs:
- QC summary, overlays, phenotype tables

Outputs:
- Structured report: QC decisions, main phenotypes, outliers, artifacts
- Suggested human-in-the-loop checks

LLM is used for summarization, not for detection.

## 4) Novelty framing
- Quality-aware automation (explicit QC and failure modes)
- Unified phenotyping (recruitment + infiltration + motility)
- Standardized outputs and reproducible manifest
- Benchmark subset with evaluation protocol

## 5) Validation checklist
### A. Segmentation
- 50–100 annotated frames
- Dice/IoU and boundary error (Hausdorff)

### B. Detection + Tracking
- 20 short clips annotated
- Precision/recall and IDF1/MOTA
- Optional fluorescence as gold standard

### C. Biological validation
- Known chemoattractant increases recruitment
- Motility inhibitor reduces speed/persistence
- Correlate phenotypes with endpoint assays

## 6) Complete story (copy-ready)
Goal: Build a robust, reusable pipeline that converts brightfield organoid–immune co-culture movies into validated migration phenotypes across conditions.  
Why now: Manual curation and tracking are bottlenecks; artifacts bias results.  
What we build: QC → organoid segmentation → immune cell detection → tracking → phenotyping, with standardized outputs and a report.  
Off-the-shelf core: Cellpose/SAM for organoid segmentation; classical + optional Cellpose for cells; TrackMate/DeepSORT tracking; standard migration metrics; LLM only for structured reporting.  
Deliverables: Dataset manifest, per-well/condition phenotypes, overlays, and benchmark subset with evaluation scripts.  
Validation: Dice/IoU for organoids; IDF1/MOTA for tracking; biological validation with known perturbations or fluorescence.  
Outcome: Reproducible recruitment and infiltration phenotypes and condition-dependent motility modes.

## 7) Fast build path
1. QC module + reporting  
2. Organoid boundary segmentation  
3. Tracking with blob detector  
4. Define phenotypes (flux, infiltration, chemotaxis index)  
5. Annotate benchmark subset  
6. Improve detection only if tracking errors dominate

## Source context
- Paper: “MAIT cell activation and recruitment in inflammation and ...” (Science Advances, doi:10.1126/sciadv.adn6331)
  - Dataset description: brightfield immune-cell migration around/into organoids.
