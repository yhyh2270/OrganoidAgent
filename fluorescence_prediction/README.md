# Fluorescence viability workflow

This module is the deterministic bridge used by the OrganoidAgent web UI and Codex jobs.

Single image:

```powershell
python scripts/run_fluorescence_prediction.py --input datasets/01_Density_experiment_10x/high_D05_10x__10X_high.tif --output-dir analysis-outputs/fluorescence_prediction/manual_single
```

Dataset or batch:

```powershell
python scripts/run_fluorescence_prediction.py --input datasets/04_Density_demo_10x --output-dir analysis-outputs/fluorescence_prediction/manual_batch --order desc
```

The wrapper uses `ORGANOID_FLUORESCENCE_PYTHON` when set, then the local Windows `torch38` environment, then the current Python interpreter. Outputs include `results.json`, `results.csv`, `report.md`, and per-image overlays, masks, crops, morphology evidence, and viability scores.
