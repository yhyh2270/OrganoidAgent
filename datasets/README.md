# Bundled demonstration datasets

The repository tracks two small research-use demonstration datasets:

| Directory | Contents | Intended workflow |
| --- | --- | --- |
| `02_Fluorescence_demo/` | Two microscopy images | YOLO + SAM segmentation and viability prediction |
| `03_Y-27632_experiment_10x/` | Six 10× TIFF images covering 10, 20, and 100 µM at D05 and D07 | Y-27632 morphology analysis |

The Y-27632 directory includes `manifest.csv` with file sizes and SHA-256 checksums.
These images are provided for research workflow demonstration and are not clinical data.

Other files placed under `datasets/` remain ignored by Git by default. Confirm that you
have the necessary rights and that files contain no identifying or restricted information
before changing `.gitignore` to publish additional data.
