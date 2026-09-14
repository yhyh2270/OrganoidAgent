# Bundled demonstration datasets

The repository tracks a curated public subset of four research-use demonstration datasets. Each dataset contributes 20 representative TIFF images; the full source collections are larger and are not included in this repository.

| Directory | Contents | Intended workflow |
| --- | --- | --- |
| `01_Density_experiment_10x/` | 20 bright-field 10× TIFF images covering low, middle, and high-density conditions | DEO multiscale morphology analysis |
| `02_Sodium_alginate_experiment_10x/` | 20 bright-field 10× TIFF images covering control and two sodium-alginate conditions | Sodium-alginate morphology analysis |
| `03_Y-27632_experiment_10x/` | 20 bright-field 10× TIFF images covering five Y-27632 concentrations and matched days | Y-27632 morphology analysis |
| `05_Fluorescence_demo/` | 20 microscopy images, including the original 470/675 examples | YOLO + SAM segmentation and viability prediction |

The three morphology directories include source manifests with file sizes and SHA-256 checksums. The included files are a curated subset selected to cover experimental conditions and representative time points. These images are provided for research workflow demonstration and are not clinical data.

Other files placed under `datasets/` remain ignored by Git by default. Confirm that you have the necessary rights and that files contain no identifying or restricted information before publishing additional data.
