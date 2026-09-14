# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Analysis](https://img.shields.io/badge/Analysis-Segmentation%20%7C%20Viability-7b2cbf)
[![CI](https://github.com/yhyh2270/OrganoidAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/yhyh2270/OrganoidAgent/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

OrganoidAgent is a local-first platform for organoid microscopy and experimental-data analysis. It integrates dataset browsing, TIFF and table previews, organoid segmentation, morphological quantification, fluorescence-based viability prediction, and agent-assisted workflows in a Tornado + PWA application. The repository includes curated subsets from four research datasets that can be used to verify the web interface, preview functions, and the density, sodium-alginate, Y-27632, and fluorescence-analysis workflows after cloning.

> OrganoidAgent is intended for research use. Model outputs require experimental validation and must not be interpreted as clinical conclusions without appropriate evidence.

## Features

| Module | Functionality |
| --- | --- |
| Dataset browsing | Indexes `datasets/`, recursively browses files, and previews CSV, Excel, TIFF, standard image, archive, and AnnData files |
| Morphology analysis | Runs multiscale Cellpose workflows and computes area, size, edge, curvature, and centrality-related metrics |
| Viability prediction | Combines YOLO, SAM, and ConvNeXt-Tiny to generate organoid masks, viability scores, and image-based evidence |
| Batch reporting | Produces JSON, CSV, and Markdown reports, crops, masks, and visualization overlays |
| Agent Studio | Provides sessions, workflow parsing, Codex jobs, status monitoring, and result management |
| PWA interface | Uses no large frontend framework, supports browser installation, and caches static resources |

## Processing pipeline

```text
Microscopy image
   ├─ YOLO: organoid localization
   ├─ SAM: fine-grained instance-mask generation
   ├─ Morphological features and quality control
   └─ ConvNeXt-Tiny: viability-score prediction
                   └─ JSON / CSV / Markdown / masks / overlays
```

The default report categorizes predictions as high viability (`>= 0.8`), intermediate viability (`0.6–0.8`), or low viability (`< 0.6`). These thresholds are used only for result summarization and ranking.

## Built-in workflows

| Workflow | Input data | Main outputs |
| --- | --- | --- |
| Viability detection | `05_Fluorescence_demo` or other microscopy images | YOLO/SAM masks, viability scores, CSV, JSON, and Markdown reports |
| Y-27632 dataset analysis | `03_Y-27632_experiment_10x` | Cellpose/signal-recovery quality comparison, morphological metrics, overlays, and reports |
| DEO morphology | Density-experiment TIFF files following the expected naming convention | Multiscale segmentation, growth, fusion, roundness, edge, and related metrics |
| Sodium alginate analysis | Sodium-alginate experiment TIFF files | Condition-aware segmentation, morphological quantification, and summaries |

Workflows operate only on files explicitly selected by the user and do not automatically expand to other datasets. All model outputs are intended for research analysis only.

## Repository structure

```text
OrganoidAgent/
├── app.py                       # Tornado API and PWA server
├── agent_studio.py              # Agent sessions and Codex job management
├── web/                         # HTML, JavaScript, CSS, service worker, and icons
├── fluorescence_prediction/     # Segmentation, viability inference, features, and reports
│   ├── SAM/                     # Segment Anything implementation
│   └── weights/                 # Local model weights (ignored by Git)
├── analysis-tools/              # Multiscale segmentation and experiment-analysis scripts
├── api-tests/                   # API and segmentation reproduction scripts and notes
├── scripts/                     # Environment checks, downloaders, and CLI entry points
├── config/workflows.json        # Analysis-environment and workflow configuration
├── datasets/                    # Curated public subsets; other local data is ignored
├── analysis-outputs/            # Runtime outputs (ignored by Git)
├── requirements.txt             # Web and data-preview dependencies
├── requirements-analysis.txt    # Full analysis dependencies
└── environment.yml              # Conda environment definition
```

## System requirements

- Python 3.10 or newer
- Windows, Linux, or macOS
- CPU support for the web interface and data previews
- An NVIDIA GPU is recommended for segmentation and viability prediction; the example Conda environment uses CUDA 12.1
- Several GB of disk space for the full environment and model weights

## Quick start

To start the web interface with dataset browsing and preview functionality only:

```bash
git clone -b main https://github.com/yhyh2270/OrganoidAgent.git
cd OrganoidAgent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py --profile core
python app.py --port 8080
```

On Windows PowerShell, replace the activation command with `.venv\Scripts\Activate.ps1`. Then open <http://localhost:8080> in a browser.

## Installation

### 1. Clone the repository

```bash
git clone -b main https://github.com/yhyh2270/OrganoidAgent.git
cd OrganoidAgent
```

### 2A. Install the full analysis environment (recommended)

```bash
conda env create -f environment.yml
conda activate organoid
python scripts/check_environment.py --profile analysis
```

The provided `environment.yml` installs the CUDA 12.1 build of PyTorch. If no NVIDIA GPU is available, replace `pytorch-cuda=12.1` with a CPU-compatible configuration before creating the environment.

### 2B. Install only the web and data-preview features

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py --profile core
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py --profile core
```

To add the full analysis functionality to an existing virtual environment:

```bash
python -m pip install -r requirements-analysis.txt
```

For a specific CUDA version, install matching `torch` and `torchvision` packages by following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Model weights

The model files are large and are not stored in the regular Git history. They are distributed as assets in the [`models-v1.0.0`](https://github.com/yhyh2270/OrganoidAgent/releases/tag/models-v1.0.0) release. Before running full viability prediction, place the following files in:

```text
fluorescence_prediction/weights/
├── viability_best.pth
├── yolo_organoid_best.pt
└── sam_vit_b_01ec64.pth
```

### Automatic download and verification (recommended)

From the repository root, run the built-in downloader. It skips files that already pass verification and applies both size and SHA-256 checks to new downloads:

```bash
python scripts/download_model_weights.py
```

To redownload all files:

```bash
python scripts/download_model_weights.py --force
```

### Manual download and verification: Windows PowerShell

```powershell
$release = "https://github.com/yhyh2270/OrganoidAgent/releases/download/models-v1.0.0"
$weightDir = "fluorescence_prediction\weights"
New-Item -ItemType Directory -Force -Path $weightDir | Out-Null

$weights = @{
  "sam_vit_b_01ec64.pth" = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
  "viability_best.pth"   = "cb328bae14ac8d1b77556636e4925e8c7549db83903c15430999c8b8ced9759b"
  "yolo_organoid_best.pt" = "2236790c8e9f027310f63306686215ff6928120b0d59938304e68f965e21b0cf"
}

foreach ($file in $weights.Keys) {
  $target = Join-Path $weightDir $file
  Invoke-WebRequest "$release/$file" -OutFile $target
  $actual = (Get-FileHash -Algorithm SHA256 $target).Hash.ToLower()
  if ($actual -ne $weights[$file]) { throw "SHA-256 mismatch: $file" }
  Write-Host "Verified $file"
}
```

### Manual download and verification: Linux/macOS

```bash
set -euo pipefail
release="https://github.com/yhyh2270/OrganoidAgent/releases/download/models-v1.0.0"
weight_dir="fluorescence_prediction/weights"
mkdir -p "$weight_dir"

curl -fL "$release/sam_vit_b_01ec64.pth" -o "$weight_dir/sam_vit_b_01ec64.pth"
curl -fL "$release/viability_best.pth" -o "$weight_dir/viability_best.pth"
curl -fL "$release/yolo_organoid_best.pt" -o "$weight_dir/yolo_organoid_best.pt"

cd "$weight_dir"
printf '%s  %s\n' \
  'ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912' 'sam_vit_b_01ec64.pth' \
  'cb328bae14ac8d1b77556636e4925e8c7549db83903c15430999c8b8ced9759b' 'viability_best.pth' \
  '2236790c8e9f027310f63306686215ff6928120b0d59938304e68f965e21b0cf' 'yolo_organoid_best.pt' \
  | sha256sum --check --strict
```

On macOS, if `sha256sum` is unavailable, install GNU coreutils (`brew install coreutils`) and replace `sha256sum` with `gsha256sum` in the final command.

### File checksums

| File | Size | SHA-256 |
| --- | ---: | --- |
| `sam_vit_b_01ec64.pth` | 375,042,383 bytes | `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` |
| `viability_best.pth` | 335,956,115 bytes | `cb328bae14ac8d1b77556636e4925e8c7549db83903c15430999c8b8ced9759b` |
| `yolo_organoid_best.pt` | 6,258,019 bytes | `2236790c8e9f027310f63306686215ff6928120b0d59938304e68f965e21b0cf` |

Check dependencies, interpreters, and model weights with:

```bash
python scripts/check_environment.py --profile analysis
```

If fluorescence inference or Cellpose runs in a separate environment, set:

```powershell
$env:ORGANOID_FLUORESCENCE_PYTHON = "C:\path\to\python.exe"
$env:ORGANOID_CELLPOSE_PYTHON = "C:\path\to\python.exe"
```

On Linux/macOS, use the same variable names with absolute interpreter paths.

For example, if Cellpose is installed in the current environment:

```bash
export ORGANOID_CELLPOSE_PYTHON="$(command -v python3)"
python app.py --port 8080
```

If these variables are not set, the application uses the Python interpreter that launched it. Paths to deployment-specific interpreters may also be configured in `config/workflows.json`; the repository does not include machine-specific absolute paths.

## Run the application

```bash
python app.py --port 8080
```

Open <http://localhost:8080>. To browse and preview additional data, place TIFF, CSV, XLSX, or other supported files in `datasets/` and refresh the page.

Quick backend checks:

```bash
curl http://localhost:8080/api/datasets
curl http://localhost:8080/api/fluorescence/status
```

Common API endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/datasets` | List local datasets |
| `GET /api/preview?path=...` | Generate a file preview |
| `GET /api/fluorescence/status` | Check inference dependencies and weights |
| `POST /api/fluorescence/run` | Create a fluorescence-viability prediction task |
| `POST /api/morphology/run` | Create a morphology-analysis task |
| `GET /api/morphology/status?id=...` | Query morphology-task status |
| `POST /api/morphology/cancel` | Cancel a morphology task |
| `POST /api/agent/chat` | Use the Agent Studio session interface |

## Command-line viability prediction

Single image:

```bash
python scripts/run_fluorescence_prediction.py \
  --input datasets/example/image.tif \
  --output-dir analysis-outputs/fluorescence_prediction/single
```

Batch directory (Windows PowerShell):

```powershell
python scripts/run_fluorescence_prediction.py `
  --input datasets\example_batch `
  --output-dir analysis-outputs\fluorescence_prediction\batch `
  --order desc
```

Main outputs include `results.json`, `results.csv`, `report.md`, and per-sample crops, masks, overlays, and structured results.

## Included datasets

The repository contains curated subsets from four experimental datasets that can be used immediately after cloning. Each dataset contributes 20 representative TIFF images; the full source collections are larger and are not included here.

| Directory | Contents | Size |
| --- | --- | ---: |
| `datasets/01_Density_experiment_10x/` | Low, middle, and high-density bright-field examples | 20 TIFF files, approximately 340 MiB |
| `datasets/02_Sodium_alginate_experiment_10x/` | Control and two sodium-alginate conditions | 20 TIFF files, approximately 340 MiB |
| `datasets/03_Y-27632_experiment_10x/` | Five Y-27632 concentrations and matched days | 20 TIFF files, approximately 340 MiB |
| `datasets/05_Fluorescence_demo/` | Fluorescence-based viability-prediction examples | 20 microscopy images, approximately 325 MiB |

The morphology directories include source `manifest.csv` files with condition, date, file-size, and SHA-256 metadata for post-transfer integrity checks. These data are provided for research workflow demonstrations only and must not be used to draw clinical conclusions. Other complete experimental datasets are not automatically committed to Git; additional public data can be downloaded on demand with:

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

Some sources may require network access, Kaggle credentials, or acceptance of the data provider's terms of use.

## Development and verification

After making changes, run at least:

```bash
python -m compileall app.py agent_studio.py fluorescence_prediction scripts
python scripts/check_environment.py --profile core
python tests/smoke_test.py
```

Then verify that:

1. The PWA opens in a browser.
2. `/api/datasets` returns JSON.
3. CSV, XLSX, TIFF, standard-image, and archive previews work.
4. When the analysis environment is configured, the weight status is `ready` and a small sample produces inference outputs.

## Troubleshooting

- **The page opens but no data are shown:** Confirm that files are under the repository's `datasets/` directory and refresh the page.
- **Inference status is not `ready`:** Run the analysis environment check and verify all three weight files and interpreter environment variables.
- **CUDA/PyTorch installation fails:** Install a PyTorch build compatible with the installed driver, or use a CPU configuration.
- **TIFF/Excel preview fails:** Confirm that `imagecodecs`, `tifffile`, `openpyxl`, and `xlrd` are installed.
- **The port is already in use:** Start the application on another port, for example `python app.py --port 8081`.

## Data, privacy, and security

- Only the four explicitly listed curated dataset directories under `datasets/` are tracked; other experimental data are ignored by default.
- `analysis-outputs/`, model weights, preview caches, local environments, and Agent jobs are not uploaded to GitHub.
- Do not commit API keys, personally identifiable patient information, or restricted raw data.
- Agent Studio can launch local Codex jobs. Use it only in trusted environments and review its file-access scope.
- Follow the licenses and data-governance requirements of external datasets and model components.

## License

The source code is released under the [Apache License 2.0](LICENSE). Copyright and attribution information is provided in [NOTICE](NOTICE). The included microscopy images are intended for research demonstrations; verify the applicable data rights and licenses before publishing or redistributing other data.
