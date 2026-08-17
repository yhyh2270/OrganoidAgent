[English](README.md) · [العربية](i18n/README.ar.md) · [Español](i18n/README.es.md) · [Français](i18n/README.fr.md) · [日本語](i18n/README.ja.md) · [한국어](i18n/README.ko.md) · [Tiếng Việt](i18n/README.vi.md) · [中文 (简体)](i18n/README.zh-Hans.md) · [中文（繁體）](i18n/README.zh-Hant.md) · [Deutsch](i18n/README.de.md) · [Русский](i18n/README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Data](https://img.shields.io/badge/Data-Local%20First-4c956c)
![Format](https://img.shields.io/badge/Preview-Multi--format-f4a261)
![Status](https://img.shields.io/badge/Status-Active-success)

OrganoidAgent is a lightweight Tornado backend and Progressive Web App (PWA) for browsing and previewing organoid datasets locally with minimal setup. It provides file-type-aware preview rendering for tables, microscopy images (including TIFF), archives, gzip text, and AnnData `.h5ad` analysis objects.

## 🎯 At a glance

| Goal | What this repo gives you |
|---|---|
| Local-first dataset exploration | Dataset discovery, metadata, and file browsing from a local `datasets/` workspace |
| Rich previews | Table, image (including TIFF), archive, `.gz`, and `.h5ad` preview pathways |
| Offline-friendly frontend | Installable PWA shell with a service worker and manifest |
| Practical operations | Archive extraction + category-filtered indexing paths |

## Overview 🔭

The core app is designed for interactive dataset exploration with minimal setup:

- Backend API and preview engine in `app.py`
- PWA frontend in `web/`
- Download helpers in `scripts/`
- Local dataset workspace in `datasets/` (git-ignored)

This repository also contains adjacent research and utility workspaces (`BioAgent`, `BioAgentUtils`, `references`, `results`, `vendor`, `papers` submodule). The primary runtime described in this README is the top-level `OrganoidAgent` app.

## Features ✨

- Local dataset indexing with size and file-count summaries
- Recursive dataset file listing with inferred file kind
- Preview support for CSV/TSV/XLS/XLSX tables
- Preview support for TIFF/JPG/PNG images
- Preview support for `.h5ad` summaries with embedding/PCA scatter preview generation
- Preview support for ZIP/TAR/TGZ archive listing + first-image preview attempt
- Preview support for `.gz` text first-lines preview
- Archive extraction endpoint for large packaged datasets
- Dataset-level metadata cards rendered from Markdown
- PWA frontend with service worker and manifest
- Basic path sanitization (`safe_dataset_path`) to confine file access under `datasets/`

### At a glance

| Area | What it provides |
|---|---|
| Dataset discovery | Directory-level dataset listing with file counts and size summaries |
| File exploration | Recursive listing and kind inference (`image`, `table`, `analysis`, `archive`, etc.) |
| Rich previews | Tables, TIFF/images, gzip text snippets, archive contents, AnnData summaries |
| Analysis visuals | `.h5ad` scatter previews from `obsm` embeddings or PCA fallback |
| Packaging support | Archive listing + extraction endpoint for large compressed bundles |
| Web UX | Installable PWA with offline-friendly service worker assets |

## Project Structure 🗂️

```text
OrganoidAgent/
├─ app.py
├─ web/
│  ├─ index.html
│  ├─ app.js
│  ├─ styles.css
│  ├─ sw.js
│  ├─ manifest.json
│  └─ icons/
├─ scripts/
│  ├─ download_organoid_datasets.py
│  ├─ download_drug_screening_datasets.py
│  └─ overlay_segmentations.py
├─ datasets/                      # downloaded data and preview cache (git-ignored)
├─ metadata/
│  └─ zenodo_10643410.md
├─ papers/                        # submodule: prompt-is-all-you-need
├─ i18n/                          # currently present for multilingual README files
├─ BioAgent/                      # related but separate app
├─ BioAgentUtils/                 # related training/data utilities
├─ references/
├─ results/
└─ vendor/                        # external submodules (copilot-sdk, paper-agent, codex)
```

## Prerequisites ✅

- Python `3.10+`
- Recommended environment manager: `conda` or `venv`

Required/optional Python packages inferred from source:

| Package | Role |
|---|---|
| `tornado` | Required for server startup |
| `pandas` | Optional: table preview support |
| `anndata`, `numpy` | Optional: `.h5ad` preview and analysis plotting |
| `Pillow` | Optional: image rendering and generated previews |
| `tifffile` | Optional: TIFF preview support |
| `requests` | Optional: dataset download scripts |
| `kaggle` | Optional: Kaggle downloads in drug-screening script |

Dependencies are split into `requirements.txt` for the web app and previews, and
`requirements-analysis.txt` for model inference, Cellpose, training, and microscopy tools.

## Installation ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Option A: reproducible conda environment (includes analysis/GPU workflows)
conda env create -f environment.yml
conda activate organoid

# Option B: pip, core web app and previews
python -m pip install -r requirements.txt

# Add fluorescence inference, Cellpose, training, and microscopy tools
python -m pip install -r requirements-analysis.txt

# Verify packages and the three bundled model checkpoints
python scripts/check_environment.py --profile analysis
```

## Usage 🚀

### Quick Start

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # optional if you already have the deps
python app.py --port 8080
```

Open `http://localhost:8080`.

### API Smoke Test

```bash
curl http://localhost:8080/api/datasets
```

### Download Data (Optional)

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

Downloaded data lives in `datasets/` (git-ignored).

## API Endpoints 🌐

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/datasets` | List datasets with summary stats |
| `GET` | `/api/datasets/{name}` | List files for one dataset |
| `GET` | `/api/datasets/{name}/metadata` | Return markdown metadata card |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | Category-oriented file listing |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | File-type-aware preview payload |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | Extract archive into sibling `_extracted` folder |
| `GET` | `/files/<path>` | Raw dataset file serving |
| `GET` | `/previews/<path>` | Generated preview asset serving |

Example preview call:

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## Configuration 🧩

Current runtime configuration is intentionally small:

- Server port: `--port` argument in `app.py` (default `8080`)
- Data directory: fixed to `datasets/` relative to repository root
- Preview cache: `datasets/.cache/previews`
- Metadata mapping: `DATASET_METADATA` dictionary in `app.py`
- GitHub API token for downloader (optional): `GITHUB_TOKEN` env var or `--github-token`

Assumption note: if you need configurable dataset roots or production server settings, these are not yet exposed in top-level configuration files.

## Examples 🧪

### Browse category-specific files

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### Extract an archive

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### Run selective download modes

```bash
# Organoid datasets: skip GEO, keep Zenodo
python scripts/download_organoid_datasets.py --skip-geo

# Drug-screening datasets: only Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## Development Notes 🛠️

- Backend serves frontend static assets from `web/`.
- Service worker and manifest are in `web/sw.js` and `web/manifest.json`.
- File-type routing and previews are implemented in `app.py`.
- Manual validation (current project guidance): PWA loads at `http://localhost:8080`
- Manual validation (current project guidance): `/api/datasets` returns JSON
- Manual validation (current project guidance): previews render for CSV/XLSX/images/archives

## Troubleshooting 🩺

- `ModuleNotFoundError` for preview libraries: run `python -m pip install -r requirements.txt`.
- Incomplete model runtime: run `python scripts/check_environment.py --profile analysis` and install `requirements-analysis.txt`.
- On a separate Cellpose environment, set `ORGANOID_CELLPOSE_PYTHON=/absolute/path/to/python` before starting the server.
- Empty dataset listing: confirm data exists under `datasets/` and directories are not dot-prefixed.
- `.h5ad` preview missing scatter image: check that `anndata`, `numpy`, and `Pillow` are installed.
- Large archive preview/extraction issues: use extraction endpoint and inspect extracted files directly.
- GitHub downloader rate limit errors: provide `GITHUB_TOKEN` via env var or CLI flag.
- Kaggle download not working: install `kaggle` and configure `~/.kaggle/kaggle.json` credentials.

## Roadmap 🧭

Potential next improvements (not yet fully implemented in this root app):

- Add automated tests for API handlers and preview functions
- Add configurable dataset root and cache settings
- Add explicit production run profile (non-debug, reverse-proxy guidance)
- Expand multilingual documentation under `i18n/`

## Contributing 🤝

Contributions are welcome. A practical workflow:

1. Fork and create a focused branch.
2. Keep changes scoped to one logical area.
3. Manually validate app startup and key endpoints.
4. Open a PR with summary, commands run, and screenshots for UI changes.

Local style conventions in this repository:

- Python: 4-space indentation, snake_case functions/files, CapWords classes
- Keep frontend logic in `web/app.js` for this app (avoid unnecessary framework rewrites)
- Keep comments concise and only where logic is non-obvious

## Project Layout (Canonical Summary) 📌

- `app.py`: Tornado server and API routes.
- `web/`: PWA assets.
- `scripts/`: dataset download helpers.
- `datasets/`: local data storage.
- `papers/`: submodule with reference materials.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## License 📄

No top-level project `LICENSE` file is currently present in this repository root.

Assumption note: until a root license is added, treat reuse/redistribution terms as unspecified for the top-level OrganoidAgent codebase.
