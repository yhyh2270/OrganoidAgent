# Repository Guidelines

## Project Structure & Module Organization
- `app.py`: Tornado backend that serves the API and the PWA frontend.
- `web/`: PWA assets (`index.html`, `app.js`, `styles.css`, `sw.js`, `manifest.json`, `icons/`).
- `scripts/`: Dataset download helpers (Zenodo/Figshare/GitHub).
- `datasets/`: Downloaded data and caches (git-ignored).
- `papers/`: Submodule holding reference material (`papers/prompt-is-all-you-need`).

## Build, Test, and Development Commands
- Start the app (backend + PWA served by Tornado):
  ```bash
  cd /home/lachlan/ProjectsLFS/OrganoidAgent
  conda activate organoid  # if you use the organoid env
  python app.py --port 8080
  ```
  Open `http://localhost:8080`.
- List datasets:
  ```bash
  curl http://localhost:8080/api/datasets
  ```
- Download datasets (optional):
  ```bash
  python scripts/download_organoid_datasets.py
  python scripts/download_drug_screening_datasets.py
  ```

## Coding Style & Naming Conventions
- Python 3.10+, 4‑space indentation, snake_case for functions/files, CapWords for classes.
- Keep frontend JavaScript in `web/app.js`; avoid large framework rewrites.
- Use concise comments only where logic is non‑obvious.

## Testing Guidelines
- No automated tests yet. Validate manually:
  - PWA loads at `http://localhost:8080`.
  - `/api/datasets` returns JSON.
  - Previews render for CSV/XLSX/images/archives.

## Commit & Pull Request Guidelines
- Use short, imperative commit messages (e.g., “Add preview for TIFF”).
- Keep commits scoped to one logical change.
- After any agent edit, immediately run `git add`, `git commit`, and `git push` unless the user explicitly says not to.
- PRs should include a brief summary, the main commands run, and screenshots for UI changes.

## Security & Configuration Tips
- Do not commit secrets. Keep API keys out of `datasets/` and `web/`.
- Large datasets belong under `datasets/` only; keep the repo lightweight.
