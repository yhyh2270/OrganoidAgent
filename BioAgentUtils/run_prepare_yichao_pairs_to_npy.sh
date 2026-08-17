#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/BioAgentUtils/prepare_yichao_pairs_to_npy.py"

echo "[prepare_npy] root:   $ROOT_DIR"
echo "[prepare_npy] script: $SCRIPT_PATH"
echo "[prepare_npy] args:   $*"

if [[ "${CONDA_DEFAULT_ENV:-}" != "organoid" ]]; then
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate organoid
  fi
fi

PYTHONUNBUFFERED=1 python -u "$SCRIPT_PATH" "$@"
