#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREP_SCRIPT="$ROOT_DIR/BioAgentUtils/run_prepare_yichao_pairs_to_npy.sh"
TRAIN_SCRIPT="$ROOT_DIR/BioAgentUtils/train_pix2pix_from_npy.py"
DATA_DIR="$ROOT_DIR/results/yichao_paired_npy"
RESULTS_ROOT="$ROOT_DIR/results"
GPU_INDEX="${GPU_INDEX:-0}"
KILL_STALE="${KILL_STALE:-0}"

echo "[run_train] project root: $ROOT_DIR"
echo "[run_train] data dir:     $DATA_DIR"
echo "[run_train] results root: $RESULTS_ROOT"
echo "[run_train] gpu index:    $GPU_INDEX"
echo "[run_train] args:         $*"
echo "[run_train] kill stale:   $KILL_STALE"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[run_train] ERROR: nvidia-smi not found. GPU is required."
  exit 1
fi
nvidia-smi -L

if [[ "$KILL_STALE" == "1" ]]; then
  echo "[run_train] killing stale pix2pix python processes (if any)..."
  timeout 5s pgrep -fa "train_pix2pix_yichao.py|train_pix2pix_from_npy.py|python -X importtime" || true
  timeout 5s pkill -9 -f "train_pix2pix_yichao.py" || true
  timeout 5s pkill -9 -f "train_pix2pix_from_npy.py" || true
  timeout 5s pkill -9 -f "python -X importtime" || true
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "organoid" ]]; then
  echo "[run_train] using active env: organoid"
else
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate organoid
  else
    echo "[run_train] ERROR: cannot activate conda env 'organoid'."
    exit 1
  fi
fi

export PYTHONNOUSERSITE=1
echo "[run_train] PYTHONNOUSERSITE=$PYTHONNOUSERSITE"
echo "[run_train] python: $(command -v python)"

echo "[run_train] preparing .npy pairs..."
PYTHONUNBUFFERED=1 "$PREP_SCRIPT"

echo "[run_train] starting GPU training..."
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

echo "[run_train] preflight: checking torch+cuda import..."
if ! timeout 45s python -u - <<'PY'
print("[preflight] importing torch...", flush=True)
import torch
print(f"[preflight] torch={torch.__version__}", flush=True)
print(f"[preflight] torch_path={torch.__file__}", flush=True)
print(f"[preflight] cuda_available={torch.cuda.is_available()}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print(f"[preflight] cuda_device_count={torch.cuda.device_count()}", flush=True)
print(f"[preflight] current_device_name={torch.cuda.get_device_name(0)}", flush=True)
PY
then
  echo "[run_train] ERROR: torch/cuda preflight failed."
  echo "[run_train] Hint: if it hangs at 'importing torch', clean stuck GPU python processes first."
  exit 1
fi

PYTHONUNBUFFERED=1 python -u "$TRAIN_SCRIPT" \
  --data-dir "$DATA_DIR" \
  --results-root "$RESULTS_ROOT" \
  --gpu-index 0 \
  "$@"
