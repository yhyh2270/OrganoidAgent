#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/lachlan/ProjectsLFS/OrganoidAgent"
SESSION="${1:-yichao_continuous_hybrid_b2f_v2_noearly}"
TARGET_ROOT="${2:-${REPO_ROOT}/analysis-outputs/yichao_fluorescence_segmentation_relaxed}"
OUTPUT_ROOT="${3:-${REPO_ROOT}/analysis-outputs/yichao_fluorescence_continuous/runs/soft_suppressed_hybrid_unet_v2_noearly_metrics}"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/train.log"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  tmux display-message -p -t "${SESSION}" '#S:#I.#P #{pane_current_command}'
  echo "log: ${LOG_FILE}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" bash
tmux send-keys -t "${SESSION}" "cd '${REPO_ROOT}'" C-m
tmux send-keys -t "${SESSION}" "source \"\$HOME/miniconda3/etc/profile.d/conda.sh\" 2>/dev/null || source \"\$HOME/anaconda3/etc/profile.d/conda.sh\" 2>/dev/null || true" C-m
tmux send-keys -t "${SESSION}" "conda activate organoid" C-m
tmux send-keys -t "${SESSION}" "export MPLBACKEND=Agg" C-m
tmux send-keys -t "${SESSION}" "echo YICHAO_CONTINUOUS_HYBRID_B2F_START \$(date -Iseconds) | tee -a '${LOG_FILE}'" C-m
tmux send-keys -t "${SESSION}" "python -u -m differentiation_prediction.yichao_fluorescence_segmentation.train_continuous_target --target-root '${TARGET_ROOT}' --output-root '${OUTPUT_ROOT}' --image-size 256 --epochs 500 --batch-size 12 --num-workers 0 --base-channels 32 --dropout 0.05 --lr 1.5e-4 --balanced-sampler --target-scale 2.5 --soft-mask-dilate 9 --soft-mask-sigma 3.5 --soft-mask-floor 0.35 --metric-threshold 0.20 --metric-support-threshold 0.08 --metric-foreground-weight 8.0 --lambda-focal 1.0 --lambda-soft-dice 0.5 --lambda-l1 1.0 --lambda-mse 0.25 --lambda-total 0.20 --lambda-bg 0.08 --eval-every 2 --panel-every 10 --save-every 10 --keep-periodic 8 --resume 2>&1 | tee -a '${LOG_FILE}'; status=\${PIPESTATUS[0]}; if [ \${status} -eq 0 ]; then echo YICHAO_CONTINUOUS_HYBRID_B2F_FINISHED \$(date -Iseconds) | tee -a '${LOG_FILE}'; else echo YICHAO_CONTINUOUS_HYBRID_B2F_FAILED \$(date -Iseconds) status=\${status} | tee -a '${LOG_FILE}'; fi" C-m

echo "started tmux session: ${SESSION}"
echo "target: ${TARGET_ROOT}"
echo "output: ${OUTPUT_ROOT}"
echo "log: ${LOG_FILE}"
