#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/lachlan/ProjectsLFS/OrganoidAgent"
SESSION="${1:-yichao_fluorescence_segmentation_v1}"
OUTPUT_ROOT="${2:-${REPO_ROOT}/analysis-outputs/yichao_fluorescence_segmentation/runs/global_gated_unet_v1}"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/train.log"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  tmux display-message -p -t "${SESSION}" '#S:#I.#P #{pane_current_command}'
  exit 0
fi

tmux new-session -d -s "${SESSION}" bash
tmux send-keys -t "${SESSION}" "cd '${REPO_ROOT}'" C-m
tmux send-keys -t "${SESSION}" "source \"\$HOME/miniconda3/etc/profile.d/conda.sh\" 2>/dev/null || source \"\$HOME/anaconda3/etc/profile.d/conda.sh\" 2>/dev/null || true" C-m
tmux send-keys -t "${SESSION}" "conda activate organoid" C-m
tmux send-keys -t "${SESSION}" "export MPLBACKEND=Agg" C-m
tmux send-keys -t "${SESSION}" "echo YICHAO_SEGMENTATION_START \$(date -Iseconds) | tee -a '${LOG_FILE}'" C-m
tmux send-keys -t "${SESSION}" "python -u -m differentiation_prediction.yichao_fluorescence_segmentation.train_segmentation --output-root '${OUTPUT_ROOT}' --image-size 384 --epochs 300 --batch-size 8 --base-channels 32 --dropout 0.05 --lr 1.5e-4 --balanced-sampler --eval-every 2 --panel-every 10 --save-every 10 --keep-periodic 6 --early-stop --early-stop-patience-evals 15 --early-stop-min-delta 0.002 --early-stop-min-epochs 30 --resume 2>&1 | tee -a '${LOG_FILE}'; status=\${PIPESTATUS[0]}; if [ \${status} -eq 0 ]; then echo YICHAO_SEGMENTATION_FINISHED \$(date -Iseconds) | tee -a '${LOG_FILE}'; else echo YICHAO_SEGMENTATION_FAILED \$(date -Iseconds) status=\${status} | tee -a '${LOG_FILE}'; fi" C-m

echo "started tmux session: ${SESSION}"
echo "output: ${OUTPUT_ROOT}"
echo "log: ${LOG_FILE}"
