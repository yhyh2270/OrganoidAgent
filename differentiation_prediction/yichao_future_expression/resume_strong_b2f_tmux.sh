#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/lachlan/ProjectsLFS/OrganoidAgent"
SESSION="${1:-yichao_b2f_strong}"
OUTPUT_ROOT="${2:-${REPO_ROOT}/analysis-outputs/yichao_future_expression/stage1_b2f_pix2pix_384_v1_noamp_long}"
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
tmux send-keys -t "${SESSION}" "echo STRONG_B2F_START \$(date -Iseconds) | tee -a '${LOG_FILE}'" C-m
tmux send-keys -t "${SESSION}" "python -u -m differentiation_prediction.yichao_future_expression.train_b2f_strong --output-root '${OUTPUT_ROOT}' --image-size 384 --path-mode original_crop --epochs 1000 --batch-size 8 --grad-accum-steps 2 --architecture pix2pix_unet --base-channels 64 --dropout 0.5 --lr 1.5e-4 --min-lr 1e-6 --balanced-sampler --eval-every 5 --panel-every 20 --save-every 20 --keep-periodic 8 --early-stop --early-stop-metric score --early-stop-patience-evals 10 --early-stop-min-delta 0.001 --early-stop-min-epochs 40 --resume 2>&1 | tee -a '${LOG_FILE}'; status=\${PIPESTATUS[0]}; if [ \${status} -eq 0 ]; then echo STRONG_B2F_FINISHED \$(date -Iseconds) | tee -a '${LOG_FILE}'; else echo STRONG_B2F_FAILED \$(date -Iseconds) status=\${status} | tee -a '${LOG_FILE}'; fi" C-m

echo "started tmux session: ${SESSION}"
echo "output: ${OUTPUT_ROOT}"
echo "log: ${LOG_FILE}"
