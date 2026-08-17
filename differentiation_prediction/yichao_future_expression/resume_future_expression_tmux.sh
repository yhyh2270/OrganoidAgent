#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${YICHAO_FUTURE_TMUX_SESSION:-yichao_future_expression}"
OUTPUT_ROOT="${YICHAO_FUTURE_OUTPUT_ROOT:-$ROOT/analysis-outputs/yichao_future_expression}"
RUNNER="$ROOT/differentiation_prediction/yichao_future_expression/run_full_pipeline.sh"
SHELL_BIN="${SHELL:-/bin/bash}"

mkdir -p "$OUTPUT_ROOT/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$SESSION"
  printf 'attach: tmux attach -t %s\n' "$SESSION"
  printf 'log dir: %s\n' "$OUTPUT_ROOT/logs"
  exit 0
fi

tmux new-session -d -s "$SESSION" -c "$ROOT" "$SHELL_BIN -l"
tmux send-keys -t "$SESSION" "cd '$ROOT'" C-m
tmux send-keys -t "$SESSION" "source /home/lachlan/miniconda3/etc/profile.d/conda.sh && conda activate organoid" C-m
tmux send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}" C-m
tmux send-keys -t "$SESSION" "export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=\${MKL_NUM_THREADS:-4}" C-m
tmux send-keys -t "$SESSION" "LOG='$OUTPUT_ROOT/logs/full_pipeline_'\"\$(date +%Y%m%d_%H%M%S)\"'.log'; echo FUTURE_EXPR_START \$(date -Iseconds) | tee -a \"\$LOG\"; bash '$RUNNER' 2>&1 | tee -a \"\$LOG\"; status=\${PIPESTATUS[0]}; if [ \$status -eq 0 ]; then echo FUTURE_EXPR_FINISHED \$(date -Iseconds) | tee -a \"\$LOG\"; else echo FUTURE_EXPR_FAILED \$(date -Iseconds) status=\$status | tee -a \"\$LOG\"; fi; echo SESSION_IDLE \$(date -Iseconds) | tee -a \"\$LOG\"" C-m

printf 'started tmux session: %s\n' "$SESSION"
printf 'attach: tmux attach -t %s\n' "$SESSION"
printf 'logs: %s/logs\n' "$OUTPUT_ROOT"
