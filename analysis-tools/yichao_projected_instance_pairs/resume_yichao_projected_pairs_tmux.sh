#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${YICHAO_PROJECTED_TMUX_SESSION:-yichao_projected_pairs}"
OUTPUT_ROOT="${YICHAO_PROJECTED_OUTPUT_ROOT:-$ROOT/analysis-outputs/yichao_projected_instance_pairs}"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" bash
tmux send-keys -t "$SESSION" "cd \"$ROOT\"" Enter
tmux send-keys -t "$SESSION" "ulimit -c 0" Enter
tmux send-keys -t "$SESSION" "export PYTHONNOUSERSITE=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 YICHAO_PROJECTED_CHUNK_NEW_IMAGES=\${YICHAO_PROJECTED_CHUNK_NEW_IMAGES:-50} YICHAO_PROJECTED_RETRY_SLEEP_SECONDS=\${YICHAO_PROJECTED_RETRY_SLEEP_SECONDS:-3} YICHAO_PROJECTED_MAX_ATTEMPTS_WITHOUT_PROGRESS=\${YICHAO_PROJECTED_MAX_ATTEMPTS_WITHOUT_PROGRESS:-20}" Enter
tmux send-keys -t "$SESSION" "mkdir -p \"$OUTPUT_ROOT/logs\"" Enter
tmux send-keys -t "$SESSION" "echo YICHAO_PROJECTED_RESUME \$(date -Iseconds) | tee -a \"$OUTPUT_ROOT/logs/projected_full_run.log\"" Enter
tmux send-keys -t "$SESSION" "bash \"$ROOT/analysis-tools/yichao_projected_instance_pairs/run_yichao_projected_instance_pair_pipeline.sh\" --output-root \"$OUTPUT_ROOT\" 2>&1 | tee -a \"$OUTPUT_ROOT/logs/projected_full_run.log\"; status=\${PIPESTATUS[0]}; if [ \$status -eq 0 ]; then echo YICHAO_PROJECTED_FINISHED \$(date -Iseconds) | tee -a \"$OUTPUT_ROOT/logs/projected_full_run.log\"; else echo YICHAO_PROJECTED_FAILED \$(date -Iseconds) status=\$status | tee -a \"$OUTPUT_ROOT/logs/projected_full_run.log\"; fi; echo SESSION_IDLE \$(date -Iseconds)" Enter

echo "$SESSION"
echo "$OUTPUT_ROOT/logs/projected_full_run.log"

