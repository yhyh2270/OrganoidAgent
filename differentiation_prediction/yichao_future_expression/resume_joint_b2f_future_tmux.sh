#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${YICHAO_JOINT_TMUX_SESSION:-yichao_joint_b2f_future}"
OUTPUT_ROOT="${YICHAO_JOINT_OUTPUT_ROOT:-$ROOT/analysis-outputs/yichao_future_expression/stage3_joint_b2f_future}"
PYTHON="${YICHAO_JOINT_PYTHON:-/home/lachlan/miniconda3/envs/organoid/bin/python}"
SHELL_BIN="${SHELL:-/bin/bash}"

mkdir -p "$OUTPUT_ROOT/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$SESSION"
  printf 'attach: tmux attach -t %s\n' "$SESSION"
  printf 'logs: %s/logs\n' "$OUTPUT_ROOT"
  exit 0
fi

tmux new-session -d -s "$SESSION" -c "$ROOT" "$SHELL_BIN -l"
tmux send-keys -t "$SESSION" "cd '$ROOT'" C-m
tmux send-keys -t "$SESSION" "source /home/lachlan/miniconda3/etc/profile.d/conda.sh && conda activate organoid" C-m
tmux send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}" C-m
tmux send-keys -t "$SESSION" "export PYTHONPATH='$ROOT'\${PYTHONPATH:+:\$PYTHONPATH}" C-m
tmux send-keys -t "$SESSION" "export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" C-m
tmux send-keys -t "$SESSION" "LOG='$OUTPUT_ROOT/logs/joint_b2f_future_'\"\$(date +%Y%m%d_%H%M%S)\"'.log'; echo JOINT_B2F_FUTURE_START \$(date -Iseconds) | tee -a \"\$LOG\"; '$PYTHON' '$ROOT/differentiation_prediction/yichao_future_expression/train_joint_b2f_future.py' --output-root '$OUTPUT_ROOT' --epochs \${YICHAO_JOINT_EPOCHS:-45} --batch-size \${YICHAO_JOINT_BATCH_SIZE:-16} --num-workers \${YICHAO_JOINT_NUM_WORKERS:-4} --b2f-base-channels \${YICHAO_JOINT_B2F_BASE:-16} --future-decoder-base \${YICHAO_JOINT_DECODER_BASE:-16} --amp --resume --feature-ablation 2>&1 | tee -a \"\$LOG\"; status=\${PIPESTATUS[0]}; if [ \$status -eq 0 ]; then echo JOINT_B2F_FUTURE_FINISHED \$(date -Iseconds) | tee -a \"\$LOG\"; else echo JOINT_B2F_FUTURE_FAILED \$(date -Iseconds) status=\$status | tee -a \"\$LOG\"; fi; echo SESSION_IDLE \$(date -Iseconds) | tee -a \"\$LOG\"" C-m

printf 'started tmux session: %s\n' "$SESSION"
printf 'attach: tmux attach -t %s\n' "$SESSION"
printf 'logs: %s/logs\n' "$OUTPUT_ROOT"
