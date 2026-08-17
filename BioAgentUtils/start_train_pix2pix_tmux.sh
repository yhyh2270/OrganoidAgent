#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${1:-pix2pix-train}"
if [[ $# -gt 0 ]]; then
  shift
fi

EXTRA_ARGS=("$@")

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
fi

tmux new-session -d -s "$SESSION_NAME" -n train

tmux send-keys -t "$SESSION_NAME":0.0 "cd \"$ROOT_DIR\"" C-m
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  tmux send-keys -t "$SESSION_NAME":0.0 "source \"$HOME/miniconda3/etc/profile.d/conda.sh\"" C-m
fi
tmux send-keys -t "$SESSION_NAME":0.0 "conda activate organoid" C-m

CMD=("$ROOT_DIR/BioAgentUtils/run_train_pix2pix_yichao.sh")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf -v CMD_STR '%q ' "${CMD[@]}"
tmux send-keys -t "$SESSION_NAME":0.0 "$CMD_STR" C-m

echo "[tmux-start] started session: $SESSION_NAME"
echo "[tmux-start] running command: $CMD_STR"
echo "[tmux-start] attach: tmux attach -t $SESSION_NAME"
echo "[tmux-start] stop from inside tmux with Ctrl+C"
