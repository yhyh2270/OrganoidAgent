#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="bioagent"
BACKEND_PORT="${BIOAGENT_PORT:-8181}"
FRONTEND_PORT="${BIOAGENT_WEB_PORT:-8091}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

printf 'window.BIOAGENT_API_BASE = "http://localhost:%s";\n' "$BACKEND_PORT" > "$ROOT_DIR/web/config.js"

tmux new-session -d -s "$SESSION" -c "$ROOT_DIR" "conda run -n organoid python app.py --port $BACKEND_PORT"
tmux split-window -v -t "$SESSION" -c "$ROOT_DIR/web" "python -m http.server $FRONTEND_PORT"
tmux select-pane -t "$SESSION":0.0

tmux attach -t "$SESSION"
