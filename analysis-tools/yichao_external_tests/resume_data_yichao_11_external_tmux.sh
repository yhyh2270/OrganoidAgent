#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lachlan/ProjectsLFS/OrganoidAgent"
SESSION="${1:-yichao_11_external_pipeline}"
SOURCE_LIF="${2:-/home/lachlan/Downloads/N39_TriRep_DF_8.lif}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  tmux display-message -p -t "${SESSION}" '#S:#I.#P #{pane_current_command}'
  exit 0
fi

tmux new-session -d -s "${SESSION}" bash
tmux send-keys -t "${SESSION}" "cd '${ROOT}'" C-m
tmux send-keys -t "${SESSION}" "source \"\$HOME/miniconda3/etc/profile.d/conda.sh\" 2>/dev/null || source \"\$HOME/anaconda3/etc/profile.d/conda.sh\" 2>/dev/null || true" C-m
tmux send-keys -t "${SESSION}" "conda activate organoid" C-m
tmux send-keys -t "${SESSION}" "export MPLBACKEND=Agg" C-m
tmux send-keys -t "${SESSION}" "bash '${ROOT}/analysis-tools/yichao_external_tests/run_yichao_external_lif_pipeline.sh' '${SOURCE_LIF}'; status=\$?; echo YICHAO_11_EXTERNAL_PIPELINE_EXIT status=\${status} \$(date -Iseconds)" C-m

echo "started tmux session: ${SESSION}"
echo "source lif: ${SOURCE_LIF}"
echo "output root: ${ROOT}/analysis-outputs/yichao_external_tests/Data-Yichao-11"
