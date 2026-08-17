#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${SESSION:-yichao_v2_pairs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/analysis-outputs/yichao_v2_instance_pairs}"
V2_ROOT="${V2_ROOT:-${ROOT}/DATA-Yichao-v2}"
CHUNK_NEW_IMAGES="${CHUNK_NEW_IMAGES:-80}"
GPU_MODE="${GPU_MODE:-auto}"

mkdir -p "${OUTPUT_ROOT}/logs"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux send-keys -t "${SESSION}" "cd '${ROOT}'" C-m
else
  tmux new-session -d -s "${SESSION}" -c "${ROOT}"
fi

tmux send-keys -t "${SESSION}" "export CHUNK_NEW_IMAGES='${CHUNK_NEW_IMAGES}' GPU_MODE='${GPU_MODE}'" C-m
tmux send-keys -t "${SESSION}" "bash '${ROOT}/analysis-tools/yichao_v2_instance_pairs/run_yichao_v2_instance_pair_pipeline.sh' '${V2_ROOT}' '${OUTPUT_ROOT}'; echo YICHAO_V2_PIPELINE_DONE \\$(date -Iseconds)" C-m
printf '%s\n' "${SESSION}"
