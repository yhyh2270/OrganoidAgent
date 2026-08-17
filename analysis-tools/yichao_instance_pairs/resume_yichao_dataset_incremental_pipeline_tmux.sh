#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET_NAME=""
LIF_PATH=""
FLAT_OUTPUT=""
GROUP_OUTPUT=""
PAIR_OUTPUT_ROOT="$ROOT/analysis-outputs/yichao_instance_pairs"
RESIZED_OUTPUT_ROOT="$ROOT/analysis-outputs/yichao_instance_pairs_resized_256"
PREVIEW_COUNT=12
PREVIEW_SEED=20260406
SESSION_NAME=""

usage() {
  cat <<'EOF'
Usage:
  bash analysis-tools/yichao_instance_pairs/resume_yichao_dataset_incremental_pipeline_tmux.sh \
    --dataset-name Data-Yichao-5 \
    --lif-path /path/to/file.lif \
    [--flat-output /path/to/jpeg_all] \
    [--group-output /path/to/jpeg_all_by_position] \
    [--pair-output-root /path/to/analysis-outputs/yichao_instance_pairs] \
    [--resized-output-root /path/to/analysis-outputs/yichao_instance_pairs_resized_256] \
    [--preview-count 12] \
    [--preview-seed 20260406] \
    [--session-name yichao_data_y5_incremental]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --lif-path)
      LIF_PATH="$2"
      shift 2
      ;;
    --flat-output)
      FLAT_OUTPUT="$2"
      shift 2
      ;;
    --group-output)
      GROUP_OUTPUT="$2"
      shift 2
      ;;
    --pair-output-root)
      PAIR_OUTPUT_ROOT="$2"
      shift 2
      ;;
    --resized-output-root)
      RESIZED_OUTPUT_ROOT="$2"
      shift 2
      ;;
    --preview-count)
      PREVIEW_COUNT="$2"
      shift 2
      ;;
    --preview-seed)
      PREVIEW_SEED="$2"
      shift 2
      ;;
    --session-name)
      SESSION_NAME="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DATASET_NAME" || -z "$LIF_PATH" ]]; then
  usage >&2
  exit 2
fi

if [[ -z "$SESSION_NAME" ]]; then
  session_stub="$(printf '%s' "$DATASET_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')"
  SESSION_NAME="${session_stub}_incremental"
fi

LOG_DIR="$ROOT/analysis-outputs/yichao_incremental_pipeline_logs/$DATASET_NAME"
LOG_PATH="$LOG_DIR/full_run.log"
mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "log: $LOG_PATH"
  exit 0
fi

PIPELINE_CMD=(
  bash "$ROOT/analysis-tools/yichao_instance_pairs/run_yichao_dataset_incremental_pipeline.sh"
  --dataset-name "$DATASET_NAME"
  --lif-path "$LIF_PATH"
  --pair-output-root "$PAIR_OUTPUT_ROOT"
  --resized-output-root "$RESIZED_OUTPUT_ROOT"
  --preview-count "$PREVIEW_COUNT"
  --preview-seed "$PREVIEW_SEED"
)
if [[ -n "$FLAT_OUTPUT" ]]; then
  PIPELINE_CMD+=(--flat-output "$FLAT_OUTPUT")
fi
if [[ -n "$GROUP_OUTPUT" ]]; then
  PIPELINE_CMD+=(--group-output "$GROUP_OUTPUT")
fi

printf -v PIPELINE_CMD_STR '%q ' "${PIPELINE_CMD[@]}"
RUN_AND_IDLE_CMD="${PIPELINE_CMD_STR}; status=\$?; if [ \$status -eq 0 ]; then echo RESUME_FINISHED \$(date -Iseconds); else echo RESUME_FAILED \$(date -Iseconds) status=\$status; fi; echo SESSION_IDLE \$(date -Iseconds)"

tmux new-session -d -s "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" "cd \"$ROOT\"" C-m
tmux send-keys -t "$SESSION_NAME" "source /home/lachlan/miniconda3/etc/profile.d/conda.sh" C-m
tmux send-keys -t "$SESSION_NAME" "conda activate organoid" C-m
tmux send-keys -t "$SESSION_NAME" "export PYTHONNOUSERSITE=1" C-m
tmux send-keys -t "$SESSION_NAME" "export CUDA_VISIBLE_DEVICES=0" C-m
tmux send-keys -t "$SESSION_NAME" "exec > >(tee -a \"$LOG_PATH\") 2>&1" C-m
tmux send-keys -t "$SESSION_NAME" "echo RESUME_START \$(date -Iseconds)" C-m
tmux send-keys -t "$SESSION_NAME" "$RUN_AND_IDLE_CMD" C-m

echo "started tmux session: $SESSION_NAME"
echo "log: $LOG_PATH"
