#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ORGANOID_PYTHON="/home/lachlan/miniconda3/envs/organoid/bin/python"
DATASET_NAME=""
LIF_PATH=""
FLAT_OUTPUT=""
GROUP_OUTPUT=""
GROUP_MODE="link"
PAIR_OUTPUT_ROOT="$ROOT/analysis-outputs/yichao_instance_pairs"
RESIZED_OUTPUT_ROOT="$ROOT/analysis-outputs/yichao_instance_pairs_resized_256"
PREVIEW_COUNT=12
PREVIEW_SEED=20260406

usage() {
  cat <<'EOF'
Usage:
  bash analysis-tools/yichao_instance_pairs/run_yichao_dataset_incremental_pipeline.sh \
    --dataset-name Data-Yichao-5 \
    --lif-path /path/to/file.lif \
    [--flat-output /path/to/jpeg_all] \
    [--group-output /path/to/jpeg_all_by_position] \
    [--pair-output-root /path/to/analysis-outputs/yichao_instance_pairs] \
    [--resized-output-root /path/to/analysis-outputs/yichao_instance_pairs_resized_256] \
    [--group-mode link|copy|move] \
    [--preview-count 12] \
    [--preview-seed 20260406]
EOF
}

resolve_path() {
  "$ORGANOID_PYTHON" - <<'PY' "$1"
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
}

count_jpegs() {
  find "$1" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) 2>/dev/null | wc -l | tr -d ' '
}

count_groups() {
  find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' '
}

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date -Iseconds)" "$message"
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
    --group-mode)
      GROUP_MODE="$2"
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

LIF_PATH="$(resolve_path "$LIF_PATH")"
if [[ ! -f "$LIF_PATH" ]]; then
  echo "Missing LIF file: $LIF_PATH" >&2
  exit 1
fi

if [[ -z "$FLAT_OUTPUT" ]]; then
  FLAT_OUTPUT="$(dirname "$LIF_PATH")/$(basename "${LIF_PATH%.lif}")_jpeg_all"
fi
if [[ -z "$GROUP_OUTPUT" ]]; then
  GROUP_OUTPUT="${FLAT_OUTPUT}_by_position"
fi

FLAT_OUTPUT="$(resolve_path "$FLAT_OUTPUT")"
GROUP_OUTPUT="$(resolve_path "$GROUP_OUTPUT")"
PAIR_OUTPUT_ROOT="$(resolve_path "$PAIR_OUTPUT_ROOT")"
RESIZED_OUTPUT_ROOT="$(resolve_path "$RESIZED_OUTPUT_ROOT")"

log "dataset incremental pipeline start dataset=$DATASET_NAME lif=$LIF_PATH"
log "extract start flat_output=$FLAT_OUTPUT"
"$ORGANOID_PYTHON" "$ROOT/BioAgentUtils/lif_to_jpeg.py" "$LIF_PATH" -o "$FLAT_OUTPUT" --quality 95
log "extract done jpeg_count=$(count_jpegs "$FLAT_OUTPUT")"

log "organize start grouped_output=$GROUP_OUTPUT mode=$GROUP_MODE"
"$ORGANOID_PYTHON" "$ROOT/BioAgentUtils/organize_lif_jpegs.py" "$FLAT_OUTPUT" -o "$GROUP_OUTPUT" --mode "$GROUP_MODE"
log "organize done group_count=$(count_groups "$GROUP_OUTPUT")"

log "instance pair pipeline start dataset=$DATASET_NAME output_root=$PAIR_OUTPUT_ROOT"
bash "$ROOT/analysis-tools/yichao_instance_pairs/run_yichao_instance_pair_pipeline.sh" \
  --output-root "$PAIR_OUTPUT_ROOT" \
  --datasets "$DATASET_NAME"
log "instance pair pipeline done"

DB_PATH="$PAIR_OUTPUT_ROOT/database/instance_pairs.sqlite"
if [[ ! -f "$DB_PATH" ]]; then
  echo "Missing instance-pair database after extraction: $DB_PATH" >&2
  exit 1
fi

log "resize export start output_root=$RESIZED_OUTPUT_ROOT"
"$ORGANOID_PYTHON" "$ROOT/differentiation_prediction/yichao_instance_pairs_256/prepare_dataset.py" \
  --db-path "$DB_PATH" \
  --output-root "$RESIZED_OUTPUT_ROOT" \
  --size 256 \
  --incremental
log "resize export done"

log "preview start dataset_filter=$DATASET_NAME count=$PREVIEW_COUNT seed=$PREVIEW_SEED"
"$ORGANOID_PYTHON" "$ROOT/differentiation_prediction/yichao_instance_pairs_256/preview_random_pairs.py" \
  --dataset-root "$RESIZED_OUTPUT_ROOT" \
  --datasets "$DATASET_NAME" \
  --count "$PREVIEW_COUNT" \
  --seed "$PREVIEW_SEED"
log "preview done"

log "dataset incremental pipeline finished dataset=$DATASET_NAME"
printf '%s\n' "$FLAT_OUTPUT" "$GROUP_OUTPUT" "$PAIR_OUTPUT_ROOT" "$RESIZED_OUTPUT_ROOT"
