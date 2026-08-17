#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_ROOT="$ROOT/analysis-outputs/yichao_projected_instance_pairs"
CHUNK_NEW_IMAGES="${YICHAO_PROJECTED_CHUNK_NEW_IMAGES:-50}"
MAX_ATTEMPTS_WITHOUT_PROGRESS="${YICHAO_PROJECTED_MAX_ATTEMPTS_WITHOUT_PROGRESS:-20}"
RETRY_SLEEP_SECONDS="${YICHAO_PROJECTED_RETRY_SLEEP_SECONDS:-3}"
ORGANOID_PYTHON="/home/lachlan/miniconda3/envs/organoid/bin/python"
DATASETS=()
TIME_AWARE_ONLY=0
BRIGHTFIELD_PROJECTION="${YICHAO_PROJECTED_BRIGHTFIELD_PROJECTION:-min}"
FLUORESCENCE_PROJECTION="${YICHAO_PROJECTED_FLUORESCENCE_PROJECTION:-max}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --datasets)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        DATASETS+=("$1")
        shift
      done
      ;;
    --time-aware-only)
      TIME_AWARE_ONLY=1
      shift
      ;;
    --brightfield-projection)
      BRIGHTFIELD_PROJECTION="$2"
      shift 2
      ;;
    --fluorescence-projection)
      FLUORESCENCE_PROJECTION="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--output-root PATH] [--datasets Data-Yichao-3 ...] [--time-aware-only] [--brightfield-projection min|max|mean|median] [--fluorescence-projection min|max|mean|median]"
      exit 0
      ;;
    *)
      if [[ "$1" == --* ]]; then
        echo "Unknown argument: $1" >&2
        exit 2
      fi
      OUTPUT_ROOT="$1"
      shift
      ;;
  esac
done

LOG_DIR="$OUTPUT_ROOT/logs"
PIPELINE_LOG="$LOG_DIR/pipeline.log"
STATUS_JSON="$OUTPUT_ROOT/pipeline_status.json"
mkdir -p "$LOG_DIR"

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date -Iseconds)" "$message" | tee -a "$PIPELINE_LOG"
}

count_completed_images() {
  if [[ ! -d "$OUTPUT_ROOT/images" ]]; then
    printf '0\n'
    return
  fi
  if [[ ${#DATASETS[@]} -eq 0 ]]; then
    find "$OUTPUT_ROOT/images" -name image_record.json -type f 2>/dev/null | wc -l | tr -d ' '
    return
  fi
  local total=0
  local dataset_count
  for dataset in "${DATASETS[@]}"; do
    dataset_count="$(find "$OUTPUT_ROOT/images/$dataset" -name image_record.json -type f 2>/dev/null | wc -l | tr -d ' ')"
    total=$((total + dataset_count))
  done
  printf '%s\n' "$total"
}

count_failure_records() {
  if [[ ! -d "$OUTPUT_ROOT/failures" ]]; then
    printf '0\n'
    return
  fi
  if [[ ${#DATASETS[@]} -eq 0 ]]; then
    find "$OUTPUT_ROOT/failures" -name '*.json' -type f 2>/dev/null | wc -l | tr -d ' '
    return
  fi
  local total=0
  local dataset_count
  for dataset in "${DATASETS[@]}"; do
    dataset_count="$(find "$OUTPUT_ROOT/failures/$dataset" -name '*.json' -type f 2>/dev/null | wc -l | tr -d ' ')"
    total=$((total + dataset_count))
  done
  printf '%s\n' "$total"
}

target_args=()
if [[ ${#DATASETS[@]} -gt 0 ]]; then
  target_args+=(--datasets "${DATASETS[@]}")
fi
if [[ "$TIME_AWARE_ONLY" -eq 1 ]]; then
  target_args+=(--time-aware-only)
fi

target_images="$("$ORGANOID_PYTHON" "$ROOT/analysis-tools/yichao_projected_instance_pairs/project_and_segment_yichao_projected_pairs.py" --count-only "${target_args[@]}")"
attempt=0
no_progress_attempts=0
dataset_filter="${DATASETS[*]:-ALL}"

write_status() {
  local status="$1"
  local completed_images="$2"
  local failure_records="$3"
  local attempt_value="$4"
  local no_progress_value="$5"
  local last_exit_code="$6"
  "$ORGANOID_PYTHON" - <<'PY' "$STATUS_JSON" "$OUTPUT_ROOT" "$status" "$target_images" "$completed_images" "$failure_records" "$CHUNK_NEW_IMAGES" "$attempt_value" "$no_progress_value" "$last_exit_code"
from pathlib import Path
import json
import sys
from datetime import datetime

path = Path(sys.argv[1])
target = int(sys.argv[4])
completed = int(sys.argv[5])
payload = {
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "output_root": sys.argv[2],
    "status": sys.argv[3],
    "target_projected_images": target,
    "completed_projected_image_records": completed,
    "remaining_projected_images": max(0, target - completed),
    "failure_records": int(sys.argv[6]),
    "chunk_new_images": int(sys.argv[7]),
    "attempt": int(sys.argv[8]),
    "consecutive_attempts_without_progress": int(sys.argv[9]),
    "last_exit_code": int(sys.argv[10]),
}
tmp = path.with_name(path.name + ".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
tmp.replace(path)
PY
}

log "projected pipeline start output_root=$OUTPUT_ROOT datasets=$dataset_filter target_projected_images=$target_images chunk_new_images=$CHUNK_NEW_IMAGES time_aware_only=$TIME_AWARE_ONLY brightfield_projection=$BRIGHTFIELD_PROJECTION fluorescence_projection=$FLUORESCENCE_PROJECTION"

initial_completed="$(count_completed_images)"
initial_failures="$(count_failure_records)"
if [[ "$initial_completed" -ge "$target_images" ]]; then
  log "all projected image records already complete, building projected database"
  bash "$ROOT/analysis-tools/yichao_projected_instance_pairs/build_yichao_projected_instance_pair_database.sh" --output-root "$OUTPUT_ROOT"
  write_status "finished" "$initial_completed" "$initial_failures" 0 0 0
  log "projected pipeline finished successfully"
  exit 0
fi

while true; do
  attempt=$((attempt + 1))
  before_completed="$(count_completed_images)"
  before_failures="$(count_failure_records)"
  write_status "running" "$before_completed" "$before_failures" "$attempt" "$no_progress_attempts" 0
  log "projected chunk start attempt=$attempt completed_before=$before_completed failures_before=$before_failures"

  cmd=(
    "$ORGANOID_PYTHON"
    "$ROOT/analysis-tools/yichao_projected_instance_pairs/project_and_segment_yichao_projected_pairs.py"
    --gpu true
    --output-root "$OUTPUT_ROOT"
    --max-new-images "$CHUNK_NEW_IMAGES"
    --brightfield-projection "$BRIGHTFIELD_PROJECTION"
    --fluorescence-projection "$FLUORESCENCE_PROJECTION"
  )
  if [[ ${#DATASETS[@]} -gt 0 ]]; then
    cmd+=(--datasets "${DATASETS[@]}")
  fi
  if [[ "$TIME_AWARE_ONLY" -eq 1 ]]; then
    cmd+=(--time-aware-only)
  fi

  if "${cmd[@]}"; then
    chunk_exit_code=0
  else
    chunk_exit_code=$?
  fi

  after_completed="$(count_completed_images)"
  after_failures="$(count_failure_records)"
  delta_completed=$((after_completed - before_completed))
  write_status "running" "$after_completed" "$after_failures" "$attempt" "$no_progress_attempts" "$chunk_exit_code"
  log "projected chunk end attempt=$attempt exit_code=$chunk_exit_code completed_after=$after_completed delta_completed=$delta_completed failures_after=$after_failures"

  if [[ "$after_completed" -ge "$target_images" ]]; then
    log "all projected image records complete, building projected database"
    bash "$ROOT/analysis-tools/yichao_projected_instance_pairs/build_yichao_projected_instance_pair_database.sh" --output-root "$OUTPUT_ROOT"
    write_status "finished" "$after_completed" "$after_failures" "$attempt" 0 0
    log "projected pipeline finished successfully"
    exit 0
  fi

  if [[ "$delta_completed" -gt 0 ]]; then
    no_progress_attempts=0
  else
    no_progress_attempts=$((no_progress_attempts + 1))
  fi
  write_status "running" "$after_completed" "$after_failures" "$attempt" "$no_progress_attempts" "$chunk_exit_code"
  if [[ "$no_progress_attempts" -ge "$MAX_ATTEMPTS_WITHOUT_PROGRESS" ]]; then
    log "projected pipeline stopping after $no_progress_attempts consecutive attempts without progress"
    write_status "error" "$after_completed" "$after_failures" "$attempt" "$no_progress_attempts" "$chunk_exit_code"
    exit 1
  fi
  log "sleeping ${RETRY_SLEEP_SECONDS}s before next projected chunk"
  sleep "$RETRY_SLEEP_SECONDS"
done
