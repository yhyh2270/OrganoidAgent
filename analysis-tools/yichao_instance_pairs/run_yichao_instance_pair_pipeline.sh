#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_ROOT="$ROOT/analysis-outputs/yichao_instance_pairs"
CHUNK_NEW_IMAGES="${YICHAO_CHUNK_NEW_IMAGES:-300}"
MAX_ATTEMPTS_WITHOUT_PROGRESS="${YICHAO_MAX_ATTEMPTS_WITHOUT_PROGRESS:-5}"
RETRY_SLEEP_SECONDS="${YICHAO_RETRY_SLEEP_SECONDS:-8}"
ORGANOID_PYTHON="/home/lachlan/miniconda3/envs/organoid/bin/python"
DATASETS=()

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
    --help|-h)
      echo "Usage: $0 [--output-root PATH] [--datasets Data-Yichao-5 ...]"
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

resolve_target_images() {
  "$ORGANOID_PYTHON" - <<'PY' "$ROOT" "${DATASETS[@]}"
from pathlib import Path
import sys

repo_root = Path(sys.argv[1])
dataset_names = set(sys.argv[2:]) or None
sys.path.insert(0, str(repo_root / "analysis-tools" / "yichao_instance_pairs"))
from common import discover_work_items

print(len(discover_work_items(repo_root, dataset_names=dataset_names)))
PY
}

write_status() {
  local status="$1"
  local target_images="$2"
  local completed_images="$3"
  local failure_records="$4"
  local attempt="$5"
  local no_progress_attempts="$6"
  local last_exit_code="$7"
  "$ORGANOID_PYTHON" - <<'PY' "$STATUS_JSON" "$OUTPUT_ROOT" "$status" "$target_images" "$completed_images" "$failure_records" "$CHUNK_NEW_IMAGES" "$attempt" "$no_progress_attempts" "$last_exit_code"
from pathlib import Path
import json
import sys
from datetime import datetime

status_path = Path(sys.argv[1])
payload = {
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "output_root": sys.argv[2],
    "status": sys.argv[3],
    "target_images": int(sys.argv[4]),
    "completed_image_records": int(sys.argv[5]),
    "remaining_images": max(0, int(sys.argv[4]) - int(sys.argv[5])),
    "failure_records": int(sys.argv[6]),
    "chunk_new_images": int(sys.argv[7]),
    "attempt": int(sys.argv[8]),
    "consecutive_attempts_without_progress": int(sys.argv[9]),
    "last_exit_code": int(sys.argv[10]),
}
tmp_path = status_path.with_name(status_path.name + ".tmp")
tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
tmp_path.replace(status_path)
PY
}

target_images="$(resolve_target_images)"
attempt=0
no_progress_attempts=0
dataset_filter="${DATASETS[*]:-ALL}"

log "pipeline start output_root=$OUTPUT_ROOT datasets=$dataset_filter target_images=$target_images chunk_new_images=$CHUNK_NEW_IMAGES"

initial_completed="$(count_completed_images)"
initial_failures="$(count_failure_records)"
if [[ "$initial_completed" -ge "$target_images" ]]; then
  log "all image records already complete, building database"
  bash "$ROOT/analysis-tools/yichao_instance_pairs/build_yichao_instance_pair_database.sh" --output-root "$OUTPUT_ROOT"
  write_status "finished" "$target_images" "$initial_completed" "$initial_failures" 0 0 0
  log "pipeline finished successfully"
  exit 0
fi

while true; do
  attempt=$((attempt + 1))
  before_completed="$(count_completed_images)"
  before_failures="$(count_failure_records)"
  write_status "running" "$target_images" "$before_completed" "$before_failures" "$attempt" "$no_progress_attempts" 0
  log "chunk start attempt=$attempt completed_before=$before_completed failures_before=$before_failures"

  extraction_cmd=(
    bash "$ROOT/analysis-tools/yichao_instance_pairs/run_yichao_instance_pair_extraction.sh"
    --gpu true
    --output-root "$OUTPUT_ROOT"
    --max-new-images "$CHUNK_NEW_IMAGES"
  )
  if [[ ${#DATASETS[@]} -gt 0 ]]; then
    extraction_cmd+=(--datasets "${DATASETS[@]}")
  fi

  if "${extraction_cmd[@]}"; then
    chunk_exit_code=0
  else
    chunk_exit_code=$?
  fi

  after_completed="$(count_completed_images)"
  after_failures="$(count_failure_records)"
  delta_completed=$((after_completed - before_completed))
  write_status "running" "$target_images" "$after_completed" "$after_failures" "$attempt" "$no_progress_attempts" "$chunk_exit_code"
  log "chunk end attempt=$attempt exit_code=$chunk_exit_code completed_after=$after_completed delta_completed=$delta_completed failures_after=$after_failures"

  if [[ "$after_completed" -ge "$target_images" ]]; then
    log "all image records complete, building database"
    bash "$ROOT/analysis-tools/yichao_instance_pairs/build_yichao_instance_pair_database.sh" --output-root "$OUTPUT_ROOT"
    write_status "finished" "$target_images" "$after_completed" "$after_failures" "$attempt" 0 0
    log "pipeline finished successfully"
    exit 0
  fi

  if [[ "$delta_completed" -gt 0 ]]; then
    no_progress_attempts=0
  else
    no_progress_attempts=$((no_progress_attempts + 1))
  fi

  write_status "running" "$target_images" "$after_completed" "$after_failures" "$attempt" "$no_progress_attempts" "$chunk_exit_code"

  if [[ "$no_progress_attempts" -ge "$MAX_ATTEMPTS_WITHOUT_PROGRESS" ]]; then
    log "pipeline stopping after $no_progress_attempts consecutive attempts without progress"
    write_status "error" "$target_images" "$after_completed" "$after_failures" "$attempt" "$no_progress_attempts" "$chunk_exit_code"
    exit 1
  fi

  log "sleeping ${RETRY_SLEEP_SECONDS}s before next chunk"
  sleep "$RETRY_SLEEP_SECONDS"
done
