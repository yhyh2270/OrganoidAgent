#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ORGANOID_PYTHON="${ORGANOID_PYTHON:-/home/lachlan/miniconda3/envs/organoid/bin/python}"
V2_ROOT="${1:-${ROOT}/DATA-Yichao-v2}"
OUTPUT_ROOT="${2:-${ROOT}/analysis-outputs/yichao_v2_instance_pairs}"
CHUNK_NEW_IMAGES="${CHUNK_NEW_IMAGES:-80}"
GPU_MODE="${GPU_MODE:-auto}"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/pipeline.log"

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "${LOG_FILE}"
}

log "prepare LIF extraction and grouping"
bash "${ROOT}/analysis-tools/yichao_v2/run_yichao_v2_lif_prepare.sh" "${V2_ROOT}" "${ROOT}/analysis-outputs/yichao_v2_prepare" 2>&1 | tee -a "${LOG_FILE}"

log "discover v2 target images"
"${ORGANOID_PYTHON}" "${ROOT}/analysis-tools/yichao_v2_instance_pairs/run_yichao_v2_instance_pair_extraction.py" \
  --v2-root "${V2_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --count-only 2>&1 | tee -a "${LOG_FILE}"

log "run v2 segmentation/cropping chunks output_root=${OUTPUT_ROOT} chunk_new_images=${CHUNK_NEW_IMAGES}"
while true; do
  set +e
  "${ORGANOID_PYTHON}" "${ROOT}/analysis-tools/yichao_v2_instance_pairs/run_yichao_v2_instance_pair_extraction.py" \
    --v2-root "${V2_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --gpu "${GPU_MODE}" \
    --max-new-images "${CHUNK_NEW_IMAGES}" \
    --progress-every 20 2>&1 | tee -a "${LOG_FILE}"
  status=${PIPESTATUS[0]}
  set -e

  if [[ -f "${OUTPUT_ROOT}/run_summary.json" ]]; then
    run_status="$("${ORGANOID_PYTHON}" - <<PY
import json
from pathlib import Path
p=Path("${OUTPUT_ROOT}/run_summary.json")
print(json.loads(p.read_text()).get("status","unknown"))
PY
)"
    log "chunk status=${status} run_status=${run_status}"
    if [[ "${run_status}" == "finished" ]]; then
      break
    fi
  else
    log "chunk status=${status} no run_summary"
  fi

  if [[ "${status}" -ne 0 && "${status}" -ne 139 ]]; then
    log "pipeline failed in extraction status=${status}"
    exit "${status}"
  fi
  sleep 2
done

log "build v2 database"
"${ORGANOID_PYTHON}" "${ROOT}/analysis-tools/yichao_v2_instance_pairs/build_yichao_v2_instance_pair_database.py" \
  --output-root "${OUTPUT_ROOT}" 2>&1 | tee -a "${LOG_FILE}"

log "pipeline finished"
printf '%s\n' \
  "${OUTPUT_ROOT}" \
  "${OUTPUT_ROOT}/database/yichao_v2_instance_pairs.sqlite" \
  "${OUTPUT_ROOT}/manifests/summary.json" \
  "${LOG_FILE}"
