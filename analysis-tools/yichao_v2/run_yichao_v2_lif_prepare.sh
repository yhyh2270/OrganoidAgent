#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ORGANOID_PYTHON="${ORGANOID_PYTHON:-/home/lachlan/miniconda3/envs/organoid/bin/python}"
V2_ROOT="${1:-${ROOT}/DATA-Yichao-v2}"
OUTPUT_ROOT="${2:-${ROOT}/analysis-outputs/yichao_v2_prepare}"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/prepare.log"

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "${LOG_FILE}"
}

count_jpegs() {
  if [[ ! -d "$1" ]]; then
    printf '0\n'
    return
  fi
  find "$1" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) 2>/dev/null | wc -l | tr -d ' '
}

log "Yichao v2 prepare start v2_root=${V2_ROOT} output_root=${OUTPUT_ROOT}"

log "inspect LIF metadata"
"${ORGANOID_PYTHON}" "${ROOT}/analysis-tools/yichao_v2/inspect_v2_lif_metadata.py" \
  --v2-root "${V2_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/metadata" 2>&1 | tee -a "${LOG_FILE}"

while IFS= read -r -d '' lif_path; do
  dataset_dir="$(dirname "${lif_path}")"
  lif_stem="$(basename "${lif_path}" .lif)"
  flat_output="${dataset_dir}/${lif_stem}_jpeg_all"
  grouped_output="${dataset_dir}/${lif_stem}_jpeg_all_by_position"

  if [[ "$(count_jpegs "${flat_output}")" -eq 0 ]]; then
    log "extract ${lif_path} -> ${flat_output}"
    "${ORGANOID_PYTHON}" "${ROOT}/BioAgentUtils/lif_to_jpeg.py" "${lif_path}" -o "${flat_output}" --quality 95 2>&1 | tee -a "${LOG_FILE}"
  else
    log "reuse extracted ${flat_output} jpg_count=$(count_jpegs "${flat_output}")"
  fi

  if [[ ! -d "${grouped_output}" || "$(find "${grouped_output}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')" -eq 0 ]]; then
    log "organize ${flat_output} -> ${grouped_output}"
    "${ORGANOID_PYTHON}" "${ROOT}/BioAgentUtils/organize_lif_jpegs.py" "${flat_output}" -o "${grouped_output}" --mode link 2>&1 | tee -a "${LOG_FILE}"
  else
    log "reuse grouped ${grouped_output}"
  fi
done < <(find "${V2_ROOT}" -mindepth 2 -maxdepth 2 -type f -name '*.lif' -print0 | sort -z)

log "prepare complete"
printf '%s\n' \
  "${V2_ROOT}" \
  "${OUTPUT_ROOT}/metadata/yichao_v2_lif_metadata_summary.json" \
  "${OUTPUT_ROOT}/metadata/yichao_v2_lif_series_metadata.csv" \
  "${LOG_FILE}"
