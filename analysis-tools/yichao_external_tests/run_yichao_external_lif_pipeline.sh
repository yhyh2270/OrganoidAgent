#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ORGANOID_PYTHON="/home/lachlan/miniconda3/envs/organoid/bin/python"

SOURCE_LIF="${1:-/home/lachlan/Downloads/N39_TriRep_DF_8.lif}"
DATASET_NAME="${2:-Data-Yichao-11}"
LIF_STEM="${3:-N39_TriRep_DF_8}"

V1_DATA_ROOT="${ROOT}/Data-Yichao-v1"
DATASET_DIR="${V1_DATA_ROOT}/${DATASET_NAME}"
DEST_LIF="${DATASET_DIR}/${LIF_STEM}.lif"
FLAT_OUTPUT="${DATASET_DIR}/${LIF_STEM}_jpeg_all"
GROUP_OUTPUT="${DATASET_DIR}/${LIF_STEM}_jpeg_all_by_position"
PROJECTED_ROOT="${ROOT}/analysis-outputs/yichao_projected_instance_pairs"
EXTERNAL_ROOT="${ROOT}/analysis-outputs/yichao_external_tests/${DATASET_NAME}"
TARGET_ROOT="${EXTERNAL_ROOT}/fluorescence_segmentation_relaxed_targets"
RUN_ROOT="${ROOT}/analysis-outputs/yichao_fluorescence_continuous/runs/soft_suppressed_hybrid_unet_v2_noearly_metrics"
CHECKPOINT="${RUN_ROOT}/last_model.pt"
LOG_DIR="${EXTERNAL_ROOT}/logs"
LOG_FILE="${LOG_DIR}/pipeline.log"

mkdir -p "${DATASET_DIR}" "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "${LOG_FILE}"
}

count_jpegs() {
  find "$1" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) 2>/dev/null | wc -l | tr -d ' '
}

log "external pipeline start dataset=${DATASET_NAME} source_lif=${SOURCE_LIF}"
if [[ ! -f "${SOURCE_LIF}" ]]; then
  echo "Missing source LIF: ${SOURCE_LIF}" >&2
  exit 1
fi

if [[ ! -f "${DEST_LIF}" ]]; then
  log "copy lif to ${DEST_LIF}"
  cp -n "${SOURCE_LIF}" "${DEST_LIF}"
else
  log "lif already present ${DEST_LIF}"
fi

log "extract jpeg planes to ${FLAT_OUTPUT}"
"${ORGANOID_PYTHON}" "${ROOT}/BioAgentUtils/lif_to_jpeg.py" "${DEST_LIF}" -o "${FLAT_OUTPUT}" --quality 95 2>&1 | tee -a "${LOG_FILE}"
log "extract complete jpeg_count=$(count_jpegs "${FLAT_OUTPUT}")"

log "organize jpeg planes to ${GROUP_OUTPUT}"
"${ORGANOID_PYTHON}" "${ROOT}/BioAgentUtils/organize_lif_jpegs.py" "${FLAT_OUTPUT}" -o "${GROUP_OUTPUT}" --mode link 2>&1 | tee -a "${LOG_FILE}"

log "projected segmentation start"
YICHAO_PROJECTED_CHUNK_NEW_IMAGES="${YICHAO_PROJECTED_CHUNK_NEW_IMAGES:-50}" \
  bash "${ROOT}/analysis-tools/yichao_projected_instance_pairs/run_yichao_projected_instance_pair_pipeline.sh" \
  --output-root "${PROJECTED_ROOT}" \
  --datasets "${DATASET_NAME}" 2>&1 | tee -a "${LOG_FILE}"
log "projected segmentation complete"

SOURCE_MANIFEST="${EXTERNAL_ROOT}/manifests/${DATASET_NAME}_external_projected_instances_manifest.csv"
log "build external-only projected manifest ${SOURCE_MANIFEST}"
"${ORGANOID_PYTHON}" -m differentiation_prediction.yichao_fluorescence_segmentation.build_external_projected_manifest \
  --source-records "${PROJECTED_ROOT}/manifests/projected_instance_records.csv" \
  --dataset "${DATASET_NAME}" \
  --output-manifest "${SOURCE_MANIFEST}" \
  --split-name external_test 2>&1 | tee -a "${LOG_FILE}"

log "build relaxed clean targets ${TARGET_ROOT}"
"${ORGANOID_PYTHON}" -m differentiation_prediction.yichao_fluorescence_segmentation.build_clean_targets \
  --source-manifest "${SOURCE_MANIFEST}" \
  --output-root "${TARGET_ROOT}" \
  --preset relaxed \
  --qc-count 80 2>&1 | tee -a "${LOG_FILE}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

log "evaluate last-epoch continuous checkpoint ${CHECKPOINT}"
"${ORGANOID_PYTHON}" -m differentiation_prediction.yichao_fluorescence_segmentation.evaluate_continuous_checkpoint \
  --run-root "${RUN_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --target-root "${TARGET_ROOT}" \
  --split external_test \
  --batch-size 12 \
  --num-workers 0 \
  --output-json "${EXTERNAL_ROOT}/evaluation/${DATASET_NAME}_last_epoch_external_metrics.json" \
  --panel-output "${EXTERNAL_ROOT}/evaluation/${DATASET_NAME}_last_epoch_external_panel.png" 2>&1 | tee -a "${LOG_FILE}"

log "external pipeline finished dataset=${DATASET_NAME}"
printf '%s\n' \
  "${DEST_LIF}" \
  "${FLAT_OUTPUT}" \
  "${GROUP_OUTPUT}" \
  "${PROJECTED_ROOT}" \
  "${TARGET_ROOT}" \
  "${EXTERNAL_ROOT}/evaluation"
