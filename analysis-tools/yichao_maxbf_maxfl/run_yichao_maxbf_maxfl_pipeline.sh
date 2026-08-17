#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ORGANOID_PYTHON="/home/lachlan/miniconda3/envs/organoid/bin/python"

TRAIN_DATASETS=(
  Data-Yichao-1
  Data-Yichao-2
  Data-Yichao-3
  Data-Yichao-4
  Data-Yichao-5
  Data-Yichao-6
  Data-Yichao-7
  Data-Yichao-8
  Data-Yichao-9
  Data-Yichao-10
)
EXTERNAL_DATASET="Data-Yichao-11"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: run_yichao_maxbf_maxfl_pipeline.sh [SOURCE_LIF]

Runs the full max-brightfield / max-fluorescence Yichao B2F workflow:
1. project and segment Data-Yichao-1..10 with max/max projection
2. build train/val/test targets from Data-Yichao-1..10 only
3. train the continuous B2F model for 500 epochs
4. project and segment Data-Yichao-11 with max/max projection
5. evaluate the 500-epoch model on Data-Yichao-11 as external holdout
EOF
  exit 0
fi
SOURCE_LIF="${1:-/home/lachlan/Downloads/N39_TriRep_DF_8.lif}"
LIF_STEM="N39_TriRep_DF_8"

PIPELINE_ROOT="${ROOT}/analysis-outputs/yichao_maxbf_maxfl_pipeline"
PROJECTED_ROOT="${ROOT}/analysis-outputs/yichao_projected_instance_pairs_maxbf_maxfl"
TRAIN_MANIFEST="${PIPELINE_ROOT}/manifests/maxbf_maxfl_1to10_projected_instances_manifest.csv"
TARGET_ROOT="${ROOT}/analysis-outputs/yichao_fluorescence_segmentation_relaxed_maxbf_maxfl_1to10"
RUN_ROOT="${ROOT}/analysis-outputs/yichao_fluorescence_continuous/runs/maxbf_maxfl_1to10_hybrid_unet_v1_500"
EXTERNAL_ROOT="${ROOT}/analysis-outputs/yichao_external_tests/${EXTERNAL_DATASET}_maxbf_maxfl"
EXTERNAL_TARGET_ROOT="${EXTERNAL_ROOT}/fluorescence_segmentation_relaxed_targets"
EXTERNAL_MANIFEST="${EXTERNAL_ROOT}/manifests/${EXTERNAL_DATASET}_external_projected_instances_manifest.csv"
V1_DATA_ROOT="${ROOT}/Data-Yichao-v1"
DATASET_DIR="${V1_DATA_ROOT}/${EXTERNAL_DATASET}"
DEST_LIF="${DATASET_DIR}/${LIF_STEM}.lif"
FLAT_OUTPUT="${DATASET_DIR}/${LIF_STEM}_jpeg_all"
GROUP_OUTPUT="${DATASET_DIR}/${LIF_STEM}_jpeg_all_by_position"
LOG_DIR="${PIPELINE_ROOT}/logs"
LOG_FILE="${LOG_DIR}/pipeline.log"

mkdir -p "${LOG_DIR}" "${DATASET_DIR}" "${EXTERNAL_ROOT}/logs"

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

ensure_external_lif_extracted() {
  if [[ ! -f "${DEST_LIF}" ]]; then
    if [[ ! -f "${SOURCE_LIF}" ]]; then
      echo "Missing source LIF and destination LIF: ${SOURCE_LIF} / ${DEST_LIF}" >&2
      exit 1
    fi
    log "copy Data-Yichao-11 lif to ${DEST_LIF}"
    cp -n "${SOURCE_LIF}" "${DEST_LIF}"
  else
    log "Data-Yichao-11 lif already present ${DEST_LIF}"
  fi

  if [[ "$(count_jpegs "${FLAT_OUTPUT}")" -eq 0 ]]; then
    log "extract Data-Yichao-11 jpeg planes to ${FLAT_OUTPUT}"
    "${ORGANOID_PYTHON}" "${ROOT}/BioAgentUtils/lif_to_jpeg.py" "${DEST_LIF}" -o "${FLAT_OUTPUT}" --quality 95 2>&1 | tee -a "${LOG_FILE}"
  else
    log "reuse Data-Yichao-11 extracted jpeg planes count=$(count_jpegs "${FLAT_OUTPUT}")"
  fi

  if [[ ! -d "${GROUP_OUTPUT}" || "$(find "${GROUP_OUTPUT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')" -eq 0 ]]; then
    log "organize Data-Yichao-11 jpeg planes to ${GROUP_OUTPUT}"
    "${ORGANOID_PYTHON}" "${ROOT}/BioAgentUtils/organize_lif_jpegs.py" "${FLAT_OUTPUT}" -o "${GROUP_OUTPUT}" --mode link 2>&1 | tee -a "${LOG_FILE}"
  else
    log "reuse Data-Yichao-11 organized position folders ${GROUP_OUTPUT}"
  fi
}

run_projected_segmentation() {
  local label="$1"
  shift
  log "projected max/max segmentation start label=${label} datasets=$*"
  YICHAO_PROJECTED_CHUNK_NEW_IMAGES="${YICHAO_PROJECTED_CHUNK_NEW_IMAGES:-50}" \
    YICHAO_PROJECTED_BRIGHTFIELD_PROJECTION=max \
    YICHAO_PROJECTED_FLUORESCENCE_PROJECTION=max \
    bash "${ROOT}/analysis-tools/yichao_projected_instance_pairs/run_yichao_projected_instance_pair_pipeline.sh" \
      --output-root "${PROJECTED_ROOT}" \
      --brightfield-projection max \
      --fluorescence-projection max \
      --datasets "$@" 2>&1 | tee -a "${LOG_FILE}"
  log "projected max/max segmentation complete label=${label}"
}

log "max/max pipeline start"
log "train datasets=${TRAIN_DATASETS[*]} external_dataset=${EXTERNAL_DATASET}"
log "outputs projected_root=${PROJECTED_ROOT} target_root=${TARGET_ROOT} run_root=${RUN_ROOT} external_root=${EXTERNAL_ROOT}"

ensure_external_lif_extracted

run_projected_segmentation "train_1_to_10" "${TRAIN_DATASETS[@]}"

log "build train/val/test manifest for Data-Yichao-1..10 ${TRAIN_MANIFEST}"
"${ORGANOID_PYTHON}" -m differentiation_prediction.yichao_fluorescence_segmentation.build_projected_training_manifest \
  --source-records "${PROJECTED_ROOT}/manifests/projected_instance_records.csv" \
  --datasets "${TRAIN_DATASETS[@]}" \
  --output-manifest "${TRAIN_MANIFEST}" \
  --require-projection-mode max \
  --projection-policy max_brightfield_max_fluorescence 2>&1 | tee -a "${LOG_FILE}"

log "build relaxed clean targets for max/max projected train data ${TARGET_ROOT}"
"${ORGANOID_PYTHON}" -m differentiation_prediction.yichao_fluorescence_segmentation.build_clean_targets \
  --source-manifest "${TRAIN_MANIFEST}" \
  --output-root "${TARGET_ROOT}" \
  --preset relaxed \
  --qc-count 120 2>&1 | tee -a "${LOG_FILE}"

log "train continuous B2F model for 500 epochs on Data-Yichao-1..10"
"${ORGANOID_PYTHON}" -u -m differentiation_prediction.yichao_fluorescence_segmentation.train_continuous_target \
  --target-root "${TARGET_ROOT}" \
  --output-root "${RUN_ROOT}" \
  --image-size 256 \
  --epochs 500 \
  --batch-size 12 \
  --num-workers 0 \
  --base-channels 32 \
  --lr 1.5e-4 \
  --min-lr 1e-6 \
  --weight-decay 2e-4 \
  --balanced-sampler \
  --eval-every 2 \
  --panel-every 10 \
  --save-every 10 \
  --keep-periodic 8 \
  --resume 2>&1 | tee -a "${LOG_FILE}"
log "training complete"

run_projected_segmentation "external_data_yichao_11" "${EXTERNAL_DATASET}"

log "build external-only Data-Yichao-11 max/max projected manifest ${EXTERNAL_MANIFEST}"
"${ORGANOID_PYTHON}" -m differentiation_prediction.yichao_fluorescence_segmentation.build_external_projected_manifest \
  --source-records "${PROJECTED_ROOT}/manifests/projected_instance_records.csv" \
  --dataset "${EXTERNAL_DATASET}" \
  --output-manifest "${EXTERNAL_MANIFEST}" \
  --split-name external_test 2>&1 | tee -a "${LOG_FILE}"

log "build relaxed clean targets for external Data-Yichao-11 ${EXTERNAL_TARGET_ROOT}"
"${ORGANOID_PYTHON}" -m differentiation_prediction.yichao_fluorescence_segmentation.build_clean_targets \
  --source-manifest "${EXTERNAL_MANIFEST}" \
  --output-root "${EXTERNAL_TARGET_ROOT}" \
  --preset relaxed \
  --qc-count 80 2>&1 | tee -a "${LOG_FILE}"

log "evaluate last epoch model on external Data-Yichao-11"
"${ORGANOID_PYTHON}" -m differentiation_prediction.yichao_fluorescence_segmentation.evaluate_continuous_checkpoint \
  --run-root "${RUN_ROOT}" \
  --checkpoint "${RUN_ROOT}/last_model.pt" \
  --target-root "${EXTERNAL_TARGET_ROOT}" \
  --split external_test \
  --batch-size 12 \
  --num-workers 0 \
  --output-json "${EXTERNAL_ROOT}/evaluation/${EXTERNAL_DATASET}_maxbf_maxfl_last_epoch_external_metrics.json" \
  --panel-output "${EXTERNAL_ROOT}/evaluation/${EXTERNAL_DATASET}_maxbf_maxfl_last_epoch_external_panel.png" 2>&1 | tee -a "${LOG_FILE}"

log "write summary"
"${ORGANOID_PYTHON}" "${ROOT}/analysis-tools/yichao_maxbf_maxfl/write_yichao_maxbf_maxfl_summary.py" \
  --output-root "${PIPELINE_ROOT}" \
  --projected-root "${PROJECTED_ROOT}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --target-root "${TARGET_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --external-root "${EXTERNAL_ROOT}" 2>&1 | tee -a "${LOG_FILE}"

log "max/max pipeline finished"
printf '%s\n' \
  "${PIPELINE_ROOT}" \
  "${PROJECTED_ROOT}" \
  "${TRAIN_MANIFEST}" \
  "${TARGET_ROOT}" \
  "${RUN_ROOT}" \
  "${EXTERNAL_ROOT}"
