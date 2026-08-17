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
Usage: run_yichao_per_z_b2f_pipeline.sh [SOURCE_LIF]

Runs the full original-z-plane Yichao B2F workflow:
1. segment Data-Yichao-11 per original z plane and append it to the existing per-z instance database
2. build train/val/test targets from Data-Yichao-1..10 only
3. train the continuous B2F model for 500 sampled epochs
4. evaluate the last-epoch model on Data-Yichao-11 as external holdout
5. render internal and external B/F/target/pred/error visualization galleries
EOF
  exit 0
fi
SOURCE_LIF="${1:-/home/lachlan/Downloads/N39_TriRep_DF_8.lif}"
LIF_STEM="N39_TriRep_DF_8"

PIPELINE_ROOT="${ROOT}/analysis-outputs/yichao_per_z_b2f_pipeline"
INSTANCE_ROOT="${ROOT}/analysis-outputs/yichao_instance_pairs"
TRAIN_MANIFEST="${PIPELINE_ROOT}/manifests/per_z_1to10_instance_manifest.csv"
TARGET_ROOT="${ROOT}/analysis-outputs/yichao_fluorescence_segmentation_relaxed_per_z_1to10"
RUN_ROOT="${ROOT}/analysis-outputs/yichao_fluorescence_continuous/runs/per_z_1to10_hybrid_unet_v1_500"
EXTERNAL_ROOT="${ROOT}/analysis-outputs/yichao_external_tests/${EXTERNAL_DATASET}_per_z"
EXTERNAL_TARGET_ROOT="${EXTERNAL_ROOT}/fluorescence_segmentation_relaxed_targets"
EXTERNAL_MANIFEST="${EXTERNAL_ROOT}/manifests/${EXTERNAL_DATASET}_external_per_z_instances_manifest.csv"
V1_DATA_ROOT="${ROOT}/Data-Yichao-v1"
DATASET_DIR="${V1_DATA_ROOT}/${EXTERNAL_DATASET}"
DEST_LIF="${DATASET_DIR}/${LIF_STEM}.lif"
FLAT_OUTPUT="${DATASET_DIR}/${LIF_STEM}_jpeg_all"
GROUP_OUTPUT="${DATASET_DIR}/${LIF_STEM}_jpeg_all_by_position"
LOG_DIR="${PIPELINE_ROOT}/logs"
LOG_FILE="${LOG_DIR}/pipeline.log"
SAMPLES_PER_EPOCH="${YICHAO_PER_Z_SAMPLES_PER_EPOCH:-20000}"

mkdir -p "${LOG_DIR}" "${DATASET_DIR}" "${PIPELINE_ROOT}/visualizations" "${EXTERNAL_ROOT}/logs"

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

target_ready() {
  local target_root="$1"
  local source_manifest="$2"
  "${ORGANOID_PYTHON}" - "$target_root" "$source_manifest" <<'PY'
import csv
import json
import sys
from pathlib import Path

target_root = Path(sys.argv[1])
source_manifest = Path(sys.argv[2])
summary_path = target_root / "manifests" / "segmentation_targets_summary.json"
manifest_path = target_root / "manifests" / "segmentation_targets_manifest.csv"
if not summary_path.exists() or not manifest_path.exists() or not source_manifest.exists():
    raise SystemExit(1)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
with source_manifest.open("r", newline="", encoding="utf-8") as handle:
    source_count = sum(1 for _ in csv.DictReader(handle))
if int(summary.get("count", -1)) != source_count:
    raise SystemExit(1)
if str(summary.get("source_manifest", "")) != str(source_manifest):
    raise SystemExit(1)
PY
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

log "per-z B2F pipeline start"
log "train datasets=${TRAIN_DATASETS[*]} external_dataset=${EXTERNAL_DATASET} samples_per_epoch=${SAMPLES_PER_EPOCH}"
log "outputs instance_root=${INSTANCE_ROOT} target_root=${TARGET_ROOT} run_root=${RUN_ROOT} external_root=${EXTERNAL_ROOT}"

ensure_external_lif_extracted

log "per-z segmentation/database start for external ${EXTERNAL_DATASET}"
YICHAO_CHUNK_NEW_IMAGES="${YICHAO_CHUNK_NEW_IMAGES:-50}" \
  bash "${ROOT}/analysis-tools/yichao_instance_pairs/run_yichao_instance_pair_pipeline.sh" \
    --output-root "${INSTANCE_ROOT}" \
    --datasets "${EXTERNAL_DATASET}" 2>&1 | tee -a "${LOG_FILE}"
log "per-z segmentation/database complete for external ${EXTERNAL_DATASET}"

log "build train/val/test per-z manifest for Data-Yichao-1..10 ${TRAIN_MANIFEST}"
"${ORGANOID_PYTHON}" -m differentiation_prediction.yichao_fluorescence_segmentation.build_instance_training_manifest \
  --source-records "${INSTANCE_ROOT}/manifests/instance_records.csv" \
  --datasets "${TRAIN_DATASETS[@]}" \
  --output-manifest "${TRAIN_MANIFEST}" \
  --manifest-policy per_z_original_no_projection 2>&1 | tee -a "${LOG_FILE}"

if target_ready "${TARGET_ROOT}" "${TRAIN_MANIFEST}"; then
  log "reuse existing relaxed clean targets for per-z train data ${TARGET_ROOT}"
else
  log "build relaxed clean targets for per-z train data ${TARGET_ROOT}"
  "${ORGANOID_PYTHON}" -m differentiation_prediction.yichao_fluorescence_segmentation.build_clean_targets \
    --source-manifest "${TRAIN_MANIFEST}" \
    --output-root "${TARGET_ROOT}" \
    --preset relaxed \
    --qc-count 120 \
    --max-overexposure-qc 120 2>&1 | tee -a "${LOG_FILE}"
fi

log "train continuous B2F model for 500 sampled epochs on per-z Data-Yichao-1..10"
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
  --samples-per-epoch "${SAMPLES_PER_EPOCH}" \
  --eval-every 2 \
  --panel-every 10 \
  --save-every 10 \
  --keep-periodic 8 \
  --resume 2>&1 | tee -a "${LOG_FILE}"
log "training complete"

log "build external-only Data-Yichao-11 per-z manifest ${EXTERNAL_MANIFEST}"
"${ORGANOID_PYTHON}" -m differentiation_prediction.yichao_fluorescence_segmentation.build_external_instance_manifest \
  --source-records "${INSTANCE_ROOT}/manifests/instance_records.csv" \
  --dataset "${EXTERNAL_DATASET}" \
  --output-manifest "${EXTERNAL_MANIFEST}" \
  --split-name external_test 2>&1 | tee -a "${LOG_FILE}"

if target_ready "${EXTERNAL_TARGET_ROOT}" "${EXTERNAL_MANIFEST}"; then
  log "reuse existing relaxed clean targets for external Data-Yichao-11 ${EXTERNAL_TARGET_ROOT}"
else
  log "build relaxed clean targets for external Data-Yichao-11 ${EXTERNAL_TARGET_ROOT}"
  "${ORGANOID_PYTHON}" -m differentiation_prediction.yichao_fluorescence_segmentation.build_clean_targets \
    --source-manifest "${EXTERNAL_MANIFEST}" \
    --output-root "${EXTERNAL_TARGET_ROOT}" \
    --preset relaxed \
    --qc-count 80 \
    --max-overexposure-qc 80 2>&1 | tee -a "${LOG_FILE}"
fi

log "evaluate last epoch model on external Data-Yichao-11 per-z"
"${ORGANOID_PYTHON}" -m differentiation_prediction.yichao_fluorescence_segmentation.evaluate_continuous_checkpoint \
  --run-root "${RUN_ROOT}" \
  --checkpoint "${RUN_ROOT}/last_model.pt" \
  --target-root "${EXTERNAL_TARGET_ROOT}" \
  --split external_test \
  --batch-size 12 \
  --num-workers 0 \
  --output-json "${EXTERNAL_ROOT}/evaluation/${EXTERNAL_DATASET}_per_z_last_epoch_external_metrics.json" \
  --panel-output "${EXTERNAL_ROOT}/evaluation/${EXTERNAL_DATASET}_per_z_last_epoch_external_panel.png" 2>&1 | tee -a "${LOG_FILE}"

log "render internal per-z test prediction gallery"
"${ORGANOID_PYTHON}" -m differentiation_prediction.yichao_fluorescence_segmentation.make_continuous_prediction_gallery \
  --run-root "${RUN_ROOT}" \
  --checkpoint "${RUN_ROOT}/last_model.pt" \
  --target-root "${TARGET_ROOT}" \
  --split test \
  --sample-count 80 \
  --output "${PIPELINE_ROOT}/visualizations/per_z_internal_test_gallery.png" 2>&1 | tee -a "${LOG_FILE}"

log "render external Data-Yichao-11 per-z prediction gallery"
"${ORGANOID_PYTHON}" -m differentiation_prediction.yichao_fluorescence_segmentation.make_continuous_prediction_gallery \
  --run-root "${RUN_ROOT}" \
  --checkpoint "${RUN_ROOT}/last_model.pt" \
  --target-root "${EXTERNAL_TARGET_ROOT}" \
  --split external_test \
  --sample-count 80 \
  --output "${PIPELINE_ROOT}/visualizations/per_z_Data-Yichao-11_external_gallery.png" 2>&1 | tee -a "${LOG_FILE}"

log "write per-z summary"
"${ORGANOID_PYTHON}" "${ROOT}/analysis-tools/yichao_per_z_b2f/write_yichao_per_z_b2f_summary.py" \
  --output-root "${PIPELINE_ROOT}" \
  --instance-root "${INSTANCE_ROOT}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --target-root "${TARGET_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --external-root "${EXTERNAL_ROOT}" 2>&1 | tee -a "${LOG_FILE}"

log "per-z B2F pipeline finished"
printf '%s\n' \
  "${PIPELINE_ROOT}" \
  "${INSTANCE_ROOT}" \
  "${TRAIN_MANIFEST}" \
  "${TARGET_ROOT}" \
  "${RUN_ROOT}" \
  "${EXTERNAL_ROOT}"
