#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${YICHAO_FUTURE_PYTHON:-/home/lachlan/miniconda3/envs/organoid/bin/python}"
OUTPUT_ROOT="${YICHAO_FUTURE_OUTPUT_ROOT:-$ROOT/analysis-outputs/yichao_future_expression}"
DB_PATH="${YICHAO_PROJECTED_DB:-$ROOT/analysis-outputs/yichao_projected_instance_pairs/database/projected_instance_pairs.sqlite}"
IMAGE_SIZE="${YICHAO_FUTURE_IMAGE_SIZE:-256}"
B2F_EPOCHS="${YICHAO_B2F_EPOCHS:-40}"
FEATURE_EPOCHS="${YICHAO_FEATURE_EPOCHS:-500}"
FUTURE_EPOCHS="${YICHAO_FUTURE_EPOCHS:-60}"
B2F_BATCH_SIZE="${YICHAO_B2F_BATCH_SIZE:-24}"
FUTURE_BATCH_SIZE="${YICHAO_FUTURE_BATCH_SIZE:-32}"
NUM_WORKERS="${YICHAO_FUTURE_NUM_WORKERS:-4}"
B2F_BASE_CHANNELS="${YICHAO_B2F_BASE_CHANNELS:-32}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*"
}

mkdir -p "$OUTPUT_ROOT/logs"

log "future-expression pipeline start output_root=$OUTPUT_ROOT image_size=$IMAGE_SIZE"
log "stage 0 build projected dataset"
"$PYTHON" "$ROOT/differentiation_prediction/yichao_future_expression/build_projected_dataset.py" \
  --db-path "$DB_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --size "$IMAGE_SIZE"

log "stage 1 train B2F feasibility model"
"$PYTHON" "$ROOT/differentiation_prediction/yichao_future_expression/train_b2f.py" \
  --data-root "$OUTPUT_ROOT" \
  --output-root "$OUTPUT_ROOT/stage1_b2f" \
  --image-size "$IMAGE_SIZE" \
  --epochs "$B2F_EPOCHS" \
  --batch-size "$B2F_BATCH_SIZE" \
  --base-channels "$B2F_BASE_CHANNELS" \
  --num-workers "$NUM_WORKERS" \
  --amp \
  --resume

log "stage 1b analyze explicit morphology features"
"$PYTHON" "$ROOT/differentiation_prediction/yichao_future_expression/analyze_features.py" \
  --data-root "$OUTPUT_ROOT" \
  --output-root "$OUTPUT_ROOT/stage1_feature_analysis" \
  --epochs "$FEATURE_EPOCHS"

log "stage 2 train early future-expression model"
"$PYTHON" "$ROOT/differentiation_prediction/yichao_future_expression/train_future_expression.py" \
  --data-root "$OUTPUT_ROOT" \
  --output-root "$OUTPUT_ROOT/stage2_future_expression" \
  --image-size "$IMAGE_SIZE" \
  --epochs "$FUTURE_EPOCHS" \
  --batch-size "$FUTURE_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --amp \
  --resume

log "future-expression pipeline finished"
printf '%s\n' "$OUTPUT_ROOT"
