#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PAIR_OUTPUT_ROOT="${1:-$ROOT/analysis-outputs/yichao_instance_pairs}"
RESIZED_OUTPUT_ROOT="${2:-$ROOT/analysis-outputs/yichao_instance_pairs_resized_256}"
ORGANOID_PYTHON="/home/lachlan/miniconda3/envs/organoid/bin/python"

bash "$ROOT/analysis-tools/yichao_instance_pairs/build_yichao_instance_pair_database.sh" \
  --output-root "$PAIR_OUTPUT_ROOT"

"$ORGANOID_PYTHON" \
  "$ROOT/differentiation_prediction/yichao_instance_pairs_256/prepare_dataset.py" \
  --db-path "$PAIR_OUTPUT_ROOT/database/instance_pairs.sqlite" \
  --output-root "$RESIZED_OUTPUT_ROOT" \
  --size 256 \
  --refresh-metadata
