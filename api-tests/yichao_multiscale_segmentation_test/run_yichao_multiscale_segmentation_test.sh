#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONNOUSERSITE=1
ORGANOID_PYTHON="/home/lachlan/miniconda3/envs/organoid/bin/python"
"$ORGANOID_PYTHON" \
  "$ROOT/api-tests/yichao_multiscale_segmentation_test/run_yichao_multiscale_segmentation_test.py" \
  "$@"
