#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONNOUSERSITE=1
ORGANOID_PYTHON="/home/lachlan/miniconda3/envs/organoid/bin/python"
"$ORGANOID_PYTHON" \
  "$ROOT/analysis-tools/yichao_instance_pairs/build_yichao_instance_pair_database.py" \
  "$@"
