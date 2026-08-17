#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${1:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[clear-gpu] nvidia-smi not found"
  exit 1
fi

TARGET_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v idx="$GPU_INDEX" '$1==idx {print $2}')"
if [[ -z "$TARGET_UUID" ]]; then
  echo "[clear-gpu] GPU index $GPU_INDEX not found"
  exit 1
fi

echo "[clear-gpu] target gpu index: $GPU_INDEX"
echo "[clear-gpu] target gpu uuid:  $TARGET_UUID"
echo "[clear-gpu] compute processes before:"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits || true

while IFS=',' read -r pid gpu_uuid used_mem; do
  pid="$(echo "$pid" | xargs)"
  gpu_uuid="$(echo "$gpu_uuid" | xargs)"
  if [[ -z "$pid" || -z "$gpu_uuid" ]]; then
    continue
  fi
  if [[ "$gpu_uuid" != "$TARGET_UUID" ]]; then
    continue
  fi
  if ! ps -p "$pid" >/dev/null 2>&1; then
    continue
  fi
  cmd="$(ps -p "$pid" -o cmd=)"
  if [[ "$cmd" == *python* ]]; then
    echo "[clear-gpu] killing pid=$pid mem=${used_mem}MiB cmd=$cmd"
    kill -9 "$pid" || true
  else
    echo "[clear-gpu] skip non-python pid=$pid cmd=$cmd"
  fi
done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits || true)

sleep 1
echo "[clear-gpu] compute processes after:"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits || true
