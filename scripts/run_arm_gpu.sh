#!/usr/bin/env bash
# Run one SLL20-30k arm on an explicit physical GPU.
# Usage: run_arm_gpu.sh a1 0
set -euo pipefail
ARM=${1:?arm}
GPU=${2:?gpu}
ROOT=/data/hyc/U-JEPANet
PY=/home/ubuntu/anaconda3/envs/vllmenv/bin/python
export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
RUN_ROOT=/data/hyc/U-JEPANet/runs/sll20_30k_20260916_1923
mkdir -p "${RUN_ROOT}/${ARM}"
echo "host=$(hostname) arm=${ARM} physical_gpu=${GPU} date=$(date -Is)" | tee "${RUN_ROOT}/${ARM}_gpu${GPU}_provenance.txt"
if [ -f "${RUN_ROOT}/${ARM}/train_log.json" ]; then
  echo "SKIP ${ARM} already complete"
  exit 0
fi
echo "===== 30k ${ARM} on GPU${GPU} ====="
/usr/bin/time -f "elapsed=%e maxrss_kb=%M" \
  "$PY" "$ROOT/train.py" \
    --config "$ROOT/configs/sll20_30k/${ARM}.yaml" \
    --device cuda --seed 42 \
    --out "${RUN_ROOT}/${ARM}" \
  2>&1 | tee "${RUN_ROOT}/${ARM}_gpu${GPU}.log"
echo "ARM_DONE ${ARM} GPU${GPU}"
