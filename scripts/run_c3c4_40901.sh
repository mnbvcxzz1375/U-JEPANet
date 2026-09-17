#!/usr/bin/env bash
# Launch C3 or C4 on an explicit 40901 GPU.
# Usage: run_c3c4.sh C3 0
set -euo pipefail
ARM=${1:?arm C3|C4}
GPU=${2:?gpu}
ROOT=/data/hyc/U-JEPANet
PY=/home/ubuntu/anaconda3/envs/vllmenv/bin/python
export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONPATH=$ROOT
export PYTHONUNBUFFERED=1
WORD=/data/hyc/PLS4MIS/code/datasets/WORD
SPLIT=$ROOT/data/splits_sll20
CACHE=${SLURM_TMPDIR:-/tmp}/ujepa_hu_cache_c_${ARM}_${GPU}
mkdir -p "$CACHE"
OUT=$ROOT/runs/c_ladder_40901/${ARM}
mkdir -p "$OUT"
echo "host=$(hostname) arm=${ARM} gpu=${GPU} date=$(date -Is) cache=$CACHE" | tee "$OUT/provenance.txt"
"$PY" "$ROOT/scripts/train_c_ladder.py" \
  --arm "$ARM" \
  --word-root "$WORD" \
  --split-dir "$SPLIT" \
  --cache-dir "$CACHE" \
  --out "$OUT" \
  --steps 30000 \
  --batch 2 \
  --device cuda \
  --seed 42 \
  2>&1 | tee "$OUT/train.log"
echo "DONE $OUT"
