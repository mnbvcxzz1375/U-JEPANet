#!/bin/bash
# V3.2 launch on 4090x. Usage: bash run_v32.sh HOST GPU ARM SEED
set -euo pipefail
HOST=${1:?}; GPU=${2:?}; ARM=${3:?A0DA|H0|H1|R1}; SEED=${4:-42}
ROOT=/data/hyc/U-JEPANet
PY=/home/ubuntu/anaconda3/envs/vllmenv/bin/python
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_ALLOC_CONF=expandable_segments:True
WORD=/data/hyc/PLS4MIS/code/datasets/WORD
SPLIT=$ROOT/data/splits_sll20
OUT=$ROOT/runs/v32/${ARM}_s${SEED}
CACHE=/tmp/ujepa_hu_v32/${HOST}_${ARM}_s${SEED}
mkdir -p "$OUT" "$CACHE" "$ROOT/runs/v32"
echo "host=$HOST arm=$ARM seed=$SEED gpu=$GPU code=V3.2"
"$PY" "$ROOT/scripts/train_v32.py" \
  --arm "$ARM" \
  --word-root "$WORD" \
  --split-dir "$SPLIT" \
  --cache-dir "$CACHE" \
  --out "$OUT" \
  --steps 30000 --batch 2 --device cuda --seed "$SEED" \
  --intensity-mode window --alpha-init 0.04 \
  2>&1 | tee "$OUT/train.log"
echo "TRAIN_DONE $OUT"
