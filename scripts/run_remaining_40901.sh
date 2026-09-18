#!/bin/bash
# Remaining experiment fleet for post-audit C-series.
# Usage examples on 40901:
#   nohup bash run_remaining_40901.sh GPU ARM SEED &
# ARM: A0D | A0DA | C3R | C4P
set -euo pipefail
GPU=${1:?gpu}
ARM=${2:?arm}
SEED=${3:-42}
ROOT=/data/hyc/U-JEPANet
PY=/home/ubuntu/anaconda3/envs/vllmenv/bin/python
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_ALLOC_CONF=expandable_segments:True
WORD=/data/hyc/PLS4MIS/code/datasets/WORD
SPLIT=$ROOT/data/splits_sll20
OUT=$ROOT/runs/remaining_20260917/${ARM}_s${SEED}
CACHE=${SLURM_TMPDIR:-/tmp}/ujepa_hu_${ARM}_${SEED}_$$
mkdir -p "$OUT" "$CACHE"
echo "host=$(hostname) arm=$ARM seed=$SEED gpu=$GPU out=$OUT"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
# C4P uses paired weak/strong; C3R is C3 with fixed z-sampler + window val
TRAIN_ARM=$ARM
[ "$ARM" = "C4P" ] && TRAIN_ARM=C4
[ "$ARM" = "C3R" ] && TRAIN_ARM=C3
"$PY" "$ROOT/scripts/train_c_ladder.py" \
  --arm "$TRAIN_ARM" \
  --word-root "$WORD" \
  --split-dir "$SPLIT" \
  --cache-dir "$CACHE" \
  --out "$OUT" \
  --steps 30000 \
  --batch 2 \
  --device cuda \
  --seed "$SEED" \
  --intensity-mode window \
  2>&1 | tee "$OUT/train.log"
echo "DONE $OUT"
# official test window
INF=A0
case "$TRAIN_ARM" in
  A0D|A0DA) INF=A0 ;;
  *) INF=A2 ;;
esac
mkdir -p $ROOT/runs/official_test_20260917/window
"$PY" "$ROOT/scripts/eval_official_test.py" \
  --word-root "$WORD" \
  --test-ids "$SPLIT/test_30.txt" \
  --arms "$INF" --arm-ckpt "${INF}=$OUT/best.pt" \
  --intensity-mode window \
  --out "$ROOT/runs/official_test_20260917/window/test_${ARM}_s${SEED}_window.json"
echo "TEST_DONE ${ARM}_s${SEED}"
