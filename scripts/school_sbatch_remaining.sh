#!/bin/bash
#SBATCH --job-name=ujepa-rem
#SBATCH --partition=hpc_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/heyecheng/U-JEPANet/runs/school/%x-%j.out
#SBATCH --error=/public/home/heyecheng/U-JEPANet/runs/school/%x-%j.err
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1
WORD=/public/share/td20230405/WORD
ARM=${1:?arm}
SEED=${2:-42}
OUT=$ROOT/runs/school/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${ARM}_s${SEED}
CACHE=${SLURM_TMPDIR:-/tmp}/ujepa_hu_${ARM}_${SEED}
mkdir -p "$OUT" "$CACHE"
TRAIN_ARM=$ARM
[ "$ARM" = "C4P" ] && TRAIN_ARM=C4
[ "$ARM" = "C3R" ] && TRAIN_ARM=C3
echo "host=$(hostname) arm=$TRAIN_ARM seed=$SEED out=$OUT"
"$PY" "$ROOT/scripts/train_c_ladder.py" \
  --arm "$TRAIN_ARM" \
  --word-root "$WORD" \
  --split-dir "$ROOT/data/splits_sll20" \
  --cache-dir "$CACHE" \
  --out "$OUT" \
  --steps 30000 --batch 2 --device cuda --seed "$SEED" \
  --intensity-mode window \
  2>&1 | tee "$OUT/train.log"
echo "TRAIN_DONE $OUT"
INF=A0
case "$TRAIN_ARM" in A0D|A0DA) INF=A0 ;; *) INF=A2 ;; esac
W=$ROOT/runs/official_test_20260917/window
mkdir -p "$W"
"$PY" "$ROOT/scripts/eval_official_test.py" \
  --word-root "$WORD" --test-ids "$ROOT/data/splits_sll20/test_30.txt" \
  --arms "$INF" --arm-ckpt "${INF}=$OUT/best.pt" \
  --intensity-mode window \
  --out "$W/test_${ARM}_s${SEED}_window.json"
echo "TEST_DONE ${ARM}_s${SEED} ALL=$($PY -c "import json;print(json.load(open('$W/test_${ARM}_s${SEED}_window.json'))['arms']['$INF']['per_organ']['ALL'])")"
