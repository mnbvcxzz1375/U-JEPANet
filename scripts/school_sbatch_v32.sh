#!/bin/bash
#SBATCH --job-name=ujepa-v32
#SBATCH --partition=hpc_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/heyecheng/U-JEPANet/runs/school/%x-%j.out
#SBATCH --error=/public/home/heyecheng/U-JEPANet/runs/school/%x-%j.err
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1
WORD=/public/share/td20230405/WORD
ARM=${1:?A0DA|H0|H1}
SEED=${2:-43}
OUT=$ROOT/runs/school/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${ARM}_s${SEED}
CACHE=${SLURM_TMPDIR:-/tmp}/ujepa_hu_v32_${ARM}_s${SEED}
mkdir -p "$OUT" "$CACHE"
echo "host=$(hostname) arm=$ARM seed=$SEED code=V3.2"
"$PY" "$ROOT/scripts/train_v32.py" \
  --arm "$ARM" \
  --word-root "$WORD" \
  --split-dir "$ROOT/data/splits_sll20" \
  --cache-dir "$CACHE" \
  --out "$OUT" \
  --steps 30000 --batch 2 --device cuda --seed "$SEED" \
  --intensity-mode window --alpha-init 0.04 \
  2>&1 | tee "$OUT/train.log"
echo "TRAIN_DONE $OUT"
python3 -c "import json;d=json.load(open('$OUT/summary.json'));print('VAL',d['best_val'],'alpha',d.get('alpha_final'))"
