#!/bin/bash
#SBATCH --job-name=ujepa-a0da
#SBATCH --partition=hpc_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/heyecheng/U-JEPANet/runs/school/%x-%j.out
#SBATCH --error=/public/home/heyecheng/U-JEPANet/runs/school/%x-%j.err
# A0DA: U-Net + dynamic crop + CT-med (strength=1) + NO JEPA
# Fills missing 2x2 cell M(J=0, A=1)
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
export PYTHONPATH=$ROOT
export PYTHONUNBUFFERED=1
WORD=/public/share/td20230405/WORD
OUT=$ROOT/runs/school/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$OUT" "$ROOT/runs/school"
export UJEPA_HU_CACHE=${SLURM_TMPDIR:-/tmp}/ujepa_hu_cache_${SLURM_JOB_ID}
mkdir -p "$UJEPA_HU_CACHE"

echo "host=$(hostname) ARM=A0DA intensity=window"
"$PY" "$ROOT/scripts/train_c_ladder.py" \
  --arm A0DA \
  --word-root "$WORD" \
  --split-dir "$ROOT/data/splits_sll20" \
  --cache-dir "$UJEPA_HU_CACHE" \
  --out "$OUT" \
  --steps 30000 \
  --batch 2 \
  --device cuda \
  --seed 42 \
  --intensity-mode window \
  2>&1 | tee "$OUT/train.log"
echo "DONE $OUT"
