#!/bin/bash
#SBATCH --job-name=ujepa-a0d
#SBATCH --partition=hpc_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/heyecheng/U-JEPANet/runs/school/%x-%j.out
#SBATCH --error=/public/home/heyecheng/U-JEPANet/runs/school/%x-%j.err
# A0 + dynamic crop (no JEPA). Matches C1 crop protocol for attribution.
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
export PYTHONPATH=$ROOT
export PYTHONUNBUFFERED=1
WORD=/public/share/td20230405/WORD
OUT=$ROOT/runs/school/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$OUT" "$ROOT/runs/school"

echo "host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import torch; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

export UJEPA_HU_CACHE=${SLURM_TMPDIR:-/tmp}/ujepa_hu_cache_${SLURM_JOB_ID}
mkdir -p "$UJEPA_HU_CACHE"

echo "ARM=A0D OUT=$OUT cache=$UJEPA_HU_CACHE"
"$PY" "$ROOT/scripts/train_c_ladder.py" \
  --arm A0D \
  --word-root "$WORD" \
  --split-dir "$ROOT/data/splits_sll20" \
  --cache-dir "$UJEPA_HU_CACHE" \
  --out "$OUT" \
  --steps 30000 \
  --batch 2 \
  --device cuda \
  --seed 42 \
  2>&1 | tee "$OUT/train.log"

echo "DONE $OUT"
