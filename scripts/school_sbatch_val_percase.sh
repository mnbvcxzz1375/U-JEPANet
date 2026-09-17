#!/bin/bash
#SBATCH --job-name=ujepa-valpc
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/public/home/heyecheng/U-JEPANet/runs/val_percase_20260917/slurm_%j_%x.out
#SBATCH --error=/public/home/heyecheng/U-JEPANet/runs/val_percase_20260917/slurm_%j_%x.err
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
WORD=/public/share/td20230405/WORD
SPLIT=$ROOT/data/splits_sll20
CKPT=/public/home/heyecheng/ujepa_test_ckpts
OUTDIR=$ROOT/runs/val_percase_20260917
ARM=${1:?arm}

export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTDIR"

case "$ARM" in
  A0) CK=$CKPT/a0_best.pt ;;
  A1) CK=$CKPT/a1_best.pt ;;
  A2) CK=$CKPT/a2_best.pt ;;
  A3) CK=$CKPT/a3_best.pt ;;
  A2-L) CK=$CKPT/a2L_best.pt ;;
  A2-LU) CK=$CKPT/a2LU_best.pt ;;
  *) echo "unknown $ARM"; exit 2 ;;
esac
# fallback to 40901-era school copies if a0 missing
if [ ! -f "$CK" ]; then
  echo "missing $CK"; exit 3
fi

echo "host=$(hostname) arm=$ARM ck=$CK"
"$PY" "$ROOT/scripts/eval_per_case_val.py" \
  --word-root "$WORD" \
  --ids "$SPLIT/val_20.txt" \
  --image-dir imagesVal \
  --label-dir labelsVal \
  --arms "$ARM" \
  --arm-ckpt "${ARM}=${CK}" \
  --out "$OUTDIR/val_${ARM}.json"
echo "VAL_PERCASE_DONE $ARM"
