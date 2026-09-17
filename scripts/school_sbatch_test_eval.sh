#!/bin/bash
#SBATCH --job-name=ujepa-test
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_%j_%x.out
#SBATCH --error=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_%j_%x.err

set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
WORD=/public/share/td20230405/WORD
SPLIT=$ROOT/data/splits_sll20
CKPT=/public/home/heyecheng/ujepa_test_ckpts
OUTDIR=$ROOT/runs/official_test_20260917
ARM=${1:?arm A1|A3|A2-L|A2-LU}

export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTDIR"
case "$ARM" in
  A1) CK=$CKPT/a1_best.pt ;;
  A2) CK=$CKPT/a2_best.pt ;;
  A3) CK=$CKPT/a3_best.pt ;;
  A2-L) CK=$CKPT/a2L_best.pt ;;
  A2-LU) CK=$CKPT/a2LU_best.pt ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

echo "host=$(hostname) arm=$ARM ckpt=$CK gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
"$PY" -c "import torch; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

"$PY" "$ROOT/scripts/eval_official_test.py" \
  --word-root "$WORD" \
  --test-ids "$SPLIT/test_30.txt" \
  --arms "$ARM" \
  --arm-ckpt "${ARM}=${CK}" \
  --out "$OUTDIR/test_${ARM}.json" \
  --device cuda

echo "SCHOOL_TEST_DONE $ARM"
