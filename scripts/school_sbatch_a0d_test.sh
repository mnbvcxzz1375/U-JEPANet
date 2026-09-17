#!/bin/bash
#SBATCH --job-name=ujepa-a0dtest
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_%j_a0dtest.out
#SBATCH --error=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_%j_a0dtest.err
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
WORD=/public/share/td20230405/WORD
SPLIT=$ROOT/data/splits_sll20
OUTDIR=$ROOT/runs/official_test_20260917
CK=/public/home/heyecheng/U-JEPANet/runs/school/ujepa-a0d_383177/best.pt

export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTDIR"

echo "host=$(hostname) ck=$CK"
ls -lh "$CK"
"$PY" -c "import torch; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# A0D is pure UNet3D (arm=A0 in ckpt config)
"$PY" "$ROOT/scripts/eval_official_test.py" \
  --word-root "$WORD" \
  --test-ids "$SPLIT/test_30.txt" \
  --arms A0 \
  --arm-ckpt "A0=${CK}" \
  --out "$OUTDIR/test_A0D.json" \
  --device cuda

echo "A0D_TEST_DONE"
"$PY" - <<'PY'
import json
d=json.load(open("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_A0D.json"))
po=d["arms"]["A0"]["per_organ"]
print("A0D ALL", po["ALL"])
known={"A0":0.7674,"C1":0.8118,"C2":0.8169,"C3":0.8203,"C4":0.8168}
for a,v in known.items():
    print(f"  vs {a}: {po['ALL']-v:+.4f}")
PY
