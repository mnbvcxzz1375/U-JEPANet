#!/bin/bash
#SBATCH --job-name=ujepa-c2test
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_%j_c2test.out
#SBATCH --error=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_%j_c2test.err
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
WORD=/public/share/td20230405/WORD
SPLIT=$ROOT/data/splits_sll20
OUTDIR=$ROOT/runs/official_test_20260917
CK=/public/home/heyecheng/U-JEPANet/runs/school/ujepa-c1_382006/best.pt

export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTDIR"

echo "host=$(hostname) ck=$CK"
ls -lh "$CK"
"$PY" -c "import torch; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# C2 is A2 architecture + dynamic crop + CT-med seg aug
"$PY" "$ROOT/scripts/eval_official_test.py" \
  --word-root "$WORD" \
  --test-ids "$SPLIT/test_30.txt" \
  --arms A2 \
  --arm-ckpt "A2=${CK}" \
  --out "$OUTDIR/test_C2.json" \
  --device cuda

echo "C2_TEST_DONE"
"$PY" - <<'PY'
import json
d=json.load(open("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_C2.json"))
po=d["arms"]["A2"]["per_organ"]
print("C2 ALL", po["ALL"])
known={"A0":0.7674,"A2":0.7636,"A2-LU":0.7700,"C1":0.8118}
for a,v in known.items():
    print(f"  vs {a}: {po['ALL']-v:+.4f}")
PY
