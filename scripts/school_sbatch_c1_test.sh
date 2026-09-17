#!/bin/bash
#SBATCH --job-name=ujepa-c1test
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_%j_c1test.out
#SBATCH --error=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_%j_c1test.err
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
WORD=/public/share/td20230405/WORD
SPLIT=$ROOT/data/splits_sll20
OUTDIR=$ROOT/runs/official_test_20260917
CK=/public/home/heyecheng/U-JEPANet/runs/school/ujepa-c1_381886/best.pt

export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTDIR"

echo "host=$(hostname) ck=$CK"
ls -lh "$CK"
"$PY" -c "import torch; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# C1 is A2 architecture (U-Net + JEPA head, dynamic crop, no CT aug)
"$PY" "$ROOT/scripts/eval_official_test.py" \
  --word-root "$WORD" \
  --test-ids "$SPLIT/test_30.txt" \
  --arms A2 \
  --arm-ckpt "A2=${CK}" \
  --out "$OUTDIR/test_C1.json" \
  --device cuda

echo "C1_TEST_DONE"
# also print ALL
"$PY" - <<'PY'
import json
d=json.load(open("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_C1.json"))
for a,v in d["arms"].items():
    print("C1_as", a, "ALL", v["per_organ"]["ALL"])
print("per_organ", json.dumps(d["arms"]["A2"]["per_organ"], indent=2))
PY
