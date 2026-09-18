#!/bin/bash
#SBATCH --job-name=ujepa-winreeval
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_%j_win.out
#SBATCH --error=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_%j_win.err
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
WORD=/public/share/td20230405/WORD
SPLIT=$ROOT/data/splits_sll20/test_30.txt
OUTDIR=$ROOT/runs/official_test_20260917/window
ARM=${1:?arm}
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTDIR"
case "$ARM" in
  A0D) CK=$ROOT/runs/school/ujepa-a0d_383177/best.pt; INF=A0 ;;
  C1) CK=$ROOT/runs/school/ujepa-c1_381886/best.pt; INF=A2 ;;
  C2) CK=$ROOT/runs/school/ujepa-c1_382006/best.pt; INF=A2 ;;
  *) echo unknown $ARM; exit 2 ;;
esac
echo "host=$(hostname) ARM=$ARM inf=$INF intensity=window ck=$CK"
"$PY" "$ROOT/scripts/eval_official_test.py" \
  --word-root "$WORD" --test-ids "$SPLIT" \
  --arms "$INF" --arm-ckpt "${INF}=${CK}" \
  --intensity-mode window \
  --out "$OUTDIR/test_${ARM}_window.json"
"$PY" - <<PY
import json
d=json.load(open("$OUTDIR/test_${ARM}_window.json"))
print("${ARM}_window_ALL", d["arms"]["$INF"]["per_organ"]["ALL"])
PY
echo "DONE $ARM"
