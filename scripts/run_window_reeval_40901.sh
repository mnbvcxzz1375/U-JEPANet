#!/bin/bash
# Re-eval dynamic-crop arms with window intensity (matches training).
# Run on 40901; GPUs 0/1 for two arms each in separate invocations.
set -euo pipefail
ROOT=/data/hyc/U-JEPANet
PY=/home/ubuntu/anaconda3/envs/vllmenv/bin/python
export PYTHONPATH=$ROOT
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
WORD=/data/hyc/PLS4MIS/code/datasets/WORD
SPLIT=$ROOT/data/splits_sll20/test_30.txt
OUTDIR=$ROOT/runs/official_test_20260917/window
mkdir -p "$OUTDIR"
GPU=${1:?gpu 0|1}
ARM=${2:?arm A0D|C1|C2|C3|C4}
case "$ARM" in
  A0D) CK=/data/hyc/U-JEPANet/runs/school_local_copy/a0d_best.pt ;;
  C1) CK=$ROOT/runs/c_ladder_40901/C1/best.pt ;;
  C2) CK=$ROOT/runs/c_ladder_40901/C2/best.pt ;;
  C3) CK=$ROOT/runs/c_ladder_40901/C3/best.pt ;;
  C4) CK=$ROOT/runs/c_ladder_40901/C4/best.pt ;;
  *) echo unknown; exit 2 ;;
esac
# C1/C2 may only live on school — allow remote paths if present
if [ ! -f "$CK" ]; then
  echo "missing $CK"; exit 3
fi
export CUDA_VISIBLE_DEVICES=$GPU
"$PY" "$ROOT/scripts/eval_official_test.py" \
  --word-root "$WORD" \
  --test-ids "$SPLIT" \
  --arms A2 \
  --arm-ckpt "A2=$CK" \
  --intensity-mode window \
  --out "$OUTDIR/test_${ARM}_window.json"
echo "DONE $ARM window ALL=$($PY -c "import json;print(json.load(open('$OUTDIR/test_${ARM}_window.json'))['arms']['A2']['per_organ']['ALL'])")"
