#!/usr/bin/env bash
# Parallel official-test eval launcher helpers.
# Usage examples:
#   bash run_test_eval_parallel.sh local_gpu1 A1,A2
#   (school) submitted via sbatch separately
set -euo pipefail

MODE=${1:?mode}
ARMS=${2:?arms comma}

PY=${PY:-/home/ubuntu/anaconda3/envs/vllmenv/bin/python}
ROOT=${ROOT:-/data/hyc/U-JEPANet}
WORD=${WORD:-/data/hyc/PLS4MIS/code/datasets/WORD}
SPLIT=${SPLIT:-$ROOT/data/splits_sll20}
OUTDIR=${OUTDIR:-$ROOT/runs/official_test_20260917}
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTDIR"

IFS=',' read -ra ARM_ARR <<< "$ARMS"
for ARM in "${ARM_ARR[@]}"; do
  ARM=$(echo "$ARM" | tr '[:lower:]' '[:upper:]' | tr -d ' ')
  LOG="$OUTDIR/eval_${ARM}.log"
  JSON="$OUTDIR/test_${ARM}.json"
  if [[ -f "$JSON" ]]; then
    if "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('arms') else 1)" "$JSON"; then
      echo "SKIP $ARM already done"
      continue
    fi
  fi
  echo "LAUNCH $ARM -> $LOG"
  "$PY" "$ROOT/scripts/eval_official_test.py" \
    --word-root "$WORD" \
    --test-ids "$SPLIT/test_30.txt" \
    --arms "$ARM" \
    --out "$JSON" \
    --device cuda \
    >"$LOG" 2>&1
  echo "DONE $ARM ALL=$("$PY" -c "import json,sys; print(f\"{json.load(open(sys.argv[1]))['arms']['$ARM']['per_organ']['ALL']:.4f}\")" "$JSON")"
done
echo "ALL_REQUESTED_DONE $ARMS"
