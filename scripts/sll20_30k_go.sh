#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/hyc/U-JEPANet
PY=/home/ubuntu/anaconda3/envs/vllmenv/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
RUN_ROOT=/data/hyc/U-JEPANet/runs/sll20_30k_20260916_1923
mkdir -p "$RUN_ROOT"
echo "5ac7f87" > "$RUN_ROOT/COMMIT_SHA"
echo "host=$(hostname)" > "$RUN_ROOT/provenance.txt"
echo "commit=5ac7f87" >> "$RUN_ROOT/provenance.txt"
echo "date=$(date -Is)" >> "$RUN_ROOT/provenance.txt"
"$PY" -c "import torch; print('torch', torch.__version__)" >> "$RUN_ROOT/provenance.txt"
for ARM in a0 a1 a2 a3; do
  if [ -f "$RUN_ROOT/${ARM}/train_log.json" ]; then
    echo "skip $ARM"
    continue
  fi
  echo "===== 30k $ARM ====="
  /usr/bin/time -f "elapsed=%e maxrss_kb=%M" \
    "$PY" "$ROOT/train.py" \
      --config "$ROOT/configs/sll20_30k/${ARM}.yaml" \
      --device cuda --seed 42 \
      --out "$RUN_ROOT/${ARM}" \
    2>&1 | tee "$RUN_ROOT/${ARM}.log"
done
"$PY" -c "
import json
from pathlib import Path
root = Path('$RUN_ROOT')
rows=[]
for arm in ['a0','a1','a2','a3']:
    p=root/arm/'train_log.json'
    if not p.exists():
        rows.append({'arm':arm,'error':'missing'}); continue
    d=json.loads(p.read_text())
    vals=d.get('val_history') or []
    last=(d.get('history') or [{}])[-1]
    rows.append({'arm':arm,'steps':d.get('steps'),'elapsed_sec':d.get('elapsed_sec'),
      'best_val':d.get('best_mean_fg_dice'),
      'final_val':vals[-1].get('mean_fg_dice') if vals else None,
      'n_val':len(vals),'last_jepa':last.get('l_jepa'),'last_seg':last.get('l_seg')})
out={'run_root':str(root),'commit':'5ac7f87','protocol':'SLL20-30k-seed42','arms':rows}
print(json.dumps(out,indent=2))
(root/'summary.json').write_text(json.dumps(out,indent=2))
"
echo "TRAIN_DONE $RUN_ROOT"
