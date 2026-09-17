#!/bin/bash
set -euo pipefail
RUN=/data/hyc/U-JEPANet/runs/sll20_30k_20260916_1923
echo "=== processes ==="
ps aux | grep train.py | grep -v grep || echo none
echo "=== gpu ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== arms ==="
for arm in a0 a1 a2 a3; do
  echo "----- ${arm} -----"
  if [ -f "${RUN}/${arm}/train_log.json" ]; then
    echo "DONE"
    /home/ubuntu/anaconda3/envs/vllmenv/bin/python - <<PY
import json
from pathlib import Path
d=json.loads(Path("${RUN}/${arm}/train_log.json").read_text())
vals=d.get("val_history") or []
print("steps", d.get("steps"), "elapsed", round(float(d.get("elapsed_sec",0)),1))
print("best_val", d.get("best_mean_fg_dice"))
last=(d.get("history") or [{}])[-1]
print("last_seg", last.get("l_seg"), "last_jepa", last.get("l_jepa"), "lambda", last.get("lambda_j"))
for v in vals[-5:]:
    print(f"  val@{int(v.get('step',0))}: {float(v.get('mean_fg_dice', float('nan'))):.4f}")
PY
  else
    echo "RUNNING or missing"
    LOG=""
    for c in "${RUN}/${arm}.log" "${RUN}/${arm}_gpu0.log" "${RUN}/${arm}_gpu1.log" "/data/hyc/U-JEPANet/runs/${arm}_gpu0.log" "/data/hyc/U-JEPANet/runs/${arm}_gpu1.log"; do
      [ -f "$c" ] && LOG="$c"
    done
    if [ -n "$LOG" ]; then
      echo "log=$LOG"
      tail -8 "$LOG"
      grep 'val@' "$LOG" | tail -4 || true
    fi
  fi
done
echo "=== watcher ==="
cat /data/hyc/U-JEPANet/runs/a3_gpu0_watcher.log 2>/dev/null | tail -10 || true
tail -5 /data/hyc/U-JEPANet/runs/a3_gpu0.log 2>/dev/null || true
