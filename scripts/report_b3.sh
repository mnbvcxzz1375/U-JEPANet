#!/bin/bash
set -euo pipefail
echo "=== processes ==="
ps aux | grep -E 'b3_pretrain|grad_conflict|train.py' | grep -v grep || echo none
echo "=== gpu ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== b3 log ==="
tail -40 /data/hyc/U-JEPANet/runs/b3_run.log 2>/dev/null || echo no_log
echo "=== b3 out ==="
ls -la /data/hyc/U-JEPANet/runs/b3_pretrain_ft 2>/dev/null || echo no_out
if [ -f /data/hyc/U-JEPANet/runs/b3_pretrain_ft/summary.json ]; then
  echo "=== summary ==="
  cat /data/hyc/U-JEPANet/runs/b3_pretrain_ft/summary.json
fi
echo "=== val lines ==="
grep 'val@' /data/hyc/U-JEPANet/runs/b3_run.log 2>/dev/null || true
