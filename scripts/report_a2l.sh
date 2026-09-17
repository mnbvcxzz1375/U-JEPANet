#!/bin/bash
set -euo pipefail
echo "=== processes ==="
ps aux | grep a2l_vs | grep -v grep || echo none
echo "=== gpu ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for m in L LU; do
  echo "===== $m ====="
  tail -8 /data/hyc/U-JEPANet/runs/a2l_${m}.log 2>/dev/null || tail -8 /data/hyc/U-JEPANet/runs/a2lu_${m}.log 2>/dev/null || echo no_log
  if [ -f /data/hyc/U-JEPANet/runs/a2l_lu/${m}/summary.json ]; then
    cat /data/hyc/U-JEPANet/runs/a2l_lu/${m}/summary.json
  fi
  grep 'val@' /data/hyc/U-JEPANet/runs/a2l_${m}.log 2>/dev/null | tail -5 || true
  grep 'val@' /data/hyc/U-JEPANet/runs/a2lu_${m}.log 2>/dev/null | tail -5 || true
done
