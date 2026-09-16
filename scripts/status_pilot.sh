#!/bin/bash
set -euo pipefail
echo "=== processes ==="
ps aux | grep -E 'train.py|pilot_3k' | grep -v grep || true
echo "=== gpu ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== run dirs ==="
ls -dt /data/hyc/U-JEPANet/runs/pilot_3k_2026* 2>/dev/null | sed -n '1,3p' || true
RUN=$(ls -dt /data/hyc/U-JEPANet/runs/pilot_3k_2026* 2>/dev/null | sed -n '1p' || true)
echo "RUN=${RUN}"
if [ -n "${RUN}" ]; then
  echo "ab57558" > "${RUN}/COMMIT_SHA"
  echo "=== driver tail ==="
  tail -20 /data/hyc/U-JEPANet/runs/pilot_3k_driver.log || true
  for arm in a0 a1 a2 a3; do
    if [ -f "${RUN}/${arm}.log" ]; then
      echo "=== ${arm} last ==="
      tail -6 "${RUN}/${arm}.log"
      echo "=== ${arm} val ==="
      grep 'val@' "${RUN}/${arm}.log" | tail -4 || true
    fi
  done
  if [ -f "${RUN}/summary.json" ]; then
    echo "=== summary.json ==="
    cat "${RUN}/summary.json"
  fi
fi
echo "=== python ==="
/home/ubuntu/anaconda3/envs/vllmenv/bin/python -c "import torch; print('vllmenv', torch.__version__, torch.cuda.is_available())"
