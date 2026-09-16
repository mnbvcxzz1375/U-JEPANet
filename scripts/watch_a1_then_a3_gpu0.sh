#!/usr/bin/env bash
# Wait until A1 has train_log.json (or process gone), then launch A3 on GPU0.
set -euo pipefail
ROOT=/data/hyc/U-JEPANet
RUN=/data/hyc/U-JEPANet/runs/sll20_30k_20260916_1923
LOG=/data/hyc/U-JEPANet/runs/a3_gpu0_watcher.log
echo "watcher start $(date -Is)" | tee "$LOG"
while true; do
  if [ -f "$RUN/a1/train_log.json" ]; then
    echo "A1 complete $(date -Is)" | tee -a "$LOG"
    break
  fi
  if ! pgrep -f "train.py.*a1.yaml" >/dev/null 2>&1; then
    echo "A1 process gone without train_log $(date -Is)" | tee -a "$LOG"
    # still try A3
    break
  fi
  sleep 60
done
echo "launch A3 on GPU0 $(date -Is)" | tee -a "$LOG"
nohup env PYTHONUNBUFFERED=1 bash "$ROOT/scripts/run_arm_gpu.sh" a3 0 \
  > /data/hyc/U-JEPANet/runs/a3_gpu0.log 2>&1 &
echo "A3_PID=$!" | tee -a "$LOG"
sleep 15
tail -20 /data/hyc/U-JEPANet/runs/a3_gpu0.log | tee -a "$LOG" || true
