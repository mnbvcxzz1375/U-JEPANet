#!/bin/bash
set -euo pipefail
LOG=/data/hyc/U-JEPANet/runs/official_test_20260917/a0da_s42_eval.log
OUT=/data/hyc/U-JEPANet/runs/official_test_20260917/window/test_A0DA_s42_window.json
for i in $(seq 1 30); do
  if [ -f "$OUT" ] && grep -q ALL "$LOG" 2>/dev/null; then
    echo "done poll $i"; break
  fi
  if ps aux | grep a0da_s42_best | grep -v grep >/dev/null; then
    echo "poll $i running $(tail -1 $LOG | tr '\n' ' ')"
    sleep 10
  else
    echo "poll $i process gone"
    tail -20 "$LOG"
    break
  fi
done
ls -l "$OUT" 2>/dev/null
tail -25 "$LOG"
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/data/hyc/U-JEPANet/runs/official_test_20260917/window/test_A0DA_s42_window.json")
if p.exists():
    d=json.loads(p.read_text())
    print("A0DA_s42_ALL", d["arms"]["A0"]["per_organ"]["ALL"])
else:
    print("no json")
PY
