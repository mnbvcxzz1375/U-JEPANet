#!/bin/bash
set -euo pipefail
for i in $(seq 1 20); do
  if ps aux | grep eval_official | grep -v grep >/dev/null; then
    echo "poll $i running"
    sleep 10
  else
    echo "poll $i done"
    break
  fi
done
python3 - <<'PY'
import json
from pathlib import Path
D=Path("/data/hyc/U-JEPANet/runs/official_test_20260917/window")
for arm in ["C3","C4"]:
    p=D/f"test_{arm}_window.json"
    if p.exists():
        d=json.loads(p.read_text())
        print(arm, "window ALL", d["arms"]["A2"]["per_organ"]["ALL"], "mode", d.get("intensity_mode"))
    else:
        print(arm, "MISSING")
# compare to legacy
legacy={"C3":0.8203055778585197,"C4":0.8168372743076378}
for arm,v in legacy.items():
    p=D/f"test_{arm}_window.json"
    if p.exists():
        w=json.loads(p.read_text())["arms"]["A2"]["per_organ"]["ALL"]
        print(f"  {arm} window-legacy = {w-v:+.4f}")
PY
