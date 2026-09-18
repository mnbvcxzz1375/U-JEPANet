#!/bin/bash
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1
W=$ROOT/runs/official_test_20260917/window
CK=$ROOT/runs/school/ujepa-a0da_383499/best.pt
if [ ! -f "$W/test_A0DA_s42_window.json" ]; then
  echo "Running A0DA s42 window test"
  "$PY" $ROOT/scripts/eval_official_test.py \
    --word-root /public/share/td20230405/WORD \
    --test-ids $ROOT/data/splits_sll20/test_30.txt \
    --arms A0 --arm-ckpt "A0=$CK" \
    --intensity-mode window \
    --out "$W/test_A0DA_s42_window.json"
fi
python3 - <<'PY'
import json
from pathlib import Path
D=Path("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/window")
rows=[]
for p in sorted(D.glob("test_*.json")):
    d=json.loads(p.read_text())
    for a,v in d["arms"].items():
        rows.append((p.name, a, v["per_organ"]["ALL"], d.get("intensity_mode")))
for r in rows:
    print(f"{r[0]:40s} {r[1]:4s} {r[2]:.4f} {r[3]}")
PY
