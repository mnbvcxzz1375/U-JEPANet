#!/bin/bash
set -euo pipefail
ROOT=/data/hyc/U-JEPANet
echo "==== 40901 procs ===="
ps aux | grep -E 'run_remaining|train_c|eval_official' | grep -v grep || echo none
echo "==== remaining logs ===="
ls -lt $ROOT/runs/remaining_20260917/ 2>/dev/null | head
for f in $ROOT/runs/remaining_20260917/*.log; do
  [ -f "$f" ] || continue
  echo "-- $(basename $f) --"
  tail -6 "$f"
done
echo "==== summaries ===="
for d in $ROOT/runs/remaining_20260917/*/; do
  [ -d "$d" ] || continue
  if [ -f "$d/summary.json" ]; then
    python3 -c "import json;d=json.load(open('$d/summary.json'));print('$d', d.get('arm'), 'best', d.get('best_val'), 'int', d.get('intensity_mode'), 'jepa', d.get('jepa_mode'))"
  fi
done
echo "==== window JSONs 40901 ===="
W=$ROOT/runs/official_test_20260917/window
ls -lt $W 2>/dev/null
python3 - <<'PY'
import json
from pathlib import Path
D=Path("/data/hyc/U-JEPANet/runs/official_test_20260917/window")
for p in sorted(D.glob("test_*.json")):
    d=json.loads(p.read_text())
    print(p.name, d.get("intensity_mode"), {a:v["per_organ"]["ALL"] for a,v in d["arms"].items()})
PY
