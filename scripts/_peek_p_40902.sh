#!/bin/bash
set -euo pipefail
ROOT=/data/hyc/U-JEPANet
echo "==== 40902 P1 ===="
tail -25 $ROOT/runs/predictive_20260918/P1_s42_40902.log 2>/dev/null || tail -25 $ROOT/runs/predictive_20260918/P1_s42/train.log 2>/dev/null
ls -l $ROOT/runs/predictive_20260918/P1_s42/ 2>/dev/null
if [ -f $ROOT/runs/predictive_20260918/P1_s42/summary.json ]; then
  python3 -c "import json;d=json.load(open('$ROOT/runs/predictive_20260918/P1_s42/summary.json'));print(d)"
fi
ls -l $ROOT/runs/official_test_20260917/window/test_P1* 2>/dev/null
python3 - <<'PY'
import json
from pathlib import Path
W=Path("/data/hyc/U-JEPANet/runs/official_test_20260917/window")
for p in sorted(W.glob("test_P*_window.json")):
    d=json.loads(p.read_text())
    print(p.name, d.get("arm"), d.get("seed"), {a:v.get("per_organ",{}).get("ALL") for a,v in d.get("arms",{}).items()})
PY
