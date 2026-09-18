#!/bin/bash
set -euo pipefail
echo "==== queue ===="
squeue -u heyecheng -o '%i %P %j %T %M %N' | head -20
echo "==== sacct pred ===="
sacct -u heyecheng --starttime today --format=JobID,JobName%20,State,Elapsed,ExitCode | grep -E 'pred|JobID' | tail -15
echo "==== summaries ===="
python3 - <<'PY'
import json
from pathlib import Path
S=Path("/public/home/heyecheng/U-JEPANet/runs/school")
for p in sorted(S.glob("ujepa-pred_*/summary.json")):
    d=json.loads(p.read_text())
    print(p.parent.name, "arm", d.get("arm"), "best_val", d.get("best_val"), "lambda_p", d.get("lambda_p"), "res", d.get("use_residual"))
W=Path("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/window")
print("==== school window JSONs ====")
for p in sorted(W.glob("test_P*.json")):
    d=json.loads(p.read_text())
    print(p.name, d.get("arm"), d.get("seed"), {a:v.get("per_organ",{}).get("ALL") for a,v in d.get("arms",{}).items()})
# also A0DA
for name in ["test_A0DA_s42_window.json","test_A0DA_s43_window.json"]:
    p=W/name
    if p.exists():
        d=json.loads(p.read_text())
        print(name, {a:v["per_organ"]["ALL"] for a,v in d["arms"].items()})
PY
echo "==== pred logs tail ===="
for f in /public/home/heyecheng/U-JEPANet/runs/school/ujepa-pred-*.out; do
  [ -f "$f" ] || continue
  echo "-- $(basename $f) --"
  grep -E 'TRAIN_DONE|TEST_DONE|TEST_ALL|best_val|arm=' "$f" | tail -8
done
