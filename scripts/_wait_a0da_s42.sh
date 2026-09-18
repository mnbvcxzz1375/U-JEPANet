#!/bin/bash
set -euo pipefail
for i in $(seq 1 25); do
  if squeue -j 384544 -h | grep -q .; then
    echo "poll $i $(squeue -j 384544 -h -o '%T %M')"
    sleep 12
  else
    echo "poll $i done"; break
  fi
done
sacct -j 384544 --format=JobID,State,Elapsed,ExitCode
tail -8 /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_384544_a0das42.out 2>/dev/null
python3 - <<'PY'
import json
from pathlib import Path
W=Path("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/window")
# collect all
data={}
for p in W.glob("test_*.json"):
    d=json.loads(p.read_text())
    for a,v in d["arms"].items():
        data[p.name]=v["per_organ"]["ALL"]
print("file ALL")
for k in sorted(data):
    print(f"  {k:42s} {data[k]:.4f}")
# key contrasts if A0DA s42 present
p=W/"test_A0DA_s42_window.json"
if p.exists():
    a0da=json.loads(p.read_text())["arms"]["A0"]["per_organ"]["ALL"]
    print("A0DA_s42", a0da)
    for name,val in [("C2_s42",0.8196868260303413),("C3R_s42",0.8204921481581497),("C4P_s42",0.8224178489194376)]:
        print(f"  {name} - A0DA_s42 = {val-a0da:+.4f}")
PY
