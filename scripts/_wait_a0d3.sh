#!/bin/bash
set -euo pipefail
for i in $(seq 1 40); do
  st=$(squeue -j 383493 -h -o '%T %M' || true)
  if [ -z "$st" ]; then echo "poll $i done"; break; fi
  echo "poll $i $st"
  tail -1 /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_383493_a0dtest.out 2>/dev/null | tr '\n' ' '; echo
  sleep 12
done
sacct -j 383493 --format=JobID,State,Elapsed,ExitCode
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_A0D.json")
if not p.exists():
    print("MISSING"); raise SystemExit
d=json.loads(p.read_text())
po=d["arms"]["A0"]["per_organ"]
print("A0D ALL", po["ALL"])
print("A0D-A0", po["ALL"]-0.7674341375449829)
print("C1-A0D", 0.8118238625540593-po["ALL"])
print("C2-A0D", 0.8169421073148923-po["ALL"])
print("C3-A0D", 0.8203055778585197-po["ALL"])
print("C4-A0D", 0.8168372743076378-po["ALL"])
PY
