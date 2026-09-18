#!/bin/bash
set -euo pipefail
for i in $(seq 1 30); do
  if squeue -j 382537 -h | grep -q .; then
    echo "poll $i still running $(squeue -j 382537 -h -o '%T %M')"
    sleep 15
  else
    echo "poll $i done"
    break
  fi
done
sacct -j 382537 --format=JobID,State,Elapsed,ExitCode
D=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917
ls -l "$D"/test_A2-LU.json 2>/dev/null || echo MISSING_JSON
tail -25 "$D"/slurm_382537_ujepa-test.out
python3 - <<'PY'
import json
from pathlib import Path
d = json.load(open("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_A2-LU.json"))
print("A2-LU ALL", d["arms"]["A2-LU"]["per_organ"]["ALL"])
print(json.dumps(d["arms"]["A2-LU"]["per_organ"], indent=2))
PY
