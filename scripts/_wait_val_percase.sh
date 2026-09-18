#!/bin/bash
set -euo pipefail
for i in $(seq 1 40); do
  s=$(squeue -j 382797,382798,382799,382800,382801,382802 -h -o '%i:%T' | tr '\n' ' ')
  echo "poll $i $s"
  if ! squeue -j 382797,382798,382799,382800,382801,382802 -h | grep -q .; then
    echo ALL_DONE
    break
  fi
  sleep 15
done
sacct -j 382797,382798,382799,382800,382801,382802 --format=JobID,State,Elapsed,ExitCode,NodeList
D=/public/home/heyecheng/U-JEPANet/runs/val_percase_20260917
ls -lt "$D" 2>/dev/null | head -20
python3 - <<'PY'
from pathlib import Path
import json
D=Path("/public/home/heyecheng/U-JEPANet/runs/val_percase_20260917")
for p in sorted(D.glob("val_*.json")):
    d=json.loads(p.read_text())
    for a,v in d.get("arms",{}).items():
        print(a, "ALL", v["per_organ"]["ALL"], "n", v.get("n"))
PY
