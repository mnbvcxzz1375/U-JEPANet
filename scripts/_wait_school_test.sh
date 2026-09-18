#!/bin/bash
set -euo pipefail
for i in $(seq 1 40); do
  s=$(squeue -j 382524,382525,382526,382527 -h -o '%i:%T' | tr '\n' ' ')
  echo "poll $i $s"
  if ! squeue -j 382524,382525,382526,382527 -h | grep -q .; then
    echo ALL_JOBS_GONE
    break
  fi
  sleep 20
done
echo "==== sacct ===="
sacct -j 382524,382525,382526,382527 --format=JobID,JobName%20,State,Elapsed,ExitCode,NodeList
echo "==== outputs ===="
ls -lt /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/ | head -30
echo "==== json all ===="
for f in /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_*.json; do
  [ -f "$f" ] || continue
  python3 - <<PY
import json
p="$f"
d=json.load(open(p))
for a,v in d.get("arms",{}).items():
    print(a, v["per_organ"]["ALL"], p)
PY
done
echo "==== slurm tails ===="
for f in /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_*.out; do
  [ -f "$f" ] || continue
  echo "-- $f --"
  tail -15 "$f"
done
for f in /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_*.err; do
  [ -f "$f" ] || continue
  if grep -q 'Error\|Traceback\|OOM\|OutOfMemory' "$f"; then
    echo "-- ERR $f --"
    tail -20 "$f"
  fi
done
