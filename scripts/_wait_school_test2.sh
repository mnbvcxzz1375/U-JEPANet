#!/bin/bash
set -euo pipefail
JOBS=382534,382535,382536,382537
for i in $(seq 1 45); do
  s=$(squeue -j $JOBS -h -o '%i:%T' | tr '\n' ' ')
  echo "poll $i $s"
  if ! squeue -j $JOBS -h | grep -q .; then
    echo ALL_JOBS_GONE
    break
  fi
  sleep 20
done
echo "==== sacct ===="
sacct -j $JOBS --format=JobID,JobName%16,State,Elapsed,ExitCode,NodeList
echo "==== results ===="
D=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917
ls -lt "$D"
for f in "$D"/test_A1.json "$D"/test_A3.json "$D"/test_A2-L.json "$D"/test_A2-LU.json; do
  if [ -f "$f" ]; then
    python3 -c "import json,sys; d=json.load(open(sys.argv[1]));
arms=d.get('arms',{});
print(sys.argv[1]);
[print(' ',a,v.get('per_organ',{}).get('ALL'),'n',v.get('n_test')) for a,v in arms.items()];
" "$f"
  else
    echo "MISSING $f"
  fi
done
echo "==== slurm status lines ===="
for f in "$D"/slurm_38253*.out; do
  [ -f "$f" ] || continue
  echo "-- $f --"
  grep -E 'host=|SKIP|ALL=|WROTE|DONE|Error|Traceback|ALL ' "$f" | tail -20
done
