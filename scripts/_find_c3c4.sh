#!/bin/bash
set -euo pipefail
echo "==== queue ===="
squeue -u heyecheng -o '%i %P %j %T %M %N'
echo "==== recent sacct ujepa ===="
sacct -u heyecheng --starttime today --format=JobID,JobName%20,State,Elapsed,ExitCode,NodeList | grep -E 'ujepa|JobID' | tail -40
echo "==== school runs ===="
ls -lt /public/home/heyecheng/U-JEPANet/runs/school/
echo "==== all c/a0d dirs ===="
find /public/home/heyecheng/U-JEPANet/runs -maxdepth 2 -type d -name 'ujepa-*' | sort
echo "==== summaries ===="
for d in /public/home/heyecheng/U-JEPANet/runs/school/ujepa-*; do
  [ -d "$d" ] || continue
  if [ -f "$d/summary.json" ]; then
    python3 -c "import json; d=json.load(open('$d/summary.json')); print('$d', {k:d.get(k) for k in ['arm','best_val','steps','strength','jepa_mode','use_jepa']})"
  else
    echo "$d NO_SUMMARY files=$(ls $d 2>/dev/null | tr '\n' ' ')"
  fi
done
echo "==== a0d log tail ===="
tail -6 /public/home/heyecheng/U-JEPANet/runs/school/ujepa-a0d-383177.out 2>/dev/null || true
