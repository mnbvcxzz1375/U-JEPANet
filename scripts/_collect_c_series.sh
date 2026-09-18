#!/bin/bash
set -euo pipefail
echo "==== queue ===="
squeue -u heyecheng -o '%i %P %j %T %M %N'
echo "==== C2 382006 ===="
sacct -j 382006 --format=JobID,State,Elapsed,ExitCode,NodeList
echo "==== A0D 383177 ===="
sacct -j 383177 --format=JobID,State,Elapsed,ExitCode,NodeList
echo "==== school runs ===="
ls -lt /public/home/heyecheng/U-JEPANet/runs/school/
echo "==== summaries ===="
for d in /public/home/heyecheng/U-JEPANet/runs/school/ujepa-c1_381886 \
         /public/home/heyecheng/U-JEPANet/runs/school/ujepa-c1_382006 \
         /public/home/heyecheng/U-JEPANet/runs/school/ujepa-a0d_*; do
  [ -d "$d" ] || continue
  echo "-- $d --"
  ls -lh "$d"
  if [ -f "$d/summary.json" ]; then
    python3 -c "import json; d=json.load(open('$d/summary.json')); print({k:d.get(k) for k in ['arm','best_val','steps','strength','jepa_mode','use_jepa']}); print('final', (d.get('val_curve') or [{}])[-1])"
  fi
done
