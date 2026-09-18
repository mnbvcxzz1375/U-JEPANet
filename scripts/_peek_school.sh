#!/bin/bash
set -euo pipefail
echo "==== school queue ===="
squeue -u heyecheng -o '%i %P %j %T %M %N'
echo "==== school sacct ===="
sacct -u heyecheng --starttime today --format=JobID,JobName%24,State,Elapsed,ExitCode | grep -E 'ujepa|JobID' | tail -20
echo "==== A0DA / rem runs ===="
ls -lt /public/home/heyecheng/U-JEPANet/runs/school/ | head -25
for d in /public/home/heyecheng/U-JEPANet/runs/school/ujepa-*; do
  [ -d "$d" ] || continue
  if [ -f "$d/summary.json" ]; then
    python3 -c "import json;d=json.load(open('$d/summary.json'));print('$d', {k:d.get(k) for k in ['arm','best_val','strength','use_jepa','intensity_mode','seed'] if k in d or True}); print('  best',d.get('best_val'),'final',(d.get('val_curve') or [{}])[-1])"
  else
    # running
    lf="$d/train.log"
    [ -f "$lf" ] || continue
    echo "$d RUNNING-ish last=$(tail -1 $lf)"
  fi
done
echo "==== window JSONs school ===="
W=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/window
for f in "$W"/test_*_window.json "$W"/test_A0_fixed_window.json; do
  [ -f "$f" ] || continue
  python3 -c "import json,sys;d=json.load(open(sys.argv[1]));
print(sys.argv[1].split('/')[-1], d.get('intensity_mode'), {a:v.get('per_organ',{}).get('ALL') for a,v in d.get('arms',{}).items()})" "$f"
done
