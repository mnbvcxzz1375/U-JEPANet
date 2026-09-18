#!/bin/bash
set -euo pipefail
echo "==== queue ===="
squeue -u heyecheng -o '%i %P %j %T %M %N'
echo "==== sacct recent ===="
sacct -u heyecheng --starttime today --format=JobID,JobName%20,State,Elapsed,ExitCode | grep -E 'ujepa|JobID' | tail -25
echo "==== A0DA ===="
D=/public/home/heyecheng/U-JEPANet/runs/school
ls -lt "$D" | head -12
for d in "$D"/ujepa-a0da_*; do
  [ -d "$d" ] || continue
  echo "-- $d --"
  if [ -f "$d/summary.json" ]; then
    python3 -c "import json;d=json.load(open('$d/summary.json'));print({k:d.get(k) for k in ['arm','best_val','steps','strength','use_jepa','intensity_mode']});print('final',(d.get('val_curve') or [{}])[-1])"
  else
    tail -6 "$d/train.log" 2>/dev/null || tail -6 "${d%/}/../$(basename $d | sed 's/_/-/')-*.out" 2>/dev/null || true
  fi
done
# also slurm out
ls -lt "$D"/*a0da*.out 2>/dev/null | head -3
for f in "$D"/*a0da*.out; do
  [ -f "$f" ] || continue
  echo "-- $f --"
  tail -15 "$f"
done
echo "==== window re-eval ===="
W=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/window
ls -lt "$W" 2>/dev/null || echo no_window_dir
for f in "$W"/test_*_window.json; do
  [ -f "$f" ] || continue
  python3 -c "import json,sys;d=json.load(open(sys.argv[1]));
arms=d.get('arms',{});
print(sys.argv[1].split('/')[-1], {a:v.get('per_organ',{}).get('ALL') for a,v in arms.items()}, d.get('intensity_mode'))" "$f"
done
for f in /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_*win*.out; do
  [ -f "$f" ] || continue
  echo "-- $(basename $f) --"
  grep -E 'ALL=|WROTE|DONE|window_ALL|SKIP|Error' "$f" | tail -6
done
