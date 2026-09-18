#!/bin/bash
set -euo pipefail
echo "==== queue ===="
squeue -j 382534,382535,382536,382537 -o '%i %P %j %T %N %M' || true
echo "==== sacct ===="
sacct -j 382534,382535,382536,382537 --format=JobID,JobName%16,State,Elapsed,ExitCode,NodeList || true
echo "==== dir ===="
ls -lt /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/ || true
echo "==== json ===="
for f in /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_A1.json \
         /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_A3.json \
         /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_A2-L.json \
         /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_A2-LU.json; do
  if [ -f "$f" ]; then
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1]);
[print(' ',a,v.get('per_organ',{}).get('ALL'),'n',v.get('n_test')) for a,v in d.get('arms',{}).items()]" "$f"
  else
    echo "MISSING $f"
  fi
done
echo "==== slurm ===="
for f in /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_38253*.out; do
  [ -f "$f" ] || continue
  echo "-- $f --"
  tail -20 "$f"
done
