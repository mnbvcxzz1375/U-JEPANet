#!/bin/bash
set -euo pipefail
for i in $(seq 1 40); do
  st=$(squeue -j 383257 -h -o '%T %M' || true)
  if [ -z "$st" ]; then echo "poll $i done"; break; fi
  echo "poll $i $st"
  tail -3 /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_383257_c2test.out 2>/dev/null | tr '\n' ' '
  echo
  sleep 15
done
sacct -j 383257 --format=JobID,State,Elapsed,ExitCode
ls -l /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_C2.json 2>/dev/null || echo NO_JSON
tail -40 /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_383257_c2test.out 2>/dev/null
tail -20 /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_383257_c2test.err 2>/dev/null
