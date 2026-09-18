#!/bin/bash
set -euo pipefail
cd /public/home/heyecheng/U-JEPANet
mkdir -p runs/official_test_20260917
# Submit all remaining arms at once. Skip if result JSON already exists.
for a in A1 A2 A3 A2-L A2-LU; do
  out="runs/official_test_20260917/test_${a}.json"
  if [ -f "$out" ]; then
    echo "SKIP $a exists $out"
    continue
  fi
  # A2 best.pt from 40901 may not be on school; ensure mapping
  sbatch --export=ALL scripts/school_sbatch_test_eval.sh "$a"
done
echo "==== queue ===="
squeue -u heyecheng -o '%i %P %j %T %N %M' | head -30
