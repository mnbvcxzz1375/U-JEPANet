#!/bin/bash
set -euo pipefail
cd /public/home/heyecheng/U-JEPANet
mkdir -p runs/official_test_20260917
for a in A1 A3 A2-L A2-LU; do
  sbatch --export=ALL scripts/school_sbatch_test_eval.sh "$a"
done
squeue -u heyecheng -o '%i %P %j %T %N %M'
