#!/bin/bash
set -euo pipefail
cd /public/home/heyecheng/ujepa_test_ckpts
mv -n ujepa_test_ckpts/*.pt .
ls -lh *.pt
cd /public/home/heyecheng/U-JEPANet
rm -f runs/official_test_20260917/test_*.json
for a in A1 A3 A2-L A2-LU; do
  sbatch --export=ALL scripts/school_sbatch_test_eval.sh "$a"
done
squeue -u heyecheng -o '%i %P %j %T %N'
