#!/bin/bash
set -euo pipefail
cd /public/home/heyecheng/U-JEPANet
mkdir -p runs/val_percase_20260917
for a in A0 A1 A2 A3 A2-L A2-LU; do
  sbatch --export=ALL scripts/school_sbatch_val_percase.sh "$a"
done
squeue -u heyecheng -o '%i %P %j %T %N' | head -20
