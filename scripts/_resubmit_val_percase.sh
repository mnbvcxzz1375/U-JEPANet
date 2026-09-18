#!/bin/bash
set -euo pipefail
scancel -n school_sbatch_val_percase.sh 2>/dev/null || true
# also cancel by job ids if name cancel fails
scancel 382790 382791 382792 382793 382794 382795 2>/dev/null || true
cd /public/home/heyecheng/U-JEPANet
mkdir -p runs/val_percase_20260917
for a in A0 A1 A2 A3 A2-L A2-LU; do
  sbatch --export=ALL scripts/school_sbatch_val_percase.sh "$a"
done
squeue -u heyecheng -o '%i %P %j %T %N' | head -15
