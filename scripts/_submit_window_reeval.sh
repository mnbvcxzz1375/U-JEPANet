#!/bin/bash
set -euo pipefail
cp /tmp/school_sbatch_window_reeval.sh /public/home/heyecheng/U-JEPANet/scripts/
cd /public/home/heyecheng/U-JEPANet
sbatch --export=ALL scripts/school_sbatch_window_reeval.sh A0D
sbatch --export=ALL scripts/school_sbatch_window_reeval.sh C1
sbatch --export=ALL scripts/school_sbatch_window_reeval.sh C2
squeue -u heyecheng -o '%i %P %j %T %N' | head -15
