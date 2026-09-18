#!/bin/bash
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
cd $ROOT
# P1 s42 moved from 40901 GPU0 to school (40901 keeps only P2 on one card)
sbatch --export=ALL scripts/school_sbatch_predictive.sh P1 42
squeue -u heyecheng -o '%i %P %j %T %M %N' | head -18
