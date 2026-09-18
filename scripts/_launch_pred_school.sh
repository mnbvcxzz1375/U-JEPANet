#!/bin/bash
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
cp /tmp/predictive_unet.py $ROOT/ujepa/
cp /tmp/train_predictive.py /tmp/school_sbatch_predictive.sh $ROOT/scripts/
cd $ROOT
sbatch --export=ALL scripts/school_sbatch_predictive.sh P3 42
sbatch --export=ALL scripts/school_sbatch_predictive.sh P2 43
sbatch --export=ALL scripts/school_sbatch_predictive.sh P1 43
squeue -u heyecheng -o '%i %P %j %T %N' | head -15
echo LAUNCHED_SCHOOL_P
