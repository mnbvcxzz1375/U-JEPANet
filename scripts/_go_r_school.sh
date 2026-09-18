#!/bin/bash
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
cp /tmp/predictive_unet.py $ROOT/ujepa/
cp /tmp/train_predictive_v11.py /tmp/school_sbatch_r_v12.sh $ROOT/scripts/
cd $ROOT
sbatch --export=ALL scripts/school_sbatch_r_v12.sh R1 43
sbatch --export=ALL scripts/school_sbatch_r_v12.sh R2 43
squeue -u heyecheng -o '%i %P %j %T %N' | head -12
echo LAUNCHED_SCHOOL_R12_s43
