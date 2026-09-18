#!/bin/bash
set -euo pipefail
HOST=$1  # 40901|40902
ARM=$2
GPU=0
ROOT=/data/hyc/U-JEPANet
cp /tmp/predictive_unet.py $ROOT/ujepa/
cp /tmp/train_predictive_v11.py /tmp/run_r_v12.sh $ROOT/scripts/
chmod +x $ROOT/scripts/run_r_v12.sh
mkdir -p $ROOT/runs/predictive_v12
nohup bash $ROOT/scripts/run_r_v12.sh $GPU "$ARM" 42 > $ROOT/runs/predictive_v12/${HOST}_${ARM}_s42.log 2>&1 &
sleep 3
ps aux | grep train_predictive_v11 | grep -v grep
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
tail -6 $ROOT/runs/predictive_v12/${HOST}_${ARM}_s42.log
echo LAUNCHED $HOST $ARM s42
