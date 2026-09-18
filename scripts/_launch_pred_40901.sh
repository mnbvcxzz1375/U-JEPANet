#!/bin/bash
set -euo pipefail
ROOT=/data/hyc/U-JEPANet
cp /tmp/predictive_unet.py $ROOT/ujepa/
cp /tmp/train_predictive.py /tmp/run_predictive_40901.sh $ROOT/scripts/
chmod +x $ROOT/scripts/run_predictive_40901.sh
mkdir -p $ROOT/runs/predictive_20260918
nohup bash $ROOT/scripts/run_predictive_40901.sh 0 P1 42 > $ROOT/runs/predictive_20260918/P1_s42.log 2>&1 &
nohup bash $ROOT/scripts/run_predictive_40901.sh 1 P2 42 > $ROOT/runs/predictive_20260918/P2_s42.log 2>&1 &
sleep 4
ps aux | grep train_predictive | grep -v grep
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo LAUNCHED_40901_P1_P2
