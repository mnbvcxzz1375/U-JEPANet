#!/bin/bash
set -euo pipefail
ROOT=/data/hyc/U-JEPANet
cp /tmp/dynamic_dataset.py /tmp/whole_volume_eval.py /tmp/paired_views.py $ROOT/ujepa/
cp /tmp/eval_official_test.py /tmp/train_c_ladder.py /tmp/run_remaining_40901.sh $ROOT/scripts/
chmod +x $ROOT/scripts/run_remaining_40901.sh
mkdir -p $ROOT/runs/remaining_20260917 $ROOT/runs/official_test_20260917/window
# GPU0: paired C4; GPU1: C3 retrain (fixed z-sampler)
nohup bash $ROOT/scripts/run_remaining_40901.sh 0 C4P 42 > $ROOT/runs/remaining_20260917/C4P_s42.log 2>&1 &
nohup bash $ROOT/scripts/run_remaining_40901.sh 1 C3R 42 > $ROOT/runs/remaining_20260917/C3R_s42.log 2>&1 &
# quick A0 window eval on CPU-safe? needs GPU - queue after? use remaining GPU wait
# A0 window on GPU0 would contend; run A0 window via python after short delay on GPU1 is busy.
# Instead evaluate A0 window on school 4090d.
sleep 3
ps aux | grep run_remaining | grep -v grep
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo LAUNCHED_40901
