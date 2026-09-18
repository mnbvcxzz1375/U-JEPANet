#!/bin/bash
set -euo pipefail
ROOT=/data/hyc/U-JEPANet
if [ ! -d "$ROOT/ujepa" ]; then
  echo "Need to deploy U-JEPANet on 40902"
  mkdir -p $ROOT
  # try rsync from 40901
  rsync -az -e 'ssh -p 40901 -o StrictHostKeyChecking=no' ubuntu@10.126.25.5:/data/hyc/U-JEPANet/ujepa $ROOT/ || true
  rsync -az -e 'ssh -p 40901 -o StrictHostKeyChecking=no' ubuntu@10.126.25.5:/data/hyc/U-JEPANet/scripts $ROOT/ || true
  rsync -az -e 'ssh -p 40901 -o StrictHostKeyChecking=no' ubuntu@10.126.25.5:/data/hyc/U-JEPANet/train.py $ROOT/ || true
fi
mkdir -p $ROOT/ujepa $ROOT/scripts $ROOT/runs/predictive_20260918 $ROOT/data
cp /tmp/predictive_unet.py $ROOT/ujepa/
cp /tmp/train_predictive.py /tmp/run_predictive_40901.sh $ROOT/scripts/
chmod +x $ROOT/scripts/run_predictive_40901.sh
# splits / word from 40901 if missing
if [ ! -f $ROOT/data/splits_sll20/test_30.txt ]; then
  mkdir -p $ROOT/data/splits_sll20
  rsync -az -e 'ssh -p 40901 -o StrictHostKeyChecking=no' ubuntu@10.126.25.5:/data/hyc/U-JEPANet/data/splits_sll20/ $ROOT/data/splits_sll20/ || true
fi
ls $ROOT/ujepa/predictive_unet.py $ROOT/scripts/train_predictive.py $ROOT/data/splits_sll20/labeled_train_20.txt
# ONE GPU only (user constraint): GPU0
export PYTHONPATH=$ROOT
nohup bash $ROOT/scripts/run_predictive_40901.sh 0 P1 42 > $ROOT/runs/predictive_20260918/P1_s42_40902.log 2>&1 &
sleep 6
ps aux | grep train_predictive | grep -v grep
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
tail -8 $ROOT/runs/predictive_20260918/P1_s42_40902.log
echo LAUNCHED_40902_P1_GPU0
