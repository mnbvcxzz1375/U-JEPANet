#!/bin/bash
set -euo pipefail
ROOT=/data/hyc/U-JEPANet
echo "==== C3 ===="
ls -lh $ROOT/runs/c_ladder_40901/C3/
tail -15 $ROOT/runs/c3_gpu0.log 2>/dev/null || tail -15 $ROOT/runs/c_ladder_40901/C3/train.log
if [ -f $ROOT/runs/c_ladder_40901/C3/summary.json ]; then
  python3 -c "import json; d=json.load(open('$ROOT/runs/c_ladder_40901/C3/summary.json')); print(d)"
fi
echo "==== C4 ===="
ls -lh $ROOT/runs/c_ladder_40901/C4/
tail -15 $ROOT/runs/c4_gpu1.log 2>/dev/null || tail -15 $ROOT/runs/c_ladder_40901/C4/train.log
if [ -f $ROOT/runs/c_ladder_40901/C4/summary.json ]; then
  python3 -c "import json; d=json.load(open('$ROOT/runs/c_ladder_40901/C4/summary.json')); print(d)"
fi
echo "==== procs ===="
ps aux | grep -E 'train_c_ladder|eval_official' | grep -v grep || true
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
