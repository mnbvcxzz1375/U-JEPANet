#!/bin/bash
set -euo pipefail
PY=/home/ubuntu/anaconda3/envs/vllmenv/bin/python
SCRIPT=/data/hyc/U-JEPANet/scripts/organ_boundary_contrast.py
WORD=/data/hyc/PLS4MIS/code/datasets/WORD
OUT=/data/hyc/U-JEPANet/runs/word_boundary_contrast.json
LOG=/data/hyc/U-JEPANet/runs/word_boundary_contrast.log
export PYTHONPATH=/data/hyc/U-JEPANet
"$PY" "$SCRIPT" --dataset word --root "$WORD" --out "$OUT" 2>&1 | tee "$LOG"
echo DONE
