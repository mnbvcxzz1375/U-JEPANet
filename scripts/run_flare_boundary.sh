#!/bin/bash
set -euo pipefail
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
SCRIPT=/public/home/heyecheng/U-JEPANet/scripts/organ_boundary_contrast.py
ROOT='/public/share/td20230405/FLARE2023'
OUT=/public/home/heyecheng/U-JEPANet/runs/school/flare_boundary_contrast.json
LOG=/public/home/heyecheng/U-JEPANet/runs/school/flare_boundary_contrast.log
echo "imagesTr subs:"; find "$ROOT/imagesTr" -name '*_0000.nii.gz' 2>/dev/null | wc -l
echo "labelsTr:"; find "$ROOT/labelsTr" -name '*.nii.gz' 2>/dev/null | wc -l
"$PY" "$SCRIPT" --dataset flare --root "$ROOT" --out "$OUT" 2>&1 | tee "$LOG"
echo DONE
