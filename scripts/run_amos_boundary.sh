#!/bin/bash
set -euo pipefail
AMOS='/public/share/td20230405/AMOS 22/amos22'
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
SCRIPT=/public/home/heyecheng/U-JEPANet/scripts/organ_boundary_contrast.py
OUT=/public/home/heyecheng/U-JEPANet/runs/school/amos_boundary_contrast.json
LOG=/public/home/heyecheng/U-JEPANet/runs/school/amos_boundary_contrast.log
echo "imagesTr count: $(ls "$AMOS/imagesTr" 2>/dev/null | wc -l)"
if [ ! -d "$AMOS/imagesTr" ]; then
  echo "AMOS extract not ready at $AMOS" | tee "$LOG"
  exit 1
fi
"$PY" "$SCRIPT" --dataset amos --root "$AMOS" --out "$OUT" 2>&1 | tee "$LOG"
echo DONE
