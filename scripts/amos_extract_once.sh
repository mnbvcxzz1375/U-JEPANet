#!/bin/bash
set -euo pipefail
ZIP='/public/share/td20230405/AMOS 22/amos22.zip'
DEST='/public/share/td20230405/AMOS 22/amos22'
LOG='/public/home/heyecheng/U-JEPANet/runs/school/amos_extract.log'
mkdir -p "$DEST"
echo "extract start $(date -Is) dest=$DEST" | tee -a "$LOG"
# Extract full amos22 tree once (imagesTr/labelsTr/imagesVa/labelsVa/imagesTs/labelsTs/meta)
unzip -n -q "$ZIP" -d '/public/share/td20230405/AMOS 22' 'amos22/*' '__MACOSX/*' || true
# Prefer moving/keeping under AMOS 22/amos22 (zip already extracts to amos22/)
if [ -d '/public/share/td20230405/AMOS 22/amos22/imagesTr' ]; then
  echo "already at DEST" | tee -a "$LOG"
else
  mkdir -p "$DEST"
fi
du -sh "$DEST" | tee -a "$LOG"
ls "$DEST" | tee -a "$LOG"
echo "extract done $(date -Is)" | tee -a "$LOG"
