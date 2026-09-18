#!/bin/bash
#SBATCH --job-name=amos-hu
#SBATCH --partition=hpc_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/public/home/heyecheng/U-JEPANet/runs/school/%x-%j.out
#SBATCH --error=/public/home/heyecheng/U-JEPANet/runs/school/%x-%j.err

set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
ZIP='/public/share/td20230405/AMOS 22/amos22.zip'
# Prefer pre-extracted tree; fall back to node scratch if missing
AMOS_ROOT='/public/share/td20230405/AMOS 22/amos22'
WORK=${SLURM_TMPDIR:-/tmp}/amos22_${SLURM_JOB_ID}
OUTDIR=$ROOT/runs/school/amos_hu_${SLURM_JOB_ID}
mkdir -p "$OUTDIR" "$ROOT/runs/school"

echo "host=$(hostname) job=$SLURM_JOB_ID amos_root=$AMOS_ROOT"
"$PY" -c "import SimpleITK,scipy,numpy; print('deps ok')"

if [ -d "$AMOS_ROOT/imagesTr" ] && [ -d "$AMOS_ROOT/labelsTr" ]; then
  echo "Using pre-extracted AMOS at $AMOS_ROOT"
  ROOT_AMOS=$AMOS_ROOT
else
  echo "Pre-extracted tree missing; extracting to $WORK"
  mkdir -p "$WORK"
  unzip -q -o "$ZIP" 'amos22/imagesTr/*' 'amos22/labelsTr/*' 'amos22/dataset.json' 'amos22/readme.md' -d "$WORK"
  ROOT_AMOS=$WORK/amos22
  echo "extract done: $(du -sh "$ROOT_AMOS" | cut -f1)"
fi
ls "$ROOT_AMOS" | head

"$PY" "$ROOT/scripts/amos_organ_hu_stats.py" \
  --root "$ROOT_AMOS" \
  --out "$OUTDIR/amos22_organ_hu_stats.json" \
  --ct-only \
  2>&1 | tee "$OUTDIR/amos22_organ_hu_stats.log"

"$PY" "$ROOT/scripts/amos_organ_hu_stats.py" \
  --root "$ROOT_AMOS" \
  --out "$OUTDIR/amos22_organ_hu_stats_all.json" \
  2>&1 | tee "$OUTDIR/amos22_organ_hu_stats_all.log" || true

cp -f "$OUTDIR"/*.json "$ROOT/runs/school/" 2>/dev/null || true
echo "DONE $OUTDIR"
