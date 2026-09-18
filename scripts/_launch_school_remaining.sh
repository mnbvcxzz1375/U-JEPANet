#!/bin/bash
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
cp /tmp/dynamic_dataset.py /tmp/whole_volume_eval.py /tmp/paired_views.py $ROOT/ujepa/
cp /tmp/eval_official_test.py /tmp/train_c_ladder.py /tmp/school_sbatch_remaining.sh $ROOT/scripts/
cd $ROOT
# Multi-seed + retrain fleet on school A800
sbatch --export=ALL scripts/school_sbatch_remaining.sh A0DA 43
sbatch --export=ALL scripts/school_sbatch_remaining.sh A0D 43
sbatch --export=ALL scripts/school_sbatch_remaining.sh C3R 43
sbatch --export=ALL scripts/school_sbatch_remaining.sh C2 43
# A0 window eval (fair baseline under train-consistent intensity)
sbatch --export=ALL scripts/school_sbatch_window_reeval.sh A0FIXED 2>/dev/null || true
# dedicated A0 window job via remaining script style
cat > /tmp/school_sbatch_a0_window.sh <<'EOF'
#!/bin/bash
#SBATCH --job-name=ujepa-a0win
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_%j_a0win.out
#SBATCH --error=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_%j_a0win.err
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1
# A0 fixed-crop best.pt must be synced; try school copy or fail clearly
CK=/public/home/heyecheng/U-JEPANet/runs/school/a0_fixed_best.pt
if [ ! -f "$CK" ]; then
  echo "MISSING $CK — upload A0 fixed best.pt first"; exit 3
fi
"$PY" $ROOT/scripts/eval_official_test.py \
  --word-root /public/share/td20230405/WORD \
  --test-ids $ROOT/data/splits_sll20/test_30.txt \
  --arms A0 --arm-ckpt "A0=$CK" \
  --intensity-mode window \
  --out $ROOT/runs/official_test_20260917/window/test_A0_fixed_window.json
echo A0_WINDOW_DONE
EOF
cp /tmp/school_sbatch_a0_window.sh $ROOT/scripts/
sbatch --export=ALL scripts/school_sbatch_a0_window.sh
echo "==== school queue ===="
squeue -u heyecheng -o '%i %P %j %T %N' | head -20
echo LAUNCHED_SCHOOL
