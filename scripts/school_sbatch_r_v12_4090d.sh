#!/bin/bash
#SBATCH --job-name=ujepa-r12
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/public/home/heyecheng/U-JEPANet/runs/school/%x-%j.out
#SBATCH --error=/public/home/heyecheng/U-JEPANet/runs/school/%x-%j.err
# Moved from hpc_gpu A800 (too slow) to 4090d
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1
WORD=/public/share/td20230405/WORD
ARM=${1:?R1|R2}
SEED=${2:-43}
OUT=$ROOT/runs/school/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${ARM}_s${SEED}_4090d
CACHE=${SLURM_TMPDIR:-/tmp}/ujepa_hu_${ARM}_${SEED}_4090d
mkdir -p "$OUT" "$CACHE"
echo "host=$(hostname) partition=gpu_4090 arm=$ARM seed=$SEED code=V1.2"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PY" "$ROOT/scripts/train_predictive_v11.py" \
  --arm "$ARM" \
  --word-root "$WORD" \
  --split-dir "$ROOT/data/splits_sll20" \
  --cache-dir "$CACHE" \
  --out "$OUT" \
  --steps 30000 --batch 2 --device cuda --seed "$SEED" \
  --intensity-mode window --target-tokens 384 \
  2>&1 | tee "$OUT/train.log"
echo "TRAIN_DONE $OUT"
W=$ROOT/runs/official_test_20260917/window
mkdir -p "$W"
"$PY" - <<PY
import sys
sys.path.insert(0, "$ROOT")
import json
from pathlib import Path
import torch
import numpy as np
import SimpleITK as sitk
from ujepa.predictive_unet import PredictiveUNet
from ujepa.metrics import dice_per_class
from ujepa.whole_volume_eval import sliding_window_logits
from ujepa.ct_augment import canonical_window
ck=torch.load("$OUT/best.pt", map_location="cpu", weights_only=False)
model=PredictiveUNet(
    feature_chns=(16,32,64,128), class_num=17, deep_stage=2,
    embed_dim=int(ck.get("embed_dim",256)),
    mask_ratio=float(ck.get("mask_ratio",0.4)),
    use_residual=bool(ck.get("use_residual", True)),
    target_tokens=int(ck.get("target_tokens", 384)),
    multiscale_pred=True,
)
model.load_state_dict(ck["model"])
model=model.cuda().eval()
ids=[ln.strip() for ln in Path("$ROOT/data/splits_sll20/test_30.txt").read_text().splitlines() if ln.strip()]
root=Path("$WORD")
acc={c:[] for c in range(1,17)}
for cid in ids:
    img=sitk.GetArrayFromImage(sitk.ReadImage(str(root/"imagesTs"/f"{cid}.nii.gz"))).astype(np.float32)
    lab=sitk.GetArrayFromImage(sitk.ReadImage(str(root/"labelsTs"/f"{cid}.nii.gz"))).astype(np.int64)
    vol=canonical_window(torch.from_numpy(img)[None,None])
    logits=sliding_window_logits(model, vol, (128,128,96), (64,64,48), torch.device("cuda"), 17)
    pred=torch.argmax(logits,dim=1)[0].cpu()
    pc=dice_per_class(pred, torch.from_numpy(lab), 17)
    for c,v in pc.items():
        if v==v: acc[c].append(v)
    print(cid, float(np.mean([v for v in pc.values() if v==v])), flush=True)
per={c:(float(np.mean(vs)) if vs else float("nan")) for c,vs in acc.items()}
vals=[v for v in per.values() if v==v]
per["ALL"]=float(np.mean(vals)) if vals else float("nan")
Path("$W/test_${ARM}_s${SEED}_v12_window.json").write_text(json.dumps({
  "arms":{"P":{"per_organ":per}},
  "arm":"$ARM","seed":$SEED,"code":"V1.2-corrected",
  "intensity_mode":"window","host":"school-4090d",
  "note":"full-context seg; F3=E3(F2*); masked-only aux",
}, indent=2))
print("TEST_ALL", per["ALL"])
print("VAL_BEST", ck.get("best_val"))
PY
echo "TEST_DONE $ARM s$SEED"
