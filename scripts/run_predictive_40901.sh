#!/bin/bash
# V1 predictive arms on 40901.
# Usage: bash run_predictive_40901.sh GPU ARM SEED
set -euo pipefail
GPU=${1:?gpu}
ARM=${2:?arm P1|P2|P3}
SEED=${3:-42}
ROOT=/data/hyc/U-JEPANet
PY=/home/ubuntu/anaconda3/envs/vllmenv/bin/python
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_ALLOC_CONF=expandable_segments:True
WORD=/data/hyc/PLS4MIS/code/datasets/WORD
SPLIT=$ROOT/data/splits_sll20
OUT=$ROOT/runs/predictive_20260918/${ARM}_s${SEED}
CACHE=/tmp/ujepa_hu_${ARM}_${SEED}_$$
mkdir -p "$OUT" "$CACHE"
echo "host=$(hostname) arm=$ARM seed=$SEED gpu=$GPU"
"$PY" "$ROOT/scripts/train_predictive.py" \
  --arm "$ARM" \
  --word-root "$WORD" \
  --split-dir "$SPLIT" \
  --cache-dir "$CACHE" \
  --out "$OUT" \
  --steps 30000 \
  --batch 2 \
  --device cuda \
  --seed "$SEED" \
  --intensity-mode window \
  2>&1 | tee "$OUT/train.log"
echo "TRAIN_DONE $OUT"
W=$ROOT/runs/official_test_20260917/window
mkdir -p "$W"
"$PY" - <<PY
import sys
sys.path.insert(0, "$ROOT")
import torch
from pathlib import Path
from ujepa.predictive_unet import PredictiveUNet
from ujepa.metrics import dice_per_class
from ujepa.whole_volume_eval import sliding_window_logits
from ujepa.ct_augment import canonical_window
import SimpleITK as sitk
import numpy as np
import json
ck=torch.load("$OUT/best.pt", map_location="cpu", weights_only=False)
model=PredictiveUNet(
    feature_chns=[16,32,64,128], class_num=17, deep_stage=2,
    embed_dim=ck.get("embed_dim",256), mask_ratio=ck.get("mask_ratio",0.4),
    use_residual=ck.get("use_residual",True), use_pred_in_forward=True,
).cuda().eval()
model.load_state_dict(ck["model"])
ids=[ln.strip() for ln in Path("$SPLIT/test_30.txt").read_text().splitlines() if ln.strip()]
root=Path("$WORD")
acc={c:[] for c in range(1,17)}
per_case=[]
for cid in ids:
    img=sitk.GetArrayFromImage(sitk.ReadImage(str(root/"imagesTs"/f"{cid}.nii.gz"))).astype(np.float32)
    lab=sitk.GetArrayFromImage(sitk.ReadImage(str(root/"labelsTs"/f"{cid}.nii.gz"))).astype(np.int64)
    vol=canonical_window(torch.from_numpy(img)[None,None])
    logits=sliding_window_logits(model, vol, (128,128,96), (64,64,48), torch.device("cuda"), 17)
    pred=torch.argmax(logits,dim=1)[0].cpu()
    pc=dice_per_class(pred, torch.from_numpy(lab), 17)
    for c,v in pc.items():
        if v==v: acc[c].append(v)
    per_case.append({"case":cid,"mean_fg_dice":float(np.mean([v for v in pc.values() if v==v]))})
    print(cid, per_case[-1]["mean_fg_dice"], flush=True)
per={c: float(np.mean(vs)) if vs else float("nan") for c,vs in acc.items()}
vals=[v for v in per.values() if v==v]
per["ALL"]=float(np.mean(vals)) if vals else float("nan")
out={"arm":"$ARM","seed":$SEED,"intensity_mode":"window","test_ALL":per["ALL"],"per_organ":per}
Path("$W/test_${ARM}_s${SEED}_window.json").write_text(json.dumps({"arms":{"P":{"per_organ":per}},"intensity_mode":"window"}, indent=2))
print("TEST_ALL", per["ALL"])
PY
echo "TEST_DONE $ARM"
