#!/bin/bash
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
WORD=/public/share/td20230405/WORD
SPLIT=$ROOT/data/splits_sll20
OUT=$ROOT/runs/school/ujepa-r12_385928_R2_s43
W=$ROOT/runs/official_test_20260917/window
"$PY" - <<'PY'
import sys
sys.path.insert(0, "/public/home/heyecheng/U-JEPANet")
import json
from pathlib import Path
import torch
import numpy as np
import SimpleITK as sitk
from ujepa.predictive_unet import PredictiveUNet
from ujepa.metrics import dice_per_class
from ujepa.whole_volume_eval import sliding_window_logits
from ujepa.ct_augment import canonical_window
OUT=Path("/public/home/heyecheng/U-JEPANet/runs/school/ujepa-r12_385928_R2_s43")
W=Path("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/window")
WORD=Path("/public/share/td20230405/WORD")
SPLIT=Path("/public/home/heyecheng/U-JEPANet/data/splits_sll20")
ck=torch.load(OUT/"best.pt", map_location="cpu", weights_only=False)
print("best_val", ck.get("best_val"))
model=PredictiveUNet(
    feature_chns=(16,32,64,128), class_num=17, deep_stage=2,
    embed_dim=int(ck.get("embed_dim",256)),
    mask_ratio=float(ck.get("mask_ratio",0.4)),
    use_residual=bool(ck.get("use_residual", True)),
    target_tokens=int(ck.get("target_tokens",384)),
    multiscale_pred=True,
)
model.load_state_dict(ck["model"])
model=model.cuda().eval()
ids=[ln.strip() for ln in (SPLIT/"test_30.txt").read_text().splitlines() if ln.strip()]
acc={c:[] for c in range(1,17)}
per_case=[]
for cid in ids:
    img=sitk.GetArrayFromImage(sitk.ReadImage(str(WORD/"imagesTs"/f"{cid}.nii.gz"))).astype(np.float32)
    lab=sitk.GetArrayFromImage(sitk.ReadImage(str(WORD/"labelsTs"/f"{cid}.nii.gz"))).astype(np.int64)
    vol=canonical_window(torch.from_numpy(img)[None,None])
    logits=sliding_window_logits(model, vol, (128,128,96), (64,64,48), torch.device("cuda"), 17)
    pred=torch.argmax(logits,dim=1)[0].cpu()
    pc=dice_per_class(pred, torch.from_numpy(lab), 17)
    for c,v in pc.items():
        if v==v: acc[c].append(v)
    mean=float(np.mean([v for v in pc.values() if v==v]))
    per_case.append({"case":cid,"mean_fg_dice":mean})
    print(cid, mean, flush=True)
per={c:(float(np.mean(vs)) if vs else float("nan")) for c,vs in acc.items()}
vals=[v for v in per.values() if v==v]
per["ALL"]=float(np.mean(vals)) if vals else float("nan")
(W/"test_R2_s43_v12_window.json").write_text(json.dumps({
  "arms":{"P":{"per_organ":per}},
  "arm":"R2","seed":43,"code":"V1.2-corrected",
  "intensity_mode":"window","best_val":ck.get("best_val"),
  "per_case": per_case,
}, indent=2))
print("TEST_ALL", per["ALL"])
PY
