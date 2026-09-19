#!/bin/bash
# Official test for GLP G0/G1 — GLP-aware sliding window + whole-CT global.
# Usage: bash test_glp_official.sh ARM SEED CKPT_OUT_DIR
set -euo pipefail
ROOT=${ROOT:-/data/hyc/U-JEPANet}
PY=${PY:-/home/ubuntu/anaconda3/envs/vllmenv/bin/python}
WORD=${WORD:-/data/hyc/PLS4MIS/code/datasets/WORD}
SPLIT=${SPLIT:-$ROOT/data/splits_sll20}
ARM=${1:?G0|G1}
SEED=${2:-42}
OUT=${3:-$ROOT/runs/glp_v3/${ARM}_s${SEED}}
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_ALLOC_CONF=expandable_segments:True
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
from ujepa.global_local_predictive import GLPUNet
from ujepa.glp_eval import sliding_window_logits_glp
from ujepa.metrics import dice_per_class
from ujepa.ct_augment import canonical_window
import torch.nn.functional as F

OUT=Path("$OUT")
W=Path("$W")
WORD=Path("$WORD")
SPLIT=Path("$SPLIT")
arm="$ARM"
seed=int("$SEED")
ck=torch.load(OUT/"best.pt", map_location="cpu", weights_only=False)
print("loaded", arm, "seed", seed, "best_val", ck.get("best_val"), "alpha_g", ck.get("alpha_g"))
model=GLPUNet(
    arm=arm,
    feature_chns=(16,32,64,128),
    class_num=17,
    embed_dim=int(ck.get("embed_dim",256)),
    multiscale_pred=True,
)
model.load_state_dict(ck["model"])
model=model.cuda().eval()
need_global = arm in ("G1","G2")
ids=[ln.strip() for ln in (SPLIT/"test_30.txt").read_text().splitlines() if ln.strip()]
acc={c:[] for c in range(1,17)}
per_case=[]
for cid in ids:
    img=sitk.GetArrayFromImage(sitk.ReadImage(str(WORD/"imagesTs"/f"{cid}.nii.gz"))).astype(np.float32)
    lab=sitk.GetArrayFromImage(sitk.ReadImage(str(WORD/"labelsTs"/f"{cid}.nii.gz"))).astype(np.int64)
    vol=canonical_window(torch.from_numpy(img)[None])[None]  # (1,1,D,H,W)
    glob=None
    if need_global:
        g=canonical_window(torch.from_numpy(img).float())
        glob=F.interpolate(g[None,None], size=(64,64,64), mode="trilinear", align_corners=False)
    logits=sliding_window_logits_glp(
        model, vol, (128,128,96), (64,64,48), torch.device("cuda"), 17,
        global_image=glob, full_shape=tuple(img.shape),
    )
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
# low-CNR vs high-CNR (WORD train CNR prior — pre-registered groups)
low=[6,7,8,9,10,12,13,14]  # GB, Eso, Panc, Duo, Colon, Adrenal, Rectum, Bladder
high=[1,2,3,4,15,16]       # Liver, Spleen, Kidneys, Femurs
low_m=float(np.mean([per[c] for c in low if c in per]))
high_m=float(np.mean([per[c] for c in high if c in per]))
outp={
  "arms":{"P":{"per_organ":per}},
  "arm":arm,"seed":seed,"code":"V3.1-GLP",
  "best_val":ck.get("best_val"),
  "alpha_g":ck.get("alpha_g"),
  "intensity_mode":"window",
  "protocol":"official imagesTs + GLP origins + whole-CT global (G1)",
  "low_CNR_mean":low_m,"high_CNR_mean":high_m,
  "per_case":per_case,
}
(W/f"test_{arm}_s{seed}_glp_window.json").write_text(json.dumps(outp, indent=2))
print("TEST_ALL", per["ALL"])
print("LOW_CNR", low_m, "HIGH_CNR", high_m)
print("WROTE", W/f"test_{arm}_s{seed}_glp_window.json")
PY
