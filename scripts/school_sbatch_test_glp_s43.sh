#!/bin/bash
#SBATCH --job-name=ujepa-glptest
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/public/home/heyecheng/U-JEPANet/runs/school/%x-%j.out
#SBATCH --error=/public/home/heyecheng/U-JEPANet/runs/school/%x-%j.err
set -euo pipefail
ROOT=/public/home/heyecheng/U-JEPANet
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
WORD=/public/share/td20230405/WORD
SPLIT=$ROOT/data/splits_sll20
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
ARM=${1:?G0|G1}
SEED=${2:-43}
OUT=$(ls -d $ROOT/runs/school/ujepa-glp_*_${ARM}_s${SEED} 2>/dev/null | head -1)
echo "host=$(hostname) ARM=$ARM SEED=$SEED OUT=$OUT"
ls -lh "$OUT/best.pt"
"$PY" - <<PY
import sys
sys.path.insert(0, "$ROOT")
import json
from pathlib import Path
import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from ujepa.global_local_predictive import GLPUNet
from ujepa.glp_eval import sliding_window_logits_glp
from ujepa.metrics import dice_per_class
from ujepa.ct_augment import canonical_window
OUT=Path("$OUT")
W=Path("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/window")
W.mkdir(parents=True, exist_ok=True)
WORD=Path("$WORD")
SPLIT=Path("$SPLIT")
arm="$ARM"; seed=int("$SEED")
ck=torch.load(OUT/"best.pt", map_location="cpu", weights_only=False)
print("best_val", ck.get("best_val"), "alpha", ck.get("alpha_g"), flush=True)
model=GLPUNet(arm=arm, feature_chns=(16,32,64,128), class_num=17,
              embed_dim=int(ck.get("embed_dim",256)), multiscale_pred=True)
model.load_state_dict(ck["model"])
model=model.cuda().eval()
need_global = arm in ("G1","G2")
ids=[ln.strip() for ln in (SPLIT/"test_30.txt").read_text().splitlines() if ln.strip()]
acc={c:[] for c in range(1,17)}
per_case=[]
for i,cid in enumerate(ids):
    img=sitk.GetArrayFromImage(sitk.ReadImage(str(WORD/"imagesTs"/f"{cid}.nii.gz"))).astype(np.float32)
    lab=sitk.GetArrayFromImage(sitk.ReadImage(str(WORD/"labelsTs"/f"{cid}.nii.gz"))).astype(np.int64)
    vol=canonical_window(torch.from_numpy(img)[None])[None]
    glob=None
    if need_global:
        g=canonical_window(torch.from_numpy(img).float())
        glob=F.interpolate(g[None,None], size=(64,64,64), mode="trilinear", align_corners=False)
    logits=sliding_window_logits_glp(model, vol, (128,128,96), (64,64,48), torch.device("cuda"), 17,
        global_image=glob, full_shape=tuple(img.shape))
    pred=torch.argmax(logits,dim=1)[0].cpu()
    pc=dice_per_class(pred, torch.from_numpy(lab), 17)
    for c,v in pc.items():
        if v==v: acc[c].append(v)
    mean=float(np.mean([v for v in pc.values() if v==v]))
    per_case.append({"case":cid,"mean_fg_dice":mean})
    print(f"[{i+1}/30] {cid} {mean:.4f}", flush=True)
per={c:(float(np.mean(vs)) if vs else float("nan")) for c,vs in acc.items()}
vals=[v for v in per.values() if v==v]
per["ALL"]=float(np.mean(vals)) if vals else float("nan")
low=[6,7,8,9,10,12,13,14]; high=[1,2,3,4,15,16]
low_m=float(np.mean([per[c] for c in low])); high_m=float(np.mean([per[c] for c in high]))
p=W/f"test_{arm}_s{seed}_glp_window.json"
p.write_text(json.dumps({
  "arms":{"P":{"per_organ":per}},
  "arm":arm,"seed":seed,"code":"V3.1-GLP",
  "best_val":ck.get("best_val"),"alpha_g":ck.get("alpha_g"),
  "low_CNR_mean":low_m,"high_CNR_mean":high_m,
  "intensity_mode":"window",
  "protocol":"imagesTs + GLP origins + whole-CT global if G1",
  "per_case":per_case,
}, indent=2))
print("TEST_ALL", per["ALL"])
print("LOW_CNR", low_m, "HIGH_CNR", high_m)
print("WROTE", p)
PY
echo "SCHOOL_S43_TEST_DONE $ARM"
