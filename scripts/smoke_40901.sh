#!/usr/bin/env bash
# U-JEPANet 40901 smoke only — NOT formal training.
set -euo pipefail

ROOT=/data/hyc/U-JEPANet
PY=/home/ubuntu/anaconda3/envs/vllmenv/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$ROOT"

cd "$ROOT"
mkdir -p "$ROOT/runs/smoke_40901_v2"

echo "=== HOST / GPU ==="
hostname
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

echo "=== PREFLIGHT ==="
"$PY" scripts/preflight.py | tee "$ROOT/runs/smoke_40901_v2/preflight.log"

echo "=== A0 GPU SMOKE ==="
"$PY" train.py --config configs/a0_baseline.yaml --smoke --max-steps 3 --device cuda \
  --out "$ROOT/runs/smoke_40901_v2/a0" | tee "$ROOT/runs/smoke_40901_v2/a0.log"

echo "=== A3 GPU SMOKE with JEPA ramp + labeled/unlabeled mix ==="
"$PY" - <<'PY' | tee "$ROOT/runs/smoke_40901_v2/a3_ramp.log"
import sys, torch
sys.path.insert(0, "/data/hyc/U-JEPANet")
from train import SyntheticWORDLike, train_loop, collate_batch
from ujepa.model import UJEPAConfig, build_model

torch.manual_seed(42)
cfg = UJEPAConfig(
    arm="A3", feature_chns=[16,32,64,128], class_num=17,
    embed_dim=256, num_heads=4, num_blocks=4, predictor_blocks=2,
    jepa_token_stride=8, deep_stage=2, multiscale_pred=True, mask_ratio=0.4,
)
model = build_model(cfg)
print("A3 params", sum(p.numel() for p in model.parameters()))
ds = SyntheticWORDLike(n=4, shape=(64,64,64), num_classes=17, seed=0, unlabeled_frac=0.5)
loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_batch)
res = train_loop(
    model, loader, torch.device("cuda"),
    max_steps=6, lr=2e-4, weight_decay=1e-4,
    lambda_j_max=0.3, seg_only_steps=2, ramp_steps=2,
    grad_clip=1.0, log_every=1, out_dir=None,
)
last = res["history"][-1]
print("A3_SMOKE", last)
assert last["lambda_j"] > 0 and last["l_jepa"] > 0
print("A3_JEPA_SMOKE_PASS")
# EMA in state_dict
assert any(k.startswith("target.target.") for k in model.state_dict())
print("EMA_STATE_DICT_OK")
PY

echo "=== REAL WORD 128x128x96 INTERFACE (train imagesTr + labelsTr_All only) ==="
"$PY" - <<'PY' | tee "$ROOT/runs/smoke_40901_v2/word_iface.log"
import os, sys, torch
sys.path.insert(0, "/data/hyc/U-JEPANet")
from ujepa.model import UJEPAConfig, build_model
from ujepa.masking import sample_token_mask, mask_hidden_fraction

WORD = "/data/hyc/PLS4MIS/code/datasets/WORD"
cfg = UJEPAConfig(
    arm="A3", feature_chns=[16,32,64,128], class_num=17,
    embed_dim=256, num_heads=4, num_blocks=4, predictor_blocks=2,
    jepa_token_stride=16, deep_stage=2, multiscale_pred=True, mask_ratio=0.4,
)
m = build_model(cfg).cuda()
# token budget
grid = m.online.jepa_token_grid((128,128,96))
n = grid[0]*grid[1]*grid[2]
print("token_grid", grid, "N", n)
assert 256 <= n <= 1024

# shape check with random 128x128x96 (memory probe, not full train)
x = torch.randn(1,1,128,128,96, device="cuda")
with torch.no_grad():
    out = m.jepa_step(x)
print("jepa_step", out["pred_tokens"].shape, "masked_frac", 1-out["visible"].float().mean().item())
logits = m.forward_seg(x)
print("seg_logits", tuple(logits[0].shape) if isinstance(logits,(list,tuple)) else tuple(logits.shape))

# one real case crop 64^3 still used for byte-level interface
import numpy as np, SimpleITK as sitk
img = sitk.GetArrayFromImage(sitk.ReadImage(f"{WORD}/imagesTr/word_0002.nii.gz")).astype("float32")
lab = sitk.GetArrayFromImage(sitk.ReadImage(f"{WORD}/labelsTr_All/word_0002.nii.gz")).astype("int64")
print("label_unique", np.unique(lab)[:20].tolist(), "n", len(np.unique(lab)))
assert lab.max() == 16
fg = np.argwhere(lab > 0)
cz, cy, cx = fg[len(fg)//2]
pd=64
z0=max(0,min(img.shape[0]-pd,int(cz)-pd//2)); y0=max(0,min(img.shape[1]-pd,int(cy)-pd//2)); x0=max(0,min(img.shape[2]-pd,int(cx)-pd//2))
c=img[z0:z0+pd,y0:y0+pd,x0:x0+pd]; lo,hi=float(c.min()),float(c.max()); c=(c-lo)/max(hi-lo,1e-6)
xt=torch.from_numpy(c).float()[None,None].cuda()
with torch.no_grad():
    out=m.forward_seg(xt)
print("real_crop_logits", tuple(out[0].shape))
print("WORD_IFACE_PASS NO_IMAGESVAL_READ NO_IMAGESTS_READ")
PY

echo "SMOKE_ALL_PASS"
