#!/usr/bin/env bash
# U-JEPANet 40901 smoke only — NOT formal training.
# Uses physical GPU1 (GPU0 has an existing python process).
set -euo pipefail

ROOT=/data/hyc/U-JEPANet
PY=/home/ubuntu/anaconda3/envs/vllmenv/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$ROOT"

cd "$ROOT"
mkdir -p "$ROOT/runs/smoke_40901"

echo "=== HOST ==="
hostname
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo "=== TORCH ==="
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)
PY

echo "=== PREFLIGHT (CPU) ==="
"$PY" scripts/preflight.py | tee "$ROOT/runs/smoke_40901/preflight.log"

echo "=== A0 GPU SMOKE ==="
"$PY" train.py --config configs/a0_baseline.yaml --smoke --max-steps 3 --device cuda \
  --out "$ROOT/runs/smoke_40901/a0" | tee "$ROOT/runs/smoke_40901/a0.log"

echo "=== A1 GPU SMOKE ==="
"$PY" train.py --config configs/a1_dualpath.yaml --smoke --max-steps 3 --device cuda \
  --out "$ROOT/runs/smoke_40901/a1" | tee "$ROOT/runs/smoke_40901/a1.log"

echo "=== A3 GPU SMOKE (JEPA ramp shortened via tiny config override) ==="
# train.py reads yaml; for JEPA activation we need seg_only_steps small.
# Inline override via a temp config.
"$PY" - <<'PY' | tee "$ROOT/runs/smoke_40901/a3_jepa_ramp.log"
import os, sys, torch
sys.path.insert(0, "/data/hyc/U-JEPANet")
from train import SyntheticWORDLike, train_loop
from ujepa.model import UJEPAConfig, build_model

torch.manual_seed(42)
cfg = UJEPAConfig(
    arm="A3",
    feature_chns=[16, 32, 64, 128],
    class_num=16,
    embed_dim=256,
    num_heads=4,
    num_blocks=4,
    predictor_blocks=2,
    image_stride=4,
    deep_stage=2,
    multiscale_pred=True,
    mask_ratio=0.4,
)
model = build_model(cfg)
print("A3 params", sum(p.numel() for p in model.parameters()))
ds = SyntheticWORDLike(n=4, shape=(64, 64, 64), num_classes=16, seed=0)
loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)
# Force JEPA on after 2 steps of seg-only, 2-step ramp, then joint.
res = train_loop(
    model,
    loader,
    torch.device("cuda"),
    max_steps=6,
    lr=2e-4,
    weight_decay=1e-4,
    lambda_j_max=0.3,
    seg_only_steps=2,
    ramp_steps=2,
    grad_clip=1.0,
    log_every=1,
    out_dir=None,
)
last = res["history"][-1]
print("A3_SMOKE_STEP", res["steps"], "l_seg", last["l_seg"], "l_jepa", last["l_jepa"], "lambda", last["lambda_j"])
assert last["lambda_j"] > 0, "JEPA weight never activated"
assert last["l_jepa"] > 0, "JEPA loss is zero after activation"
print("A3_JEPA_SMOKE_PASS")
PY

echo "=== REAL WORD ONE-CASE INTERFACE SMOKE (labelsTr_All, train-only) ==="
"$PY" - <<'PY' | tee "$ROOT/runs/smoke_40901/word_case_interface.log"
import os, sys, torch
sys.path.insert(0, "/data/hyc/U-JEPANet")
from ujepa.model import UJEPAConfig, build_model

WORD = "/data/hyc/PLS4MIS/code/datasets/WORD"
img_p = os.path.join(WORD, "imagesTr", "word_0002.nii.gz")
lab_p = os.path.join(WORD, "labelsTr_All", "word_0002.nii.gz")
try:
    import SimpleITK as sitk
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_p)).astype("float32")
    lab = sitk.GetArrayFromImage(sitk.ReadImage(lab_p)).astype("int64")
except Exception as e:
    print("SKIP_REAL_WORD", type(e).__name__, str(e))
    raise SystemExit(0)

# ZYX -> crop a 64^3 patch near foreground for a cheap interface smoke
import numpy as np
fg = np.argwhere(lab > 0)
if len(fg) == 0:
    cz, cy, cx = np.array(img.shape) // 2
else:
    cz, cy, cx = fg[len(fg) // 2]
pd = 64
z0 = max(0, min(img.shape[0] - pd, int(cz) - pd // 2))
y0 = max(0, min(img.shape[1] - pd, int(cy) - pd // 2))
x0 = max(0, min(img.shape[2] - pd, int(cx) - pd // 2))
crop = img[z0:z0+pd, y0:y0+pd, x0:x0+pd]
# per-crop min-max like PL-Seg style simple normalize
lo, hi = float(crop.min()), float(crop.max())
crop = (crop - lo) / max(hi - lo, 1e-6)
xt = torch.from_numpy(crop).float()[None, None].cuda()
yt = torch.from_numpy(lab[z0:z0+pd, y0:y0+pd, x0:x0+pd]).long()[None].cuda()

for arm in ("A0", "A1", "A3"):
    cfg = UJEPAConfig(arm=arm, feature_chns=[16,32,64,128], class_num=16, embed_dim=256,
                      num_heads=4, num_blocks=4, predictor_blocks=2, image_stride=4,
                      deep_stage=2, multiscale_pred=True, mask_ratio=0.4)
    m = build_model(cfg).cuda()
    if arm == "A0":
        out = m(xt)
    else:
        out = m.forward_seg(xt)
    if isinstance(out, (list, tuple)):
        shape = tuple(out[0].shape)
    else:
        shape = tuple(out.shape)
    print(f"WORD_IFACE {arm} logits {shape} max={float(out[0].max()) if not isinstance(out,(list,tuple)) else float(out[0].max())}")
    assert shape[0] == 1 and shape[1] == 16

print("WORD_IFACE_PASS case=word_0002 view=labelsTr_All train-only crop=64^3")
print("NO_IMAGESVAL_READ NO_IMAGESTS_READ")
PY

echo "SMOKE_ALL_PASS"
