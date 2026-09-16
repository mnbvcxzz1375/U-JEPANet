#!/usr/bin/env bash
# Freeze split, cache SLL patches, smoke dual-loader, then sequential A0-A3 30k.
set -euo pipefail

ROOT=/data/hyc/U-JEPANet
PY=/home/ubuntu/anaconda3/envs/vllmenv/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
WORD=/data/hyc/PLS4MIS/code/datasets/WORD
SPLIT=$ROOT/data/splits_sll20
CACHE=$ROOT/data/word_sll20_cache
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT=$ROOT/runs/sll20_30k_${STAMP}
COMMIT=${COMMIT_SHA:-5ac7f87}

mkdir -p "$RUN_ROOT" "$ROOT/data"

if [ ! -f "$SPLIT/labeled_train_20.txt" ]; then
  "$PY" "$ROOT/scripts/freeze_sll_split.py" \
    --word-root "$WORD" --out "$SPLIT" --seed-split 20260916 --n-labeled 20 \
    | tee "$RUN_ROOT/split_freeze.log"
fi

if [ ! -f "$CACHE/splits/train_all.txt" ]; then
  "$PY" "$ROOT/scripts/cache_sll_patches.py" \
    --word-root "$WORD" --split-dir "$SPLIT" --out "$CACHE" \
    --patch 128,128,96 --crops-labeled 12 --crops-unlabeled 6 --seed 42 \
    | tee "$RUN_ROOT/cache_sll.log"
fi

{
  echo "host=$(hostname)"
  echo "commit=$COMMIT"
  echo "date=$(date -Is)"
  echo "protocol=WORD-SLL-20% dual_loader whole_val seed42"
  echo "python=$PY"
  "$PY" -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
  nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
  echo "labeled=$(wc -l < "$SPLIT/labeled_train_20.txt") unlabeled=$(wc -l < "$SPLIT/unlabeled_train_80.txt") val=$(wc -l < "$SPLIT/val_20.txt")"
  echo "patches_labeled=$(wc -l < "$CACHE/splits/train_labeled.txt") patches_all=$(wc -l < "$CACHE/splits/train_all.txt")"
} | tee "$RUN_ROOT/provenance.txt"

# Short smoke of dual loader + one whole-val case path (2 steps)
"$PY" - <<PY | tee "$RUN_ROOT/smoke_dual.log"
import sys
sys.path.insert(0, "$ROOT")
from train import build_sll_dual_loaders, train_loop
from ujepa.model import UJEPAConfig, build_model
import torch, yaml
cfg = UJEPAConfig(arm="A3", feature_chns=[16,32,64,128], class_num=17, embed_dim=256,
                  num_heads=4, num_blocks=2, predictor_blocks=1, jepa_token_stride=16,
                  deep_stage=2, multiscale_pred=True, mask_ratio=0.4)
raw = {
  "data": {"kind":"npy_sll","root":"$CACHE","word_root":"$WORD","val_ids_file":"$SPLIT/val_20.txt","patch_size":[128,128,96]},
  "train": {"batch_size":1},
}
model = build_model(cfg)
seg_loader, jepa_loader, val_ids = build_sll_dual_loaders(cfg, raw, 42)
print("seg", len(seg_loader.dataset), "jepa", len(jepa_loader.dataset), "val", len(val_ids))
res = train_loop(model, seg_loader, torch.device("cuda"), max_steps=2,
  lr=2e-4, weight_decay=1e-4, lambda_j_max=0.3, seg_only_steps=0, ramp_steps=0,
  grad_clip=1.0, log_every=1, out_dir=None, save_every=0,
  jepa_loader=jepa_loader, val_mode="crop", val_every=0, seed=42)
print("SMOKE_DUAL_PASS", res["history"][-1])
PY

for ARM in a0 a1 a2 a3; do
  echo "===== 30k $ARM =====" | tee -a "$RUN_ROOT/summary.log"
  /usr/bin/time -f "elapsed=%e maxrss_kb=%M" \
    "$PY" "$ROOT/train.py" \
      --config "$ROOT/configs/sll20_30k/${ARM}.yaml" \
      --device cuda --seed 42 \
      --out "$RUN_ROOT/${ARM}" \
    2>&1 | tee "$RUN_ROOT/${ARM}.log"
done

"$PY" - <<PY | tee "$RUN_ROOT/summary.json"
import json
from pathlib import Path
root = Path("$RUN_ROOT")
rows=[]
for arm in ["a0","a1","a2","a3"]:
    p=root/arm/"train_log.json"
    if not p.exists():
        rows.append({"arm":arm,"error":"missing"}); continue
    d=json.loads(p.read_text())
    vals=d.get("val_history") or []
    last=(d.get("history") or [{}])[-1]
    rows.append({
      "arm":arm,"steps":d.get("steps"),"elapsed_sec":d.get("elapsed_sec"),
      "best_val":d.get("best_mean_fg_dice"),
      "final_val":vals[-1].get("mean_fg_dice") if vals else None,
      "n_val":len(vals),
      "last_jepa":last.get("l_jepa"),
      "last_seg":last.get("l_seg"),
    })
out={"run_root":"$RUN_ROOT","commit":"$COMMIT","protocol":"SLL20-30k-seed42","arms":rows}
print(json.dumps(out,indent=2))
(root/"summary.json").write_text(json.dumps(out,indent=2))
PY

echo "SLL20_30K_DONE $RUN_ROOT"
