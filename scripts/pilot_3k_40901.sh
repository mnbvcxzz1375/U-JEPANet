#!/usr/bin/env bash
# 3k pilot A0-A3 on 40901 GPU1 — mechanism health check, NOT formal 30k.
set -euo pipefail

ROOT=/data/hyc/U-JEPANet
PY=/home/ubuntu/anaconda3/envs/vllmenv/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$ROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="$ROOT/runs/pilot_3k_${STAMP}"
COMMIT=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)

mkdir -p "$RUN_ROOT"
{
  echo "host=$(hostname)"
  echo "commit=$COMMIT"
  echo "date=$(date -Is)"
  echo "cuda_visible=$CUDA_VISIBLE_DEVICES"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
  "$PY" -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
} | tee "$RUN_ROOT/provenance.txt"

for ARM in a0 a1 a2 a3; do
  echo "===== PILOT $ARM =====" | tee -a "$RUN_ROOT/summary.log"
  /usr/bin/time -f "elapsed=%e maxrss_kb=%M" \
    "$PY" "$ROOT/train.py" \
      --config "$ROOT/configs/pilot_3k/${ARM}.yaml" \
      --device cuda \
      --seed 42 \
      --out "$RUN_ROOT/${ARM}" \
    2>&1 | tee "$RUN_ROOT/${ARM}.log"
done

"$PY" - <<PY | tee "$RUN_ROOT/summary.json"
import json, os
from pathlib import Path
root = Path("$RUN_ROOT")
rows = []
for arm in ["a0","a1","a2","a3"]:
    p = root / arm / "train_log.json"
    if not p.exists():
        rows.append({"arm": arm, "error": "missing train_log"})
        continue
    d = json.loads(p.read_text())
    hist = d.get("history") or []
    last = hist[-1] if hist else {}
    vals = d.get("val_history") or []
    best = d.get("best_mean_fg_dice")
    rows.append({
        "arm": arm,
        "steps": d.get("steps"),
        "commit": d.get("commit"),
        "elapsed_sec": d.get("elapsed_sec"),
        "last_seg": last.get("l_seg"),
        "last_jepa": last.get("l_jepa"),
        "last_lambda": last.get("lambda_j"),
        "best_val_mean_fg_dice": best,
        "final_val_mean_fg_dice": vals[-1].get("mean_fg_dice") if vals else None,
        "n_val_points": len(vals),
    })
out = {"run_root": "$RUN_ROOT", "commit": "$COMMIT", "arms": rows}
print(json.dumps(out, indent=2))
(root / "summary.json").write_text(json.dumps(out, indent=2))
PY

echo "PILOT_DONE $RUN_ROOT"
