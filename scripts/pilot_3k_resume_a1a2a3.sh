#!/usr/bin/env bash
# Resume pilot for A1-A3 only (A0 already complete in same run dir).
set -euo pipefail

ROOT=/data/hyc/U-JEPANet
PY=/home/ubuntu/anaconda3/envs/vllmenv/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
RUN_ROOT=${1:-/data/hyc/U-JEPANet/runs/pilot_3k_20260916_185115}
COMMIT=${2:-ab57558-fixA1}

mkdir -p "$RUN_ROOT"
echo "ab57558" > "$RUN_ROOT/COMMIT_SHA"
{
  echo "host=$(hostname)"
  echo "commit=$COMMIT"
  echo "date=$(date -Is)"
  echo "python=$PY"
  "$PY" -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
} | tee -a "$RUN_ROOT/provenance.txt"

for ARM in a1 a2 a3; do
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
import json
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
    rows.append({
        "arm": arm,
        "steps": d.get("steps"),
        "commit": d.get("commit"),
        "elapsed_sec": d.get("elapsed_sec"),
        "last_seg": last.get("l_seg"),
        "last_jepa": last.get("l_jepa"),
        "last_lambda": last.get("lambda_j"),
        "best_val_mean_fg_dice": d.get("best_mean_fg_dice"),
        "final_val_mean_fg_dice": vals[-1].get("mean_fg_dice") if vals else None,
        "n_val_points": len(vals),
    })
out = {"run_root": "$RUN_ROOT", "commit": "$COMMIT", "arms": rows}
print(json.dumps(out, indent=2))
(root / "summary.json").write_text(json.dumps(out, indent=2))
PY

echo "PILOT_RESUME_DONE $RUN_ROOT"
