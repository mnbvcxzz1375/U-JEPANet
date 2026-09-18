#!/bin/bash
set -euo pipefail
D=/public/home/heyecheng/U-JEPANet/runs/school
echo "==== 381886 dir ===="
ls -lh "$D/ujepa-c1_381886/"
echo "==== train_log tail ===="
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/public/home/heyecheng/U-JEPANet/runs/school/ujepa-c1_381886/train_log.json")
if p.exists():
    d=json.loads(p.read_text())
    print("keys", list(d.keys())[:20])
    print("best", d.get("best_mean_fg_dice"), "steps", d.get("steps"))
    vals=d.get("val_history") or []
    print("n_val", len(vals))
    if vals:
        print("last val", vals[-1])
        best=max(vals, key=lambda x: x.get("mean_fg_dice", -1))
        print("best val", best)
else:
    print("no train_log.json")
    # try summary
    for name in ["summary.json","summary.log","c1_summary.json"]:
        q=Path("/public/home/heyecheng/U-JEPANet/runs/school/ujepa-c1_381886")/name
        if q.exists():
            print(name, q.read_text()[:2000])
PY
echo "==== slurm tail ===="
tail -40 /public/home/heyecheng/U-JEPANet/runs/school/ujepa-c1-381886.out 2>/dev/null || tail -40 /public/home/heyecheng/U-JEPANet/runs/school/ujepa-c1-381886.err
echo "==== config in ckpt ===="
/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python - <<'PY'
import torch
p="/public/home/heyecheng/U-JEPANet/runs/school/ujepa-c1_381886/best.pt"
ck=torch.load(p, map_location="cpu", weights_only=False)
print("top keys", list(ck.keys()))
print("config", ck.get("config"))
print("best", ck.get("best_mean_fg_dice"), ck.get("step"))
PY
