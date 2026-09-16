from __future__ import annotations

import json
from pathlib import Path

p = Path(r"E:\VScodeProject\U-JEPANet\runs\sll20_a0_train_log.json")
d = json.loads(p.read_text(encoding="utf-8"))
vals = d.get("val_history") or []
print("steps", d.get("steps"), "elapsed_sec", round(float(d.get("elapsed_sec", 0)), 1))
print("best_val", d.get("best_mean_fg_dice"))
for v in vals:
    print(f"step={int(v.get('step', 0)):5d} mean_fg_dice={float(v.get('mean_fg_dice', float('nan'))):.4f} n={v.get('n_val_cases')}")
