from __future__ import annotations

import json
from pathlib import Path

d = json.loads(Path("results/sll20_30k_20260916_1923/summary.json").read_text(encoding="utf-8"))
for a in d["arms"]:
    print(
        f"{a['arm'].upper()}  best={a['best_val']:.4f} @ {a['best_step']}  "
        f"final={a['final_val']:.4f}  jepa={a['last_jepa']}  "
        f"sec={a['elapsed_sec']:.0f}"
    )
print()
# curve at key steps
keys = [6000, 8000, 10000, 16000, 30000]
print("step | " + " | ".join(a["arm"].upper() for a in d["arms"]))
for s in keys:
    vals = []
    for a in d["arms"]:
        v = next((x["mean_fg_dice"] for x in a["val_curve"] if x["step"] == s), None)
        vals.append(f"{v:.4f}" if v is not None else "—")
    print(f"{s:5d} | " + " | ".join(vals))
