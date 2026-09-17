from __future__ import annotations

import json
from pathlib import Path

ORGAN_NAMES = {
    1: "Liver",
    2: "Spleen",
    3: "Kidney(L)",
    4: "Kidney(R)",
    5: "Stomach",
    6: "Gallbladder",
    7: "Esophagus",
    8: "Pancreas",
    9: "Duodenum",
    10: "Colon",
    11: "Intestine",
    12: "Adrenal",
    13: "Rectum",
    14: "Bladder",
    15: "Femur(L)",
    16: "Femur(R)",
}

p = Path("results/sll20_30k_20260916_1923/per_organ_dice.json")
d = json.loads(p.read_text(encoding="utf-8"))
arms = ["a0", "a1", "a2", "a3"]
# rewrite organ_names in file
d["organ_names"] = {str(k): v for k, v in ORGAN_NAMES.items()}
p.write_text(json.dumps(d, indent=2), encoding="utf-8")

print("organ | " + " | ".join(arms) + " | A2-A0 | A3-A0")
for c in range(1, 17):
    vals = []
    for a in arms:
        v = d["arms"][a].get(str(c), d["arms"][a].get(c))
        vals.append(f"{float(v):.4f}" if v is not None else "—")
    a2d = float(d["arms"]["a2"].get(str(c), d["arms"]["a2"].get(c))) - float(d["arms"]["a0"].get(str(c), d["arms"]["a0"].get(c)))
    a3d = float(d["arms"]["a3"].get(str(c), d["arms"]["a3"].get(c))) - float(d["arms"]["a0"].get(str(c), d["arms"]["a0"].get(c)))
    print(f"{c:02d} {ORGAN_NAMES[c]:12s} | " + " | ".join(vals) + f" | {a2d:+.4f} | {a3d:+.4f}")
print("ALL | " + " | ".join(f"{float(d['arms'][a]['ALL']):.4f}" for a in arms))
