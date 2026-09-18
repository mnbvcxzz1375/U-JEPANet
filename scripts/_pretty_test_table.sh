#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import json
from pathlib import Path

organ = {1:"Liver",2:"Spleen",3:"Kidney(L)",4:"Kidney(R)",5:"Stomach",6:"Gallbladder",
7:"Esophagus",8:"Pancreas",9:"Duodenum",10:"Colon",11:"Intestine",12:"Adrenal",
13:"Rectum",14:"Bladder",15:"Femur(L)",16:"Femur(R)"}

def load(path):
    d = json.loads(Path(path).read_text())
    out = {}
    for a,v in d.get("arms", {}).items():
        po = {}
        for k,val in v["per_organ"].items():
            if k == "ALL":
                po["ALL"] = float(val)
            else:
                po[int(k)] = float(val) if val is not None else float("nan")
        out[a] = po
    return out

school = Path("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917")
# pull A0 from 40901 via pre-copied or merge file if present; else hardcode known ALL and fetch file
# Prefer explicit per-arm jsons
arms = {}
for name, path in [
    ("A1", school/"test_A1.json"),
    ("A2", school/"test_A2.json"),
    ("A3", school/"test_A3.json"),
    ("A2-L", school/"test_A2-L.json"),
    ("A2-LU", school/"test_A2-LU.json"),
]:
    arms.update(load(path))

# A0 from local temp if we scp it; else create from known
a0_path = Path("/tmp/test_A0.json")
if a0_path.exists():
    arms.update(load(a0_path))
else:
    arms["A0"] = {"ALL": 0.7674341375449829}

order = ["A0","A1","A2","A3","A2-L","A2-LU"]
order = [a for a in order if a in arms]
print("organ | " + " | ".join(order))
for c in range(1,17):
    row = [f"{arms[a].get(c, float('nan')):.4f}" for a in order]
    print(f"{c:02d} {organ[c]:14s} | " + " | ".join(row))
print("ALL              | " + " | ".join(f"{arms[a]['ALL']:.4f}" for a in order))
print()
print("delta vs A0:")
for a in order:
    if a=="A0": continue
    print(f"  {a}: {arms[a]['ALL']-arms['A0']['ALL']:+.4f}")
# dump
out = {
  "protocol": "WORD official imagesTs/labelsTs whole-volume 128x128x96 stride 64x64x48",
  "authorized": "user 2026-09-17",
  "selection": "locked best.pt from imagesVal screening; no test-based reselection",
  "arms": {a: {"per_organ": {str(k):v for k,v in arms[a].items()}, "ALL": arms[a]["ALL"]} for a in order},
}
Path("/tmp/merged_official_test_pretty.json").write_text(json.dumps(out, indent=2))
print("WROTE /tmp/merged_official_test_pretty.json")
PY
