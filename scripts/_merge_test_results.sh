#!/bin/bash
set -euo pipefail
D=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917
python3 - <<'PY'
import json, glob, os
from pathlib import Path
D = Path("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917")
organ = {1:"Liver",2:"Spleen",3:"Kidney(L)",4:"Kidney(R)",5:"Stomach",6:"Gallbladder",
7:"Esophagus",8:"Pancreas",9:"Duodenum",10:"Colon",11:"Intestine",12:"Adrenal",
13:"Rectum",14:"Bladder",15:"Femur(L)",16:"Femur(R)"}
arms = {}
for p in sorted(D.glob("test_*.json")):
    d = json.loads(p.read_text())
    for a,v in d.get("arms", {}).items():
        arms[a] = {"per_organ": v["per_organ"], "ckpt": v.get("ckpt"), "n_test": v.get("n_test"), "src": str(p)}
print("arms", list(arms))
print("organ | " + " | ".join(arms))
for c in range(1,17):
    row = [f"{arms[a]['per_organ'].get(c, float('nan')):.4f}" for a in arms]
    print(f"{c:02d} {organ[c]:14s} | " + " | ".join(row))
print("ALL              | " + " | ".join(f"{arms[a]['per_organ']['ALL']:.4f}" for a in arms))
# dump merged
out = {
  "protocol": "WORD official imagesTs/labelsTs whole-volume 128x128x96 stride 64x64x48",
  "authorized": "user 2026-09-17",
  "note": "A0/A2 from 40901; A1/A3/A2-L/A2-LU from school 4090d. Single best.pt per arm, no test-based selection.",
  "hosts": {"A0":"40901","A2":"40901","A1":"school-gpu03","A3":"school-gpu04","A2-L":"school-gpu03","A2-LU":"school-gpu03/04"},
  "arms": {a: {"per_organ": v["per_organ"], "n_test": v["n_test"], "ckpt": v["ckpt"], "src": v["src"]} for a,v in arms.items()},
}
(D/"merged_official_test.json").write_text(json.dumps(out, indent=2))
print("WROTE", D/"merged_official_test.json")
PY
# also copy A0/A2 json from 40901 if missing on school
rsync -az -e 'ssh -p 40901 -o StrictHostKeyChecking=no' ubuntu@10.126.25.5:/data/hyc/U-JEPANet/runs/official_test_20260917/test_A0.json ubuntu@10.126.25.5:/data/hyc/U-JEPANet/runs/official_test_20260917/test_A2.json "$D/" 2>/dev/null || true
ls -lt "$D"/test_*.json "$D"/merged_official_test.json 2>/dev/null || true
