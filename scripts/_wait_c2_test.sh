#!/bin/bash
set -euo pipefail
for i in $(seq 1 30); do
  if squeue -j 383257 -h | grep -q .; then
    echo "poll $i $(squeue -j 383257 -h -o '%T %M')"
    sleep 12
  else
    echo "poll $i done"
    break
  fi
done
sacct -j 383257 --format=JobID,State,Elapsed,ExitCode,NodeList
D=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917
ls -l "$D"/test_C2.json 2>/dev/null || echo MISSING
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_C2.json")
if not p.exists():
    print("no json"); raise SystemExit
d=json.loads(p.read_text())
organ={1:"Liver",2:"Spleen",3:"Kidney(L)",4:"Kidney(R)",5:"Stomach",6:"Gallbladder",
7:"Esophagus",8:"Pancreas",9:"Duodenum",10:"Colon",11:"Intestine",12:"Adrenal",
13:"Rectum",14:"Bladder",15:"Femur(L)",16:"Femur(R)"}
po=d["arms"]["A2"]["per_organ"]
print("C2 ALL", po["ALL"])
print("C2 vs C1", po["ALL"]-0.8118238625540593)
print("C2 vs A0", po["ALL"]-0.7674341375449829)
# load C1
c1=json.loads(Path("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_C1.json").read_text())["arms"]["A2"]["per_organ"]
print("organ | C2 | C1 | C2-C1")
for c in range(1,17):
    print(f"{c:02d} {organ[c]:14s} | {po[str(c)]:.4f} | {c1[str(c)]:.4f} | {po[str(c)]-c1[str(c)]:+.4f}")
pcs=d.get("per_case",{}).get("A2",[])
pcs_sorted=sorted(pcs, key=lambda x: x["mean_fg_dice"])
print("worst 8 C2 cases:")
for pc in pcs_sorted[:8]:
    per={int(k):v for k,v in pc["per_class"].items() if v is not None}
    wo=min(per, key=per.get) if per else None
    print(f"  {pc['case']} mean={pc['mean_fg_dice']:.4f} worst={organ.get(wo,wo)}")
PY
