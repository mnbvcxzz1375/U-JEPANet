#!/bin/bash
set -euo pipefail
for i in $(seq 1 35); do
  st=$(squeue -j 383493 -h -o '%T %M' || true)
  if [ -z "$st" ]; then echo "poll $i done"; break; fi
  echo "poll $i $st"
  tail -2 /public/home/heyecheng/U-JEPANet/runs/official_test_20260917/slurm_383493_a0dtest.out 2>/dev/null | tr '\n' ' '; echo
  sleep 12
done
sacct -j 383493 --format=JobID,State,Elapsed,ExitCode
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_A0D.json")
if not p.exists():
    print("MISSING"); raise SystemExit
d=json.loads(p.read_text())
organ={1:"Liver",2:"Spleen",3:"Kidney(L)",4:"Kidney(R)",5:"Stomach",6:"Gallbladder",
7:"Esophagus",8:"Pancreas",9:"Duodenum",10:"Colon",11:"Intestine",12:"Adrenal",
13:"Rectum",14:"Bladder",15:"Femur(L)",16:"Femur(R)"}
po=d["arms"]["A0"]["per_organ"]
print("A0D ALL", po["ALL"])
print("A0D - A0", po["ALL"]-0.7674341375449829)
print("C1 - A0D", 0.8118238625540593-po["ALL"])
print("C3 - A0D", 0.8203055778585197-po["ALL"])
print("organ | A0D | A0 | C1 | C3")
# local C1/C3 if present on school
c1p=Path("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_C1.json")
c3p=Path("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_C3.json")
c1=json.loads(c1p.read_text())["arms"]["A2"]["per_organ"] if c1p.exists() else {}
c3=json.loads(c3p.read_text())["arms"]["A2"]["per_organ"] if c3p.exists() else {}
a0={"1":0.9367,"2":0.9200,"3":0.9002,"4":0.9214,"5":0.8385,"6":0.6201,"7":0.6305,"8":0.7110,"9":0.5426,"10":0.7500,"11":0.7817,"12":0.5754,"13":0.5876,"14":0.8265,"15":0.8457,"16":0.8910,"ALL":0.7674}
for c in range(1,17):
    row=[f"{po[str(c)]:.4f}", f"{a0[str(c)]:.4f}"]
    row.append(f"{c1[str(c)]:.4f}" if c1 else "—")
    row.append(f"{c3[str(c)]:.4f}" if c3 else "—")
    print(f"{c:02d} {organ[c]:14s} | " + " | ".join(row))
pcs=d.get("per_case",{}).get("A0",[])
print("worst 6 A0D:")
for pc in sorted(pcs, key=lambda x:x["mean_fg_dice"])[:6]:
    per={int(k):v for k,v in pc["per_class"].items() if v is not None}
    wo=min(per,key=per.get) if per else None
    print(f"  {pc['case']} {pc['mean_fg_dice']:.4f} {organ.get(wo,wo)}")
PY
