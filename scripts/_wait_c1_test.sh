#!/bin/bash
set -euo pipefail
for i in $(seq 1 30); do
  if squeue -j 382955 -h | grep -q .; then
    echo "poll $i $(squeue -j 382955 -h -o '%T %M')"
    sleep 12
  else
    echo "poll $i done"
    break
  fi
done
sacct -j 382955 --format=JobID,State,Elapsed,ExitCode,NodeList
D=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917
ls -l "$D"/test_C1.json 2>/dev/null || echo MISSING
tail -40 "$D"/slurm_*c1test.out 2>/dev/null | tail -50
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_C1.json")
if not p.exists():
    print("no json")
    raise SystemExit
d=json.loads(p.read_text())
organ={1:"Liver",2:"Spleen",3:"Kidney(L)",4:"Kidney(R)",5:"Stomach",6:"Gallbladder",
7:"Esophagus",8:"Pancreas",9:"Duodenum",10:"Colon",11:"Intestine",12:"Adrenal",
13:"Rectum",14:"Bladder",15:"Femur(L)",16:"Femur(R)"}
# compare against known ALLs
known={"A0":0.7674,"A1":0.7657,"A2":0.7636,"A3":0.7580,"A2-L":0.7663,"A2-LU":0.7700}
po=d["arms"]["A2"]["per_organ"]
print("C1 ALL", po["ALL"])
for a,v in known.items():
    print(f"  vs {a}: {po['ALL']-v:+.4f}")
print("organ | C1 | A0 | A2 | A2-LU")
# load others if present
others={}
for name,path in [
 ("A0","/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_A0.json"),
 ("A2","/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_A2.json"),
 ("A2-LU","/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/test_A2-LU.json"),
]:
    q=Path(path)
    # A0 may only be on 40901; use merged if needed
    if q.exists():
        dd=json.loads(q.read_text())
        others[name]=dd["arms"][name]["per_organ"]
merged=Path("/public/home/heyecheng/U-JEPANet/runs/official_test_20260917/merged_official_test.json")
if merged.exists() and "A0" not in others:
    dd=json.loads(merged.read_text())
    for a,v in dd.get("arms",{}).items():
        others.setdefault(a, v if "per_organ" not in v else v["per_organ"])
# fallback from local known A0 organ from analysis
for c in range(1,17):
    row=[f"{po[str(c)]:.4f}"]
    for a in ["A0","A2","A2-LU"]:
        if a in others:
            row.append(f"{others[a][str(c)]:.4f}" if str(c) in others[a] else "—")
    print(f"{c:02d} {organ[c]:14s} | " + " | ".join(row))
print("C1 per_case count", len(d.get("per_case",{}).get("A2",[])))
# worst cases
pcs=d.get("per_case",{}).get("A2",[])
pcs_sorted=sorted(pcs, key=lambda x: x["mean_fg_dice"])
print("worst 8 C1 cases:")
for pc in pcs_sorted[:8]:
    per={int(k):v for k,v in pc["per_class"].items() if v is not None}
    wo=min(per, key=per.get) if per else None
    print(f"  {pc['case']} mean={pc['mean_fg_dice']:.4f} worst={organ.get(wo,wo)}")
PY
