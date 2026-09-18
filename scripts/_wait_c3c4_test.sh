#!/bin/bash
set -euo pipefail
D=/data/hyc/U-JEPANet/runs/official_test_20260917
for i in $(seq 1 40); do
  if ps aux | grep -E 'eval_official_test.py --arms A2' | grep -v grep >/dev/null; then
    echo "poll $i still running"
    tail -2 $D/eval_C3.log 2>/dev/null | tr '\n' ' '; echo
    tail -2 $D/eval_C4.log 2>/dev/null | tr '\n' ' '; echo
    sleep 15
  else
    echo "poll $i all done"
    break
  fi
done
echo "==== C3 ===="
tail -25 $D/eval_C3.log
echo "==== C4 ===="
tail -25 $D/eval_C4.log
ls -l $D/test_C3.json $D/test_C4.json 2>/dev/null
python3 - <<'PY'
import json
from pathlib import Path
organ={1:"Liver",2:"Spleen",3:"Kidney(L)",4:"Kidney(R)",5:"Stomach",6:"Gallbladder",
7:"Esophagus",8:"Pancreas",9:"Duodenum",10:"Colon",11:"Intestine",12:"Adrenal",
13:"Rectum",14:"Bladder",15:"Femur(L)",16:"Femur(R)"}
D=Path("/data/hyc/U-JEPANet/runs/official_test_20260917")
known={"A0":0.7674,"C1":0.8118,"C2":0.8169}
for arm,f in [("C3","test_C3.json"),("C4","test_C4.json")]:
    p=D/f
    if not p.exists():
        print(arm,"MISSING"); continue
    d=json.loads(p.read_text())
    po=d["arms"]["A2"]["per_organ"]
    print(arm,"ALL",po["ALL"])
    for a,v in known.items():
        print(f"  vs {a}: {po['ALL']-v:+.4f}")
PY
