#!/bin/bash
set -euo pipefail
ZIP='/public/share/td20230405/AMOS 22/amos22.zip'
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
unzip -p "$ZIP" amos22/readme.md | head -80
echo '==== dataset.json labels ===='
unzip -p "$ZIP" amos22/dataset.json > /tmp/amos_ds.json
"$PY" - <<'PY'
import json
d=json.load(open("/tmp/amos_ds.json"))
print("keys", list(d.keys())[:20])
lab=d.get("labels") or d.get("labelsTr")
print("labels", json.dumps(lab, indent=2)[:2500])
print("numTraining", d.get("numTraining"))
print("modality", d.get("modality"))
PY
echo '==== counts ===='
unzip -l "$ZIP" | grep -c 'imagesTr/.*nii' || true
unzip -l "$ZIP" | grep -c 'labelsTr/.*nii' || true
