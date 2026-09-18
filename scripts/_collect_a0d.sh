#!/bin/bash
set -euo pipefail
D=/public/home/heyecheng/U-JEPANet/runs/school/ujepa-a0d_383177
echo "==== job ===="
sacct -j 383177 --format=JobID,State,Elapsed,ExitCode,NodeList
echo "==== dir ===="
ls -lh "$D"
echo "==== summary ===="
if [ -f "$D/summary.json" ]; then
  python3 -c "import json; d=json.load(open('$D/summary.json')); print({k:d.get(k) for k in ['arm','best_val','steps','strength','jepa_mode','use_jepa','elapsed_sec']}); print('final', (d.get('val_curve') or [{}])[-1]); print('n_val', len(d.get('val_curve') or []))"
fi
echo "==== log tail ===="
tail -20 /public/home/heyecheng/U-JEPANet/runs/school/ujepa-a0d-383177.out
