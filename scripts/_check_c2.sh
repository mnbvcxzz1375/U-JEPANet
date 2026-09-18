#!/bin/bash
set -euo pipefail
echo "==== C2 job 382006 ===="
sacct -j 382006 --format=JobID,JobName%20,State,Elapsed,ExitCode,NodeList
ls -lh /public/home/heyecheng/U-JEPANet/runs/school/ujepa-c1_382006/ 2>/dev/null || echo no_dir
if [ -f /public/home/heyecheng/U-JEPANet/runs/school/ujepa-c1_382006/summary.json ]; then
  python3 -c "import json; d=json.load(open('/public/home/heyecheng/U-JEPANet/runs/school/ujepa-c1_382006/summary.json')); print('arm',d.get('arm'),'best_val',d.get('best_val'),'steps',d.get('steps'),'strength',d.get('strength'),'jepa_mode',d.get('jepa_mode')); print('final_val', (d.get('val_curve') or [{}])[-1])"
fi
echo "==== queue C2-related ===="
squeue -u heyecheng -o '%i %P %j %T %M %N' | head -15
