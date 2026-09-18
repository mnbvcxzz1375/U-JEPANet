#!/bin/bash
set -euo pipefail
D=/public/home/heyecheng/U-JEPANet/runs/official_test_20260917
ls -la "$D"
echo "==== full out A1 ===="
cat "$D/slurm_382524_ujepa-test.out"
echo "==== json heads ===="
for f in "$D"/test_*.json; do
  echo "-- $f --"
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('arms',list(d.get('arms',{}).keys()));
import pprint
for a,v in d.get('arms',{}).items():
  print(a, 'ALL', v.get('per_organ',{}).get('ALL'), 'n', v.get('n_test'), 'keys', list(v.keys())[:8])
  po=v.get('per_organ',{})
  print(' sample', {k:po.get(k) for k in list(po)[:5]})
" "$f"
done
echo "==== python/cuda ===="
/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python -c "import torch,SimpleITK,numpy; print(torch.__version__, torch.cuda.is_available(), SimpleITK.__version__, numpy.__version__)"
