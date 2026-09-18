#!/bin/bash
set -euo pipefail
echo "==== school ===="
sinfo -p hpc_gpu,gpu_4090 -N -o '%N %P %t %C %G'
squeue -u heyecheng -o '%i %P %j %T %M %N'
echo "==== a0da progress ===="
tail -5 /public/home/heyecheng/U-JEPANet/runs/school/ujepa-a0da-383499.out 2>/dev/null || true
echo "==== ckpt dirs ===="
ls /public/home/heyecheng/U-JEPANet/runs/school/ | head -20
