#!/bin/bash
set -euo pipefail
echo "==== job 381886 ===="
sacct -j 381886 --format=JobID,JobName%24,State,Elapsed,ExitCode,NodeList,WorkDir
scontrol show job 381886 | grep -E 'JobName|StdOut|StdErr|Command|WorkDir|JobState|Partition'
echo "==== c1 outputs ===="
# try common locations
for d in \
  /public/home/heyecheng/U-JEPANet/runs/c_ladder \
  /public/home/heyecheng/U-JEPANet/runs \
  /public/home/heyecheng/U-JEPANet/runs/c1 \
  /public/home/heyecheng/U-JEPANet/runs/c_ladder_a800 \
  /public/home/heyecheng/U-JEPANet/runs/c_ladder_40901
 do
  if [ -d "$d" ]; then
    echo "-- $d --"
    ls -lt "$d" | head -15
  fi
done
echo "==== find C1 best ===="
find /public/home/heyecheng/U-JEPANet/runs -name 'best.pt' 2>/dev/null | head -30
echo "==== slurm out ===="
ls -lt /public/home/heyecheng/U-JEPANet/runs 2>/dev/null | head
# search sbatch logs
find /public/home/heyecheng -name '*381886*' 2>/dev/null | head
