#!/bin/bash
set -euo pipefail
# Upload fixed code to school + 40901, submit A0DA, prep window re-eval.
echo "=== school: push code + submit A0DA ==="
scp "E:\VScodeProject\U-JEPANet\ujepa\dynamic_dataset.py" \
    "E:\VScodeProject\U-JEPANet\ujepa\whole_volume_eval.py" \
    "E:\VScodeProject\U-JEPANet\ujepa\paired_views.py" \
    "E:\VScodeProject\U-JEPANet\scripts\train_c_ladder.py" \
    "E:\VScodeProject\U-JEPANet\scripts\eval_official_test.py" \
    "E:\VScodeProject\U-JEPANet\scripts\school_sbatch_a0da.sh" \
    school-platform:/tmp/
ssh school-platform "cp /tmp/dynamic_dataset.py /tmp/whole_volume_eval.py /tmp/paired_views.py /public/home/heyecheng/U-JEPANet/ujepa/; cp /tmp/train_c_ladder.py /tmp/eval_official_test.py /tmp/school_sbatch_a0da.sh /public/home/heyecheng/U-JEPANet/scripts/; cd /public/home/heyecheng/U-JEPANet; sbatch --export=ALL scripts/school_sbatch_a0da.sh; squeue -u heyecheng -o '%i %P %j %T %N' | head -12"
echo "=== 40901: push code + ckpts for window re-eval ==="
scp -P 40901 "E:\VScodeProject\U-JEPANet\ujepa\dynamic_dataset.py" "E:\VScodeProject\U-JEPANet\ujepa\whole_volume_eval.py" "E:\VScodeProject\U-JEPANet\ujepa\paired_views.py" "E:\VScodeProject\U-JEPANet\scripts\eval_official_test.py" "E:\VScodeProject\U-JEPANet\scripts\train_c_ladder.py" "E:\VScodeProject\U-JEPANet\scripts\run_window_reeval_40901.sh" ubuntu@10.126.25.5:/tmp/
ssh -p 40901 ubuntu@10.126.25.5 "cp /tmp/dynamic_dataset.py /tmp/whole_volume_eval.py /tmp/paired_views.py /data/hyc/U-JEPANet/ujepa/; cp /tmp/eval_official_test.py /tmp/train_c_ladder.py /tmp/run_window_reeval_40901.sh /data/hyc/U-JEPANet/scripts/; mkdir -p /data/hyc/U-JEPANet/runs/c_ladder_40901/{C1,C2,C3,C4} /data/hyc/U-JEPANet/runs/school_local_copy; ls /data/hyc/U-JEPANet/runs/c_ladder_40901/C3/best.pt /data/hyc/U-JEPANet/runs/c_ladder_40901/C4/best.pt"
# pull school A0D/C1/C2 ckpts to 40901
scp school-platform:/public/home/heyecheng/U-JEPANet/runs/school/ujepa-a0d_383177/best.pt ubuntu@10.126.25.5:/data/hyc/U-JEPANet/runs/school_local_copy/a0d_best.pt
scp school-platform:/public/home/heyecheng/U-JEPANet/runs/school/ujepa-c1_381886/best.pt ubuntu@10.126.25.5:/data/hyc/U-JEPANet/runs/c_ladder_40901/C1/best.pt
scp school-platform:/public/home/heyecheng/U-JEPANet/runs/school/ujepa-c1_382006/best.pt ubuntu@10.126.25.5:/data/hyc/U-JEPANet/runs/c_ladder_40901/C2/best.pt
echo "=== 40901 dual GPU window re-eval ==="
ssh -p 40901 ubuntu@10.126.25.5 "cd /data/hyc/U-JEPANet; chmod +x scripts/run_window_reeval_40901.sh; nohup bash scripts/run_window_reeval_40901.sh 0 C3 > runs/official_test_20260917/window_C3.log 2>&1 & nohup bash scripts/run_window_reeval_40901.sh 1 C4 > runs/official_test_20260917/window_C4.log 2>&1 & sleep 2; ps aux | grep eval_official | grep -v grep"
echo LAUNCHED
