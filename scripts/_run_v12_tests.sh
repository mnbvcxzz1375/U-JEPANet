#!/bin/bash
set -euo pipefail
scp -P 40901 "E:\VScodeProject\U-JEPANet\ujepa\predictive_unet.py" "E:\VScodeProject\U-JEPANet\tests\test_predictive_v12.py" ubuntu@10.126.25.5:/tmp/
ssh -p 40901 ubuntu@10.126.25.5 "cp /tmp/predictive_unet.py /data/hyc/U-JEPANet/ujepa/; /home/ubuntu/anaconda3/envs/vllmenv/bin/python /tmp/test_predictive_v12.py"
