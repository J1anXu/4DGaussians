import subprocess
from datetime import datetime
import os
import pytz

timezone = pytz.timezone("America/Chicago")
current_time = datetime.now(timezone).strftime("%Y-%m-%d_%H-%M-%S")

idx = 1

command1 = (
    f"python trainadmm_with_ckpt_{idx}.py "
    f"-s data/dynerf/cut_roasted_beef --port 600{idx} "
    f'--expname "admm_{idx}/cut_roasted_beef" '
    f"--configs arguments/dynerf/cut_roasted_beef_admm_ckpt_{idx}.py "
    '--start_checkpoint "output/dynerf/cut_roasted_beef/chkpnt_fine_14000.pth" '
)

command2 = (
    f"python render{idx}.py "
    f'--model_path "output/admm_{idx}/cut_roasted_beef" '
    "--skip_train --skip_video "
    "--configs arguments/dynerf/cut_roasted_beef.py "
)

command3 = (
    f'python metrics{idx}.py --model_path "output/admm_{idx}/cut_roasted_beef" '
)

# 运行命令
subprocess.run(command1, shell=True, check=True)
subprocess.run(command2, shell=True, check=True)
subprocess.run(command3, shell=True, check=True)


