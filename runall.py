import subprocess
from datetime import datetime
import os
import pytz

timezone = pytz.timezone("America/Chicago")
current_time = datetime.now(timezone).strftime("%Y-%m-%d_%H-%M-%S")


command1 = (
    f"python trainadmm_with_ckpt2.py "
    "-s data/dynerf/cut_roasted_beef --port 6018 "
    '--expname "admm_p_50/cut_roasted_beef" '
    "--configs arguments/dynerf/cut_roasted_beef_admm_ckpt_spa.py "
    '--start_checkpoint "output/dynerf/cut_roasted_beef/chkpnt_fine_14000.pth" '
)

command2 = (
    f"python render2.py "
    '--model_path "output/admm_p_50/cut_roasted_beef" '
    "--skip_train --skip_video "
    "--configs arguments/dynerf/cut_roasted_beef.py "
)

command3 = (
    f'python metrics2.py --model_path "output/admm_p_50/cut_roasted_beef" '
)

# 运行命令
subprocess.run(command1, shell=True, check=True)
subprocess.run(command2, shell=True, check=True)
subprocess.run(command3, shell=True, check=True)


