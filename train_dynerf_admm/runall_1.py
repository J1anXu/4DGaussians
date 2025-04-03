import subprocess
from datetime import datetime
import os
import pytz
import subprocess
import os,sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
# 
timezone = pytz.timezone("America/Chicago")
current_time = datetime.now(timezone).strftime("%Y-%m-%d_%H-%M-%S")

idx = 1
scene = "cut_roasted_beef"

command1 = (
    f"python train_dynerf_admm/train_admm_with_ckpt_{idx}.py "
    f"-s data/dynerf/{scene} --port 600{idx} "
    f'--expname "admm/{scene}" '
    f"--configs arguments/dynerf/{scene}_admm_ckpt.py "
    f'--start_checkpoint "output/dynerf/{scene}/chkpnt_fine_14000.pth" '
)

command2 = (
    f"python render{idx}.py "
    f'--model_path "output/admm/{scene}" '
    "--skip_train --skip_video "
    f"--configs arguments/dynerf/{scene}_admm_ckpt.py "
)

command3 = (
    f'python metrics{idx}.py --model_path "output/admm/{scene}" '
)

# 运行命令
subprocess.run(command1, shell=True, check=True)
subprocess.run(command2, shell=True, check=True)
subprocess.run(command3, shell=True, check=True)


