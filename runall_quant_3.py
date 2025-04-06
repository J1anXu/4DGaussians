import subprocess
from datetime import datetime
import os
import pytz

timezone = pytz.timezone("America/Chicago")
current_time = datetime.now(timezone).strftime("%Y-%m-%d_%H-%M-%S")

idx = 3
scene = "cut_roasted_beef"
# cook_spinach
# cut_roasted_beef
# flame_salmon_1
# flame_steak
# sear_steak
command1 = (
    f"python train_admm_with_ckpt_quant_{idx}.py "
    f"-s data/dynerf/{scene} --port 600{idx} "
    f'--expname "admm_quant_{idx}/{scene}" '
    f"--configs arguments/dynerf/{scene}_admm_ckpt_quant_{idx}.py "
    f'--start_checkpoint "output/dynerf/{scene}/chkpnt_fine_14000.pth" '
)

command2 = (
    f"python render{idx}.py "
    f'--model_path "output/admm_quant_{idx}/{scene}" '
    "--skip_train --skip_video "
    f"--configs arguments/dynerf/{scene}.py "
)

command3 = (
    f'python metrics{idx}.py --model_path "output/admm_quant_{idx}/{scene}" '
)

# 运行命令
subprocess.run(command1, shell=True, check=True)
subprocess.run(command2, shell=True, check=True)
subprocess.run(command3, shell=True, check=True)

print(">>>>>>>[admm]<<<<<<<")
