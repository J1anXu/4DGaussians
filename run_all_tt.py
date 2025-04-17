import subprocess
from datetime import datetime
import os
import pytz

timezone = pytz.timezone("America/Chicago")
current_time = datetime.now(timezone).strftime("%Y-%m-%d_%H-%M-%S")

idx = 3
# cook_spinach
# cut_roasted_beef
# flame_salmon_1
# flame_steak
# sear_steak
scene = "cook_spinach"
expname = "dynerf_tt"
command1 = (
    f"python train.py "
    f"-s data/dynerf/{scene} --port 600{idx} "
    f'--expname "{expname}/{scene}" '
    f"--configs arguments/dynerf/{scene}.py "
)

command2 = (
    f"python render_parallel.py "
    f'--model_path "output/{expname}/{scene}" '
    "--skip_train --skip_video "
    f"--configs arguments/dynerf/{scene}.py "
)

command3 = (
    f'python metrics0.py --model_path "output/{expname}/{scene}" '
)

# 运行命令
##subprocess.run(command1, shell=True, check=True)
subprocess.run(command2, shell=True, check=True)
#subprocess.run(command3, shell=True, check=True)


