import subprocess
from datetime import datetime
import os
import pytz

idx = 5
scene = "sear_steak"


log_dir = "log"
log_file = os.path.join(log_dir, f"{scene}_admm.log")
os.makedirs(log_dir, exist_ok=True)

def run_command(command):
    with open(log_file, "a") as log:
        log.write(f"Executing: {command}\n")
        process = subprocess.run(command, shell=True, stdout=log, stderr=log)
        if process.returncode != 0:
            log.write(f"Command failed: {command}\n")
            exit(1)


timezone = pytz.timezone("America/Chicago")
current_time = datetime.now(timezone).strftime("%Y-%m-%d_%H-%M-%S")


command1 = (
    f"python trainadmm_with_ckpt_{idx}.py "
    f"-s data/dynerf/{scene} --port 600{idx} "
    f'--expname "admm/{scene}" '
    f"--configs arguments/dynerf/{scene}_admm_ckpt.py "
    f'--start_checkpoint "output/dynerf/{scene}/chkpnt_fine_14000.pth" '
)

command2 = (
    f"python render.py "
    f'--model_path "output/admm/{scene}" '
    "--skip_train --skip_video "
    f"--configs arguments/dynerf/{scene}.py "
)

command3 = (
    f'python metrics.py --model_path "output/admm/{scene}" '
)

commands = [command1,command2,command3]

for cmd in commands:
    run_command(cmd)