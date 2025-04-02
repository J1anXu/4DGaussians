import subprocess
import os,sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
# 
scene = "cook_spinach"
n = 0

log_dir = "og_log"
log_file = os.path.join(log_dir, f"{scene}.log")
os.makedirs(log_dir, exist_ok=True)

def run_command(command):
    with open(log_file, "a") as log:
        log.write(f"Executing: {command}\n")
        process = subprocess.run(command, shell=True, stdout=log, stderr=log)
        if process.returncode != 0:
            log.write(f"Command failed: {command}\n")
            exit(1)

commands = [
    f"python train_dynerf/train{n}.py -s ../4DGS_ADMM/data/dynerf/{scene} --port 600{n} --expname 'dynerf/{scene}' --configs arguments/dynerf/{scene}.py",
    f"python train_dynerf/render{n}.py --model_path 'output/dynerf/{scene}/'  --skip_train --skip_video --configs arguments/dynerf/{scene}.py",
    f"python train_dynerf/metrics{n}.py --model_path 'output/dynerf/{scene}/'"
]

for cmd in commands:
    run_command(cmd)
