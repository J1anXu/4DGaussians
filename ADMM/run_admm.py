import subprocess
from datetime import datetime
import argparse
import pytz

# 设置时区
timezone = pytz.timezone("America/Chicago")
current_time = datetime.now(timezone).strftime("%Y-%m-%d_%H-%M-%S")

# 解析命令行参数
parser = argparse.ArgumentParser()
# 输入scene 
scene_list = [
    "cook_spinach",
    "cut_roasted_beef",
    "flame_salmon_1",
    "flame_steak",
    "sear_steak"
]

# 输入idx 用于选择配置文件
# 1, 2, 3, 4, 5
parser.add_argument('-i', '--index', type=int, required=True, help='Index number')
args = parser.parse_args()

idx = args.index
gpu_list = [3, 4, 5]
gpu = gpu_list[(idx - 1) % len(gpu_list)]
scene = scene_list[idx]



# 检查 index 合法性
if idx < 0 or idx >= len(scene_list):
    raise ValueError(f"Invalid index {idx}, must be between 0 and {len(scene_list) - 1}")

# 设置命令
command1 = (
    f"python train_admm_with_ckpt.py "
    f"-s data/dynerf/{scene} --port 600{idx} " 
    f'--expname "admm_{idx}/{scene}" '
    f"--configs arguments/dynerf/{scene}_admm_ckpt_{idx}.py "
    f'--start_checkpoint "output/dynerf/{scene}/chkpnt_fine_14000.pth" '
)

command2 = (
    f"python render_parallel.py "
    f'--model_path "output/admm_{idx}/{scene}" '
    "--skip_train --skip_video "
    f"--configs arguments/dynerf/{scene}.py "
)

command3 = (
    f'python metrics_parallel.py --model_path "output/admm_{idx}/{scene}" '
)

# 执行命令
subprocess.run(command1, shell=True, check=True)
subprocess.run(command2, shell=True, check=True)
subprocess.run(command3, shell=True, check=True)
