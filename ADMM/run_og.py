import argparse
import subprocess
from datetime import datetime
import pytz

# 设置时区与时间
timezone = pytz.timezone("America/Chicago")
current_time = datetime.now(timezone).strftime("%Y-%m-%d_%H-%M-%S")

# 所有可选 scene 名称（顺序和 index 对应）
scene_list = [
    "cook_spinach",
    "cut_roasted_beef",
    "flame_salmon_1",
    "flame_steak",
    "sear_steak"
]

# 解析命令行参数
parser = argparse.ArgumentParser(description="Run DyNeRF training, rendering, and evaluation")
parser.add_argument("-i", "--index", type=int, required=True, help="Index of the scene (0-based)")

args = parser.parse_args()
idx = args.index
gpu_list = [3, 4, 5]
gpu = gpu_list[(idx - 1) % len(gpu_list)]



# 检查 index 合法性
if idx < 0 or idx >= len(scene_list):
    raise ValueError(f"Invalid index {idx}, must be between 0 and {len(scene_list) - 1}")

scene = scene_list[idx]

# 设置输出目录 这是原始的训练 所以都存在dynerf里
expname = "dynerf"

# 构造命令
command1 = (
    f"CUDA_VISIBLE_DEVICES={gpu} python train.py "
    f"-s data/{expname}/{scene} --port 600{idx} "
    f'--expname "{expname}/{scene}" '
    f"--configs arguments/{expname}/{scene}.py "
)

command2 = (
    f"python render_parallel.py "
    f'--model_path "output/{expname}/{scene}" '
    "--skip_train --skip_video "
    f"--configs arguments/{expname}/{scene}.py "
)

command3 = (
    f'python metrics0.py --model_path "output/{expname}/{scene}" '
)

# 运行命令
print(f"Running training for scene: {scene} (index {idx})")
subprocess.run(command1, shell=True, check=True)
subprocess.run(command2, shell=True, check=True)
subprocess.run(command3, shell=True, check=True)
