import subprocess
import os
import logging

# 设置 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式
    handlers=[logging.StreamHandler()],  # 输出到控制台
)

logger = logging.getLogger(__name__)  # 获取 logger


def run_command(command):
    """运行命令并等待其完成"""
    logger.info(f"Executing: {command}")
    result = subprocess.run(command, shell=True, check=True)
    if result.returncode == 0:
        logger.info(f"Successfully executed: {command}")
    else:
        logger.error(f"Error occurred while executing: {command}")


if __name__ == "__main__":
    # 第一个命令
    command1 = (
        "python trainadmm_with_ckpt2.py "
        "-s data/dynerf/cut_roasted_beef --port 6018 "
        '--expname "admm_p_50/cut_roasted_beef" '
        "--configs arguments/dynerf/cut_roasted_beef_admm_ckpt_spa.py "
        '--start_checkpoint "output/dynerf/cut_roasted_beef/chkpnt_fine_14000.pth"'
    )

    # 第二个命令
    command2 = (
        "python render2.py "
        '--model_path "output/admm_p_50/cut_roasted_beef" '
        "--skip_train --skip_video "
        "--configs arguments/dynerf/cut_roasted_beef.py"
    )

    # 第三个命令
    command3 = 'python metrics2.py --model_path "output/admm_p_50/cut_roasted_beef"'

    # 顺序执行命令
    run_command(command1)
    run_command(command2)
    run_command(command3)
