import subprocess
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"


def run_command(command):
    """运行命令并等待其完成"""
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True, check=True)
    if result.returncode == 0:
        print(f"Successfully executed: {command}")
    else:
        print(f"Error occurred while executing: {command}")


if __name__ == "__main__":
    # 第一个命令
    command1 = (
        "python trainadmm_with_ckpt2.py "
        "-s data/dynerf/cut_roasted_beef --port 6002 "
        '--expname "admm_p_50/cut_roasted_beef_2" '
        "--configs arguments/dynerf/cut_roasted_beef_admm_ckpt1.py "
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
