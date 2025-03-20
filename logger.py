import logging
import os
from datetime import datetime
import pytz


def initialize_logger(log_dir="./log", timezone_str="America/Chicago", prefix=""):
    """
    Initializes the logging system with a timestamped log file and a given timezone.

    Parameters:
    log_dir (str): Directory to store log files. Default is './log'.
    timezone_str (str): Timezone for the timestamp. Default is "Etc/GMT+4" (UTC-4).
    """
    # 如果日志系统已经初始化，则不重复初始化
    if logging.getLogger().hasHandlers():
        return

    # 设置时区
    timezone = pytz.timezone(timezone_str)
    current_time = datetime.now(timezone).strftime("%Y-%m-%d_%H-%M-%S")

    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 设置日志文件名
    log_file = os.path.join(log_dir, f"{prefix}_{current_time}.log")

    # 配置日志系统
    logging.basicConfig(
        level=logging.INFO,  # 设置日志记录的最低级别
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            # logging.StreamHandler()  # 控制台输出
        ],
    )
    logging.info("Log system initialized.")
    return current_time
