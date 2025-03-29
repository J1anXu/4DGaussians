
import torch
import os
import shutil
from utils.tensor_analysis_utils import analyze_tensor
from imp_score_utils import norm_tensor_01
# 从本地文件加载张量




def analysis(t,fila_name):
    # Z-score 标准化	
    mean = t.mean()
    std = t.std()
    normalized_z = (t - mean) / std



    # Z-score 标准化 + Tanh  ---->[-1, 1]
    normalized_z_tanh = torch.tanh(normalized_z)

    # (Z-score 标准化 + Tanh + 1)/ 2 --->[0,1]
    normalized_z_tanh_norm = (normalized_z_tanh+1)/2

    # 对数缩放
    c = 1
    normalized_log = (torch.log(t + c) -10)/10




    analyze_tensor(t,fila_name)
    analyze_tensor(norm_tensor_01(t), fila_name+"_norm_tensor_01")
    analyze_tensor(normalized_z,fila_name+"_norm")
    analyze_tensor(normalized_z_tanh,fila_name+"_z_tanh")
    analyze_tensor(normalized_z_tanh_norm,fila_name+"_z_tanh_shift")
    analyze_tensor(normalized_log,fila_name+"_log")


if __name__ == "__main__":
    dir_name = "analysis_result"

    shutil.rmtree(dir_name)  # 删除文件夹及其所有内容
    os.makedirs(dir_name, exist_ok=True)  # 如果文件夹已存在，不会抛出异常
    fila_name = "acc_s_sum"
    loaded_tensor = torch.load(f"{fila_name}.pt")
    analysis(1-loaded_tensor,"1-error")

    fila_name = "acc_s_sum"
    loaded_tensor = torch.load(f"{fila_name}.pt")
    analysis(loaded_tensor,"error")

    fila_name = "opacity"
    loaded_tensor = torch.load(f"{fila_name}.pt")
    analysis(loaded_tensor,fila_name)