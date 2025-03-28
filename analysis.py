
import torch
from utils.tensor_analysis_utils import analyze_tensor
# 从本地文件加载张量
loaded_tensor = torch.load("time_0_blending_weight.pt")

# 确保加载后数据一致
print(loaded_tensor.shape)  # torch.Size([25801])
analyze_tensor(loaded_tensor,"time_0_blending_weight")

tensor_z = loaded_tensor


# Z-score 标准化	
mean = tensor_z.mean()
std = tensor_z.std()
normalized_z = (tensor_z - mean) / std



# Z-score 标准化 + Tanh  ---->[-1, 1]
normalized_z_tanh = torch.tanh(normalized_z)

# (Z-score 标准化 + Tanh + 1)/ 2 --->[0,1]
normalized_z_tanh_norm = (normalized_z_tanh+1)/2
# 对数缩放
c = 1
normalized_log = (torch.log(tensor_z + c) -10)/10




analyze_tensor(tensor_z,"tensor_z")
analyze_tensor(normalized_z,"normalized_z")
analyze_tensor(normalized_z_tanh,"normalized_z_tanh")
analyze_tensor(normalized_z_tanh_norm,"normalized_z_tanh_norm")
analyze_tensor(normalized_log,"normalized_log")
