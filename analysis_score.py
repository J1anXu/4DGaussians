import torch
from impScoreUtils import *

# 加载张量
scores = torch.load("scores.pt")
# 检查数据是否为空
if scores is None:
    print("Error: scores.pt 加载后是 None，可能文件损坏或保存的不是张量。")
    exit()

# 如果数据是字典，打印它的键
if isinstance(scores, dict):
    print("scores.pt 包含多个键:", scores.keys())
    exit()

# 确保数据是张量
if not isinstance(scores, torch.Tensor):
    print(f"Error: 读取的数据类型为 {type(scores)}，而不是 torch.Tensor")
    exit()

# 计算归一化
scores = norm_tensor_01(scores)


# 计算统计信息
shape = scores.shape
mean_val = scores.mean().item()
max_val = scores.max().item()
min_val = scores.min().item()
greater_than_mean = (scores > mean_val).sum().item()

# 打印统计信息
print(f"张量形状: {shape}")
print(f"均值: {mean_val}")
print(f"最大值: {max_val}")
print(f"最小值: {min_val}")
print(f"大于均值的元素数量: {greater_than_mean}")

# 保存统计信息到文件
with open("scores_stats.txt", "w") as f:
    f.write(f"张量形状: {shape}\n")
    f.write(f"均值: {mean_val}\n")
    f.write(f"最大值: {max_val}\n")
    f.write(f"最小值: {min_val}\n")
    f.write(f"大于均值的元素数量: {greater_than_mean}\n")
