import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime

def analyze_tensor(tensor, filename="tensor_histogram", save_dir="analysis_result"):
    # 获取统计信息
    shape = str(tensor.shape)
    max_val = tensor.max().item()
    min_val = tensor.min().item()
    mean_val = tensor.mean().item()
    positive_count = torch.sum(tensor > 0).item()
    total_count = tensor.numel()
    positive_ratio = positive_count / total_count * 100

    # 打印信息
    print("Tensor Analysis:")
    print(f"Shape: {shape}")
    print(f"Max Value: {max_val}")
    print(f"Min Value: {min_val}")
    print(f"Mean Value: {mean_val}")
    print(f"Count of elements > 0: {positive_count} ({positive_ratio:.2f}%)")

    # 创建文件夹
    os.makedirs(save_dir, exist_ok=True)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, f"{filename}.png")

    # 绘制直方图
    plt.figure(figsize=(8, 5))
    plt.hist(tensor.cpu().numpy(), bins=30, alpha=0.75, color='blue', edgecolor='black')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Tensor Histogram")
    plt.grid(True)

    # 在图像右上角添加统计信息
    text = (f"Shape: {shape}\n"
            f"Max: {max_val:.4f}\n"
            f"Min: {min_val:.4f}\n"
            f"Mean: {mean_val:.4f}\n"
            f"> 0: {positive_count} ({positive_ratio:.2f}%)")
    plt.text(0.95, 0.95, text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

    # 保存到本地
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图像，避免显示

    print(f"Histogram saved as {save_path}")

