import torch

def analyze_pth(pth_path):
    state_dict = torch.load(pth_path, map_location='cpu')

    if not isinstance(state_dict, dict):
        print("这可能是一个完整模型，而不是 state_dict，建议用 torch.save(model.state_dict()) 保存。")
        return

    print(f"\n分析文件: {pth_path}")
    total_params = 0
    total_size_bytes = 0

    print("\n参数结构和大小（按层）：\n")
    for name, param in state_dict.items():
        numel = param.numel()
        size_bytes = param.element_size() * numel
        total_params += numel
        total_size_bytes += size_bytes

        shape_str = str(tuple(param.shape))
        print(f"{name:<50}  shape={shape_str:<20}  size={size_bytes / (1024**2):.2f} MB")

    print(f"\n📦 总参数数量：{total_params:,}")
    print(f"📦 总大小：{total_size_bytes / (1024**2):.2f} MB")

# 用法示例
analyze_pth("/data2/jian/4DGS_ADMM/output/dynerf/cook_spinach/point_cloud/iteration_14000/deformation.pth")
