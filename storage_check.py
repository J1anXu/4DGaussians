import os

def print_all_file_sizes(folder_path):
    if not os.path.isdir(folder_path):
        print("路径不是一个有效的文件夹")
        return

    print(f"📁 文件夹: {folder_path}\n")
    total_size = 0
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                size = os.path.getsize(filepath)
                total_size += size
                print(f"{filepath}  ——  {size / 1024:.2f} KB")
            except Exception as e:
                print(f"❌ 无法读取 {filepath}: {e}")

    print(f"\n📦 总大小: {total_size / (1024**2):.2f} MB")

# 示例使用


folder = "/data2/jian/4DGS_ADMM/output/dynerf/cut_roasted_beef/point_cloud/iteration_14000"
print_all_file_sizes(folder)


folder = "/data2/jian/4DGS_ADMM/output/admm_3/cut_roasted_beef/point_cloud/iteration_27000"
print_all_file_sizes(folder)


folder = "/data2/jian/4DGS_ADMM/output/admm_quant_3/cut_roasted_beef/point_cloud/iteration_29000"
print_all_file_sizes(folder)