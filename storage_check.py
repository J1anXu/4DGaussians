import os

def print_all_file_sizes(folder_path):
    if not os.path.isdir(folder_path):
        print(f"{folder_path} 不是一个有效的文件夹")
        return

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
    print(f"📁 文件夹: {folder_path} 📦 总大小: {total_size / (1024**2):.2f} MB \n")

# 示例使用


folder = "/data2/jian/4DGS_ADMM/output/dynerf/cook_spinach/point_cloud/iteration_14000"
print_all_file_sizes(folder)


folder = "/data2/jian/4DGS_ADMM/output/admm_2/cook_spinach/point_cloud/iteration_21000"
print_all_file_sizes(folder)

f = "/data2/jian/4DGS_ADMM/output/admm_quant_4/flame_steak/point_cloud/iteration_23001"
print_all_file_sizes(f)

f = "/data2/jian/4DGS_ADMM/output/admm_1/cook_spinach/point_cloud/iteration_14000/decom"
print_all_file_sizes(f)


