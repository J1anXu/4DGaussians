import os
import cv2
import numpy as np

def compute_and_save_diffs(image_folder):
    # 获取所有图片文件并按数字编号排序
    images = sorted(
        [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    
    # 创建存储 diff 结果的文件夹
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 确保输出文件夹与 diff_frame.py 在同一目录下
    output_folder = os.path.join(script_dir, "frameDiffRes")

    # 确保目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    for i in range(len(images) - 1):
        img1_path = os.path.join(image_folder, images[i])
        img2_path = os.path.join(image_folder, images[i + 1])
        
        # 读取图片
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"Skipping {images[i]} or {images[i+1]} due to read error.")
            continue
        
        # 确保两张图片尺寸相同
        if img1.shape != img2.shape:
            print(f"Skipping {images[i]} and {images[i+1]} due to size mismatch.")
            continue
        
        # 计算差异
        diff = cv2.absdiff(img1, img2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # 创建绿色 mask
        mask = np.zeros_like(img1)
        mask[gray_diff > 25] = (0, 255, 0)  # 设定阈值 25，可调整
        
        # 将绿色 mask 叠加到 img2 上
        result = cv2.addWeighted(img2, 1, mask, 1, 0)
        
        # 生成输出文件名
        output_path = os.path.join(output_folder, f"{images[i].split('.')[0]}_{images[i+1].split('.')[0]}.png")
        cv2.imwrite(output_path, result)
        print(f"Saved: {output_path}")

# 使用示例
image_folder = "/data2/jian/4DGaussians/data/dynerf/cut_roasted_beef/cam00/images"  # 替换为实际路径
compute_and_save_diffs(image_folder)
