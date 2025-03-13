import os
import cv2
import numpy as np
from tqdm import tqdm

def compare_images(folder1, folder2):
    # 创建存储 diff 结果的文件夹
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 确保输出文件夹与 diff_frame.py 在同一目录下
    output_folder = os.path.join(script_dir, "diff_res")
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图片文件并按数字编号排序
    images1 = sorted(
        [f for f in os.listdir(folder1) if f.endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    images2 = sorted(
        [f for f in os.listdir(folder2) if f.endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    print(f"img1 nums = {len(images1)}")
    print(f"img2 nums = {len(images2)}")

    
    for img_name in tqdm(images1, desc="Comparing images"):
        img1_path = os.path.join(folder1, img_name)
        img2_path = os.path.join(folder2, "0"+img_name)
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"Skipping {img_name}, could not load one of the images.")
            continue
        
        if img1.shape != img2.shape:
            print(f"Skipping {img_name}, images have different sizes.")
            continue
        
        # 计算像素差异
        diff = cv2.absdiff(img1, img2)
        mask = np.any(diff > 10, axis=-1)  # 设定阈值 10，超过视为不同
        
        # 标记不同像素为红色
        img_result = img1.copy()
        img_result[mask] = [0, 0, 255]  # 红色标记
        
        # 保存结果
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, img_result)
    
    print("Comparison complete! Results saved in", output_folder)

# 示例使用
renders = "/data2/jian/4DGaussians/output/admm_p_50/cut_roasted_beef/test/ours_30000/renders"
ground = "/data2/jian/4DGaussians/data/dynerf/cut_roasted_beef/cam00/images"
compare_images(ground, renders)