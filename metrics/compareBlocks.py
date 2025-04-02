import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def get_folder_name(image_path):
    """ 获取图片所在文件夹的名称 """
    return os.path.basename(os.path.dirname(image_path))

def preprocess_image(image, n):
    """ 调整图像大小，使其适应 2^N x 2^N 的分割，不剪切，而是用黑色填充 """
    h, w = image.shape[:2]
    new_h, new_w = ((h + (2**n) - 1) // (2**n)) * (2**n), ((w + (2**n) - 1) // (2**n)) * (2**n)
    padded_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)  # 黑色填充
    padded_image[:h, :w] = image
    return padded_image

def split_image(image, n, prefix, output_folder):
    """ 将图像分割成 2^N x 2^N 份并保存 """
    image = preprocess_image(image, n)
    h, w = image.shape[:2]
    rows, cols = 2**n, 2**n
    block_h, block_w = h // rows, w // cols
    blocks = {}
    
    os.makedirs(output_folder, exist_ok=True)
    
    for i in range(rows):
        for j in range(cols):
            block = image[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
            blocks[(i, j)] = block
            filename = os.path.join(output_folder, f"{prefix}_block_{i}_{j}.png")
            cv2.imwrite(filename, block)
    
    return blocks

def compute_ssim(blocks_a, blocks_b):
    """ 计算两张图片对应块的 SSIM """
    ssim_values = {}
    
    for key in blocks_a:
        block_a, block_b = blocks_a[key], blocks_b[key]
        
        if block_a.shape == block_b.shape:
            gray_a = cv2.cvtColor(block_a, cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(block_b, cv2.COLOR_BGR2GRAY)
            ssim_values[key] = ssim(gray_a, gray_b)
        else:
            ssim_values[key] = None  # 形状不匹配
    
    return ssim_values

def main(image_a_path, image_b_path, n):
    """ 主函数 """
    image_a = cv2.imread(image_a_path)
    image_b = cv2.imread(image_b_path)
    
    if image_a is None or image_b is None:
        print("无法加载图像")
        return
    
    folder_name = get_folder_name(image_a_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
    output_folder = os.path.join(script_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    image_a_name = os.path.splitext(os.path.basename(image_a_path))[0]
    image_b_name = os.path.splitext(os.path.basename(image_b_path))[0]
    
    blocks_a = split_image(image_a, n, image_a_name, output_folder)
    blocks_b = split_image(image_b, n, image_b_name, output_folder)
    
    ssim_results = compute_ssim(blocks_a, blocks_b)
    
    result_file = os.path.join(output_folder, "ssim_results.txt")
    with open(result_file, "w") as f:
        for key, value in ssim_results.items():
            result_line = f"Block {key}: SSIM = {value:.4f}\n"
            print(result_line.strip())
            f.write(result_line)
    
    # 拼接并保存对比图
    comparison_folder = os.path.join(output_folder, "comparison")
    os.makedirs(comparison_folder, exist_ok=True)
    
    for key in blocks_a:
        block_a, block_b = blocks_a[key], blocks_b[key]
        combined = np.hstack((block_a, block_b))
        comparison_filename = os.path.join(comparison_folder, f"comparison_block_{key[0]}_{key[1]}.png")
        cv2.imwrite(comparison_filename, combined)

# 示例调用


# 示例调用
path_A = "/data2/jian/4DGaussians/output/dynerf/cut_roasted_beef/video/ours_14000/renders/00000.png"
path_B = "/data2/jian/4DGaussians/output/admm_p_75/cut_roasted_beef/video/ours_20000/draw/00000.png"
main(path_A, path_B, 2)  # N=3，将图片切分为 8x8 份
