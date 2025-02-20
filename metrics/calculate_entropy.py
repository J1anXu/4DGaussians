import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.stats import entropy

def calculate_entropy(image_chunk):
    """计算图像块的熵值，作为信息量的度量"""
    # 转为灰度图像
    image_chunk_gray = image_chunk.convert('L')
    # 将图片转换为一个numpy数组
    image_array = np.array(image_chunk_gray).flatten()
    # 计算灰度值的频率分布
    hist, _ = np.histogram(image_array, bins=256, range=(0, 256), density=True)
    # 去除零值，以便计算熵
    hist = hist[hist > 0]
    # 计算熵
    return entropy(hist)

def process_image(image_path_A, image_path_B, num_blocks_x, num_blocks_y):
    # 打开图像 A（用于计算）和图像 B（用于绘制）
    image_A = Image.open(image_path_A)
    image_B = Image.open(image_path_B)
    image_width, image_height = image_A.size

    # 计算每个块的大小
    block_width = image_width // num_blocks_x
    block_height = image_height // num_blocks_y

    # 创建一个新的图片用于绘制标记（基于 B 图）
    output_image = image_B.copy()
    draw = ImageDraw.Draw(output_image)
    
    font = ImageFont.load_default()  # 默认字体，如果需要其他字体可以指定

    # 切分图像 A 并计算每块的信息量
    for i in range(num_blocks_x):
        for j in range(num_blocks_y):
            # 计算每个块的区域
            left = i * block_width
            upper = j * block_height
            right = left + block_width
            lower = upper + block_height

            # 确保最后一块区域覆盖完整个图像（防止因除不尽产生遗漏）
            right = min(right, image_width)
            lower = min(lower, image_height)

            box = (left, upper, right, lower)
            image_chunk_A = image_A.crop(box)

            # 计算该块的熵值
            block_entropy = calculate_entropy(image_chunk_A)

            # 在 B 图上绘制一个矩形框并标注信息量（熵值）
            draw.rectangle(box, outline="red", width=1)
            text = f"{block_entropy:.2f}"
            draw.text((left + 5, upper + 5), text, fill="white", font=font)

    # 获取当前文件所在的路径
    current_folder = os.path.dirname(os.path.abspath(__file__))
    # 构建保存文件的路径
    output_path = os.path.join(current_folder, 'output_image.png')

    # 保存新的图片
    output_image.save(output_path)
    print(f"输出图片已保存：{output_path}")

# 示例使用
image_path_A = '/data2/jian/4DGaussians/output/dynerf/cut_roasted_beef/video/ours_14000/renders/00000.png'  # 用于计算熵的图片 A
image_path_B = '/data2/jian/4DGaussians/output/admm_p_75/cut_roasted_beef/video/ours_20000/draw/00000.png'  # 用于绘制标记的图片 B
num_blocks_x = 20  # 切分成20块（x轴方向）
num_blocks_y = 20  # 切分成20块（y轴方向）
process_image(image_path_A, image_path_B, num_blocks_x, num_blocks_y)
