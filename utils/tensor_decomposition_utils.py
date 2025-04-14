import sys
import numpy as np
import random
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer_ms import render, render_with_topk_mask, render_point_time, network_gui, render_with_topk_score
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
from utils.scene_utils import render_training_image
from imp_score_utils import *
import shutil
from tntorch import Tensor as TT


def decom_tt(gaussians, rank_dict, decom_path):
    os.makedirs(decom_path,exist_ok=True)
    # 先拿到 HexPlaneField 对象
    grids = gaussians._deformation.deformation_net.grid.grids
    print(f"[decom_tt] decom_path = {decom_path}")
    for level_1 in range(0, 2):
        for level_2 in range(0, 6):
            target_tensor = grids[level_1][level_2].data  # .data 拿 raw tensor，不带梯度
            # TT分解 rank 可以设为 4~20 之间试试
            # 获取对应的 rank
            rank = rank_dict.get((level_1, level_2), 10)  # 默认 rank=10 如果没有指定
            tt_tensor = TT(target_tensor, ranks_tt=rank)
            file_name = f"tt_cores_grid_{level_1}_{level_2}.pt"
            file_path = os.path.join(decom_path, file_name)
            torch.save(tt_tensor.cores, file_path)
            # print(f"save success {file_path}")
            reconstructed = reconstruct_tt_tensor(tt_tensor, target_tensor.shape).to(target_tensor.device)
            rel_error = (reconstructed - target_tensor).norm() / target_tensor.norm()
            print(f"{level_1}{level_2}rel_error:", rel_error.item())
            # 将重建的张量替换回去
            grids[level_1][level_2].data.copy_(reconstructed)  # 替换 grids 中的原始张量

# 逐步从 tt_tensor 的 TT 核心恢复 full tensor
def reconstruct_tt_tensor(tt_tensor, original_shape):
    result = tt_tensor.cores[0]
    for core in tt_tensor.cores[1:]:
        result = torch.einsum('aib, bjc -> aijc', result, core).reshape(
            result.shape[0], -1, core.shape[2])
    result = result.squeeze(0)
    return result.view(original_shape)