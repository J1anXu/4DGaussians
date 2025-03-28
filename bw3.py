# -*- coding:utf-8 -*-
# 
# Author: 
# Time: 

import torch
import fnmatch
import numpy as np
import os
from imp_score_utils import time_0_blending_weight,get_unactivate_opacity, get_pruning_iter1_mask,get_pruning_iter2_mask,get_topk_score,norm_tensor_01,render_info_for_each_img

#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
#  交替方向乘子法（ADMM） 优化算法，通常用于解决带有约束和正则化的优化问题。
class BW:
    def __init__(self, gaussians, opt, args, scene, pipe, background):
        self.gaussians = gaussians 
        self.opt = opt # 指定张量存储的位置（CPU 或 GPU）
        self.args = args # 标量，用于 ADMM 中的惩罚项
        # u和z是ADMM算法中的辅助变量（对偶变量）
        self.args = args
        self.scene = scene
        self.pipe = pipe
        self.background = background
        d,w=  render_info_for_each_img(gaussians, opt, args, scene, pipe, background)
        self.dict = d
        self.curr_acc_w  = w
        print(f" init BW success, len = {len(d)}")
        
    # 注意,paper中的a就是这里的opacity
    def update(self, key, value):
        value = value.detach().cpu()
        with torch.no_grad():
            pre = self.dict[key]
            diff = value - pre  # 计算 diff
            self.dict[key] = value  # 更新字典
            self.curr_acc_w.add_(diff)  # 使用 in-place 操作，避免额外显存分配



    def get_curr_acc_w(self):
        with torch.no_grad():
            return norm_tensor_01(self.curr_acc_w)



