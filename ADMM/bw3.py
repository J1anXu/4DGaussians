# -*- coding:utf-8 -*-
# 
# Author: 
# Time: 

import torch
import fnmatch
import numpy as np
import os
from imp_score_utils import rendering_info_for_each_img,norm_tensor_01,blending_weight_for_each_img,norm_tensor_11
from utils.tensor_analysis_utils import analyze_tensor
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
        acc_w_list, acc_w_sum, acc_s_list, acc_s_sum =  rendering_info_for_each_img(gaussians, opt, args, scene, pipe, background)
        self.acc_w_list = acc_w_list
        self.acc_w_sum  = acc_w_sum
        self.acc_s_list = acc_s_list
        self.acc_s_sum = acc_s_sum
        torch.save(acc_s_sum, "acc_s_sum.pt")
        print("BW init success")
        
    # 注意,paper中的a就是这里的opacity
    def update_weights(self, key, value):
        value = value.detach().cpu()
        with torch.no_grad():
            pre = self.acc_w_list[key]
            diff = value - pre  # 计算 diff
            self.acc_w_list[key] = value  # 更新字典
            self.acc_w_sum.add_(diff)  # 使用 in-place 操作，避免额外显存分配

    def update_error_scores(self, key, value):
        value = value.detach().cpu()
        with torch.no_grad():
            pre = self.acc_s_list[key]
            diff = value - pre  # 计算 diff
            self.acc_s_list[key] = value  # 更新字典
            self.acc_s_sum.add_(diff)  # 使用 in-place 操作，避免额外显存分配

    def get_curr_acc_w(self):
        with torch.no_grad():
            return norm_tensor_01(self.acc_w_sum)


    def get_curr_acc_s(self):
        with torch.no_grad():
            return norm_tensor_11(self.acc_s_sum)

    def get_actual_acc_s(self):
        return norm_tensor_11(self.acc_s_sum)