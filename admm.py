# -*- coding:utf-8 -*-
# 
# Author: 
# Time: 

import torch
import fnmatch
import numpy as np
import os

#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
#  交替方向乘子法（ADMM） 优化算法，通常用于解决带有约束和正则化的优化问题。
class ADMM:
    def __init__(self, gsmodel, rho, device):
        self.gsmodel = gsmodel 
        self.device = device # 指定张量存储的位置（CPU 或 GPU）
        self.init_rho = rho # 标量，用于 ADMM 中的惩罚项
        # u和z是ADMM算法中的辅助变量（对偶变量）
        self.u = {}
        self.z = {}
        self.rho = self.init_rho
        opacity = self.gsmodel.get_opacity
        self.u = torch.zeros(opacity.shape).to(device)
        self.z = torch.Tensor(opacity.data.cpu().clone().detach()).to(device)

    # 注意,paper中的a就是这里的opacity
    def update(self, opt, update_u = True):
        # 先更新辅助变量
        z = self.gsmodel.get_opacity + self.u # z = a + λ (Update z via Eq. 16) 
        # 裁剪z,实现h(z)的映射
        self.z = torch.Tensor(self.prune_z(z, opt)).to(self.device) # z ← proxh(a + λ)
        # 更新高斯乘子
        if update_u: # λ ← λ + a − z.
            with torch.no_grad():
                diff =  self.gsmodel.get_opacity - self.z
                self.u += diff
 
    #  该方法根据不同的策略（由 opt 参数控制）来更新 z：
    def prune_z(self, z, opt):
        z_update = self.metrics_sort(z,opt)  
        return z_update

    def get_admm_loss(self, loss): # 该方法将 ADMM 的惩罚项添加到损失函数中
        return 0.5 * self.rho * (torch.norm(self.gsmodel.get_opacity - self.z + self.u, p=2)) ** 2

    def adjust_rho(self, epoch, epochs, factor=5): # 根据训练的当前进度（epoch 和 epochs）调整 rho 值。通常，rho 会随着训练的进展而增大，从而增加对约束的惩罚
        if epoch > int(0.85 * epochs):
            self.rho = factor * self.init_rho
    
    def metrics_sort(self, z,opt): # 根据透明度的排序值来更新 z。它通过将透明度按升序排序并应用一个阈值来选择透明度值
        index = int(opt.opacity_admm_threshold2 * len(z))
        z_sort = {}
        z_update = torch.zeros(z.shape)
        z_sort, _ = torch.sort(z, 0)
        z_threshold = z_sort[index-1]
        z_update= ((z > z_threshold) * z)  
        return z_update
    
    def metrics_sample(self, z, opt): # 该方法根据透明度值的相对权重进行随机采样。首先，将透明度值归一化为概率分布，并按此分布随机选择样本。
        index = int((1 - opt.opacity_admm_threshold2) * len(z))
        prob = z / torch.sum(z)
        prob = prob.reshape(-1).cpu().numpy()
        indices = torch.tensor(np.random.choice(len(z), index, p = prob, replace=False))
        expand_indices = torch.zeros(z.shape[0] - len(indices)).int()
        indices = torch.cat((indices, expand_indices),0).to(self.device)
        z_update = torch.zeros(z.shape).to(self.device)
        z_update[indices] = z[indices]
        return z_update

    def metrics_imp_score(self, z, imp_score, opt): # 该方法基于重要性分数（imp_score）更新 z。重要性分数低于某个阈值的透明度值会被置为 0，从而对重要性较低的部分进行稀疏化。
        index = int(opt.opacity_admm_threshold2 * len(z))
        imp_score_sort = {}
        imp_score_sort, _ = torch.sort(imp_score, 0)
        imp_score_threshold = imp_score_sort[index-1]
        indices = imp_score < imp_score_threshold 
        z[indices == 1] = 0  
        return z        


