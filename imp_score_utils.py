import numpy as np
import torch
from random import randint
from gaussian_renderer import render, render_with_topk_mask, render_with_topk_score, render_point_time
from tqdm import tqdm
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from imp_score_utils import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import concurrent.futures
from utils.parallel_utils import run_tasks_in_parallel
from utils.tensor_analysis_utils import analyze_tensor, plot_2_sum_tensors

import os

def get_topk_mask(gaussians, scene, pipe, args, background, related_gs_num, cache = False):
    tensor_path = args.start_checkpoint + f"_topk_mask_{related_gs_num}.pt"
    # 检查文件是否存在
    if cache and os.path.exists(tensor_path):
        print(f"get topk_mask from {tensor_path}")
        return torch.load(tensor_path)
    viewpoint_stack = scene.getTrainCameras()
    valid_prune_mask = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
    for view in tqdm(viewpoint_stack, desc=f"topk_mask, K={related_gs_num}"):
        # if view.time != 0.0:  # 相较于ImportantScore3 唯一的区别
        #     continue
        with torch.no_grad():
            renderTopk_pkg = render_with_topk_mask(view, gaussians, pipe, background, topk=related_gs_num)
            topk_mask = renderTopk_pkg["topk_mask"]
            valid_prune_mask = torch.logical_or(valid_prune_mask, topk_mask)
    # 计算 valid_prune_mask 中 True 的数量
    num_true = torch.sum(valid_prune_mask).item()
    torch.save(valid_prune_mask, tensor_path)
    return valid_prune_mask

def get_topk_score(gaussians, scene, pipe, args, background, related_gs_num, cache = False):
    tensor_path = args.start_checkpoint + f"top{related_gs_num}_score_.pt"
    # 检查文件是否存在
    if cache and os.path.exists(tensor_path):
        print(f"get topk_score from {tensor_path}")
        return torch.load(tensor_path)
    viewpoint_stack = scene.getTrainCameras()
    final_score = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    for view in tqdm(viewpoint_stack, desc=f"topk_score, K={related_gs_num}"):
        # if view.time != 0.0:  # 相较于ImportantScore3 唯一的区别
        #     continue
        with torch.no_grad():
            renderTopk_pkg = render_with_topk_score(view, gaussians, pipe, background, topk=related_gs_num, score_function = args.score_function)
            topk_score = renderTopk_pkg["topk_score"]
            final_score += topk_score
    # 计算 valid_prune_mask 中 True 的数量
    torch.save(final_score, tensor_path)
    return final_score


def movingLength(gaussians, times):
    # 获取高斯点数量和时间步数量
    num_gaussians = gaussians.get_xyz.shape[0]

    # 初始化累计移动距离张量 (num_gaussians,)
    moving_length_table = torch.zeros(num_gaussians, device="cuda" if gaussians.get_xyz.is_cuda else "cpu")

    # 记录上一时间步的坐标
    prev_means3D = None

    for time in times:
        # 获取当前时间步的高斯点位置
        render_point_time_res = render_point_time(time, gaussians, cam_type=None)
        means3D_at_time_tensor = render_point_time_res["means3D_final"]  # 形状 (num_gaussians, 3)

        if prev_means3D is not None:
            # 计算欧几里得距离 ||x_t - x_{t-1}||
            distance = torch.norm(means3D_at_time_tensor - prev_means3D, dim=1)  # 形状 (num_gaussians,)

            # 累加到移动长度表
            moving_length_table += distance

        # 更新上一时刻的位置
        prev_means3D = means3D_at_time_tensor.clone()

    normalized_moving_length = norm_tensor_01(moving_length_table)
    return normalized_moving_length


def norm_tensor_01(tensor):
    # [0,1]
    min_val = tensor.min()
    max_val = tensor.max()
    eps = 1e-8  # 避免除以零
    normalized_tensor = (tensor - min_val) / (max_val - min_val + eps)
    return normalized_tensor

def norm_tensor_with_clipping(tensor, lower_percentile=1, upper_percentile=99):
    # 计算分位数
    lower_bound = torch.quantile(tensor, lower_percentile / 100)
    upper_bound = torch.quantile(tensor, upper_percentile / 100)
    
    # 截断数据
    clipped_tensor = torch.clamp(tensor, min=lower_bound, max=upper_bound)
    
    # 归一化到 [0,1]
    min_val = clipped_tensor.min()
    max_val = clipped_tensor.max()
    eps = 1e-8  # 避免除以零
    normalized_tensor = (clipped_tensor - min_val) / (max_val - min_val + eps)
    
    return normalized_tensor



def time_0_blending_weight(gaussians, opt: OptimizationParams, args, scene, pipe, background, cache = False):
    tensor_path = args.start_checkpoint + "_t0bw.pt"
    # 检查文件是否存在
    if cache and os.path.exists(tensor_path):
        print("get time_0_bleding_weight from cache")
        return torch.load(tensor_path)

    # 对于相同的ckpt, 结果总是一样的
    imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda() 
    accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()  
    views = scene.getTrainCameras()  

    total_views = len(views)  
    with torch.no_grad():
        for view in tqdm(views, total=total_views, desc="time0BlendingWeight"):
            if view.time != 0.0:  
                continue
            render_pkg = render(view, gaussians, pipe, background)
            accum_weights = render_pkg["accum_weights"]
            area_proj = render_pkg["area_proj"]
            area_max = render_pkg["area_max"]
            accum_area_max = accum_area_max + area_max
            if opt.outdoor == True:
                mask_t = area_max != 0
                temp = imp_score + accum_weights / area_proj
                imp_score[mask_t] = temp[mask_t]
            else:
                imp_score += accum_weights
    imp_score[accum_area_max == 0] = 0  # 对于从未在任何视角中被看到的点，重要性设为 0 ，确保不可见点的 imp_score = 0
    # scores = imp_score / imp_score.sum()  # 归一化 imp_score 作为采样概率 prob
    scores = norm_tensor_01(imp_score)
    torch.save(scores, tensor_path)
    print(f"time_0_blending_weight saved to {tensor_path}")
    non_zero_count = torch.count_nonzero(scores)  # 统计非零元素个数
    print(f"Non-zero count: {non_zero_count}")
    print("imp_score return success")
    return scores

def blending_weight_for_each_img(gaussians, opt: OptimizationParams, args, scene, pipe, background):
    res = {}
    accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()  
    views = scene.getTrainCameras()  
    total_views = len(views)  
    acc_imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda() 

    with torch.no_grad():
        for view in tqdm(views, total=total_views, desc="blending_weight_for_each_img"):
            imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()   
            # if view.time != 0.0:  
            #     continue
            render_pkg = render(view, gaussians, pipe, background)
            accum_weights = render_pkg["accum_weights"]
            area_proj = render_pkg["area_proj"]
            area_max = render_pkg["area_max"]
            accum_area_max = accum_area_max + area_max
            if opt.outdoor == True:
                mask_t = area_max != 0
                temp = imp_score + accum_weights / area_proj
                imp_score[mask_t] = temp[mask_t]
                acc_imp_score+=imp_score
            else:
                imp_score += accum_weights
                acc_imp_score+=imp_score

            # 对于从未在任何视角中被看到的点，重要性设为 0 ，确保不可见点的 imp_score = 0
            imp_score[accum_area_max == 0] = 0  
            res[view.uid] = imp_score
    return res,acc_imp_score


def get_unactivate_opacity(gaussians):
    opacity = gaussians._opacity[:, 0]
    scores = opacity
    return scores

def get_01_opacity(gaussians):
    # activation了的
    opacity = gaussians.get_opacity
    scores = opacity
    return scores

def time_all_blending_weight(gaussians, opt: OptimizationParams, args, scene, pipe, background, cache = False):
    tensor_path = args.start_checkpoint + "_tallbw.pt"
    # 检查文件是否存在
    if cache and os.path.exists(tensor_path):
        print("get time_all_blending_weight from cache")
        return torch.load(tensor_path)
    
    imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()  # 存储每个高斯点的重要性评分，初始为 0
    accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()  # 累积每个点在不同视角下的最大投影面积
    views = scene.getTrainCameras()  # 获取所有训练视角（相机位置）
    total_views = len(views)  # 获取视角总数

    for view in tqdm(views, total=total_views, desc="allTimeBledWeight"):
        render_pkg = render(view, gaussians, pipe, background)
        accum_weights = render_pkg["accum_weights"]
        area_proj = render_pkg["area_proj"]
        area_max = render_pkg["area_max"]
        accum_area_max = accum_area_max + area_max
        if opt.outdoor == True:
            mask_t = area_max != 0
            temp = imp_score + accum_weights / area_proj
            imp_score[mask_t] = temp[mask_t]
        else:
            imp_score += accum_weights

    imp_score[accum_area_max == 0] = 0  # 对于从未在任何视角中被看到的点，重要性设为 0 ，确保不可见点的 imp_score = 0
    # scores = imp_score / imp_score.sum()  # 归一化 imp_score 作为采样概率 prob
    scores = norm_tensor_01(imp_score)
    torch.save(scores, tensor_path)
    print(f"time_all_blending_weight saved to {tensor_path}")
    non_zero_count = torch.count_nonzero(scores)  # 统计非零元素个数
    print(f"Non-zero count: {non_zero_count}")
    print("imp_score return success")
    return scores



def get_pruning_iter1_mask(gaussians, opt, args, scene, pipe, background):

    # topk_score = get_topk_score(gaussians, scene, pipe, args, background, args.related_gs_num, False)
    # bias = norm_tensor_01(topk_score)
    bias = None

    if args.simp_iteration1_score_type == 0:
        scores = time_0_blending_weight(gaussians, opt, args, scene, pipe, background, False)

    elif args.simp_iteration1_score_type == 1:
        scores = time_all_blending_weight(gaussians, opt, args, scene, pipe, background, True)

    elif args.simp_iteration1_score_type == 2:
        scores = get_unactivate_opacity(gaussians)

    elif args.simp_iteration1_score_type == 3:
        scores = get_01_opacity(gaussians).squeeze(1)


    if args.simp_iteration1_score_type == 10: # time_0_blending_weight + bias
        scores = time_0_blending_weight(gaussians, opt, args, scene, pipe, background, True)
        scores = scores + bias

    elif args.simp_iteration1_score_type == 11: # time_all_blending_weight + bias
        scores = time_all_blending_weight(gaussians, opt, args, scene, pipe, background, True)
        scores = scores + bias

    elif args.simp_iteration1_score_type == 12: # unactivate_opacity + bias
        scores = get_unactivate_opacity(gaussians)
        scores = scores + bias

    elif args.simp_iteration1_score_type == 13:  # 01_opacity + bias
        scores = get_01_opacity(gaussians).squeeze(1)
        scores = scores + bias




    if args.simp_iteration1_score_type == 20: # time_0_blending_weight + (1-bias)
        scores = time_0_blending_weight(gaussians, opt, args, scene, pipe, background, True)
        scores = scores + (1-bias)

    elif args.simp_iteration1_score_type == 21: # time_all_blending_weight + (1-bias)
        scores = time_all_blending_weight(gaussians, opt, args, scene, pipe, background, True)
        scores = scores + (1-bias)

    elif args.simp_iteration1_score_type == 22: # unactivate_opacity + (1-bias)
        scores = get_unactivate_opacity(gaussians)
        scores = scores + (1-bias)

    elif args.simp_iteration1_score_type == 23:  # 01_opacity + (1-bias)
        scores = get_01_opacity(gaussians).squeeze(1)
        scores = scores + (1-bias)




    elif args.simp_iteration1_score_type == -1:
        scores = get_topk_score(gaussians, scene, pipe, args, background, args.related_gs_num, True)



    elif args.simp_iteration1_score_type == 4: # time_0_blending_weight * bias
        scores, topk_score = run_tasks_in_parallel(
            (time_0_blending_weight, gaussians, opt, args, scene, pipe, background, True),
            (get_topk_score, gaussians, scene, pipe, args, background, args.related_gs_num, True)
        )
        scores =  scores * bias
    
    elif args.simp_iteration1_score_type == 5: # unactivate_opacity * bias
        scores, topk_score = run_tasks_in_parallel(
            (get_unactivate_opacity, gaussians),
            (get_topk_score, gaussians, scene, pipe, args, background, args.related_gs_num, True)
        )
        scores = scores * bias
    


    elif args.simp_iteration1_score_type == 41: # time_0_blending_weight * (1+bias)
        scores, topk_score = run_tasks_in_parallel(
            (time_0_blending_weight, gaussians, opt, args, scene, pipe, background, True),
            (get_topk_score, gaussians, scene, pipe, args, background, args.related_gs_num, True)
        )
        scores =  scores * (1+bias)
    
    elif args.simp_iteration1_score_type == 51: # unactivate_opacity * (1+bias)
        scores, topk_score = run_tasks_in_parallel(
            (get_unactivate_opacity, gaussians),
            (get_topk_score, gaussians, scene, pipe, args, background, args.related_gs_num, True)
        )
        scores = scores * (1+bias)


    elif args.simp_iteration1_score_type == 42: # time_0_blending_weight * bias**2
        scores, topk_score = run_tasks_in_parallel(
            (time_0_blending_weight, gaussians, opt, args, scene, pipe, background, True),
            (get_topk_score, gaussians, scene, pipe, args, background, args.related_gs_num, True)
        )
        scores =  scores * bias**2
    
    elif args.simp_iteration1_score_type == 52: # unactivate_opacity * bias**2
        scores, topk_score = run_tasks_in_parallel(
            (get_unactivate_opacity, gaussians),
            (get_topk_score, gaussians, scene, pipe, args, background, args.related_gs_num, True)
        )
        scores = scores * bias**2


    scores_sorted, _ = torch.sort(scores, 0)
    threshold_idx = int(opt.opacity_admm_threshold1 * len(scores_sorted))
    abs_threshold = scores_sorted[threshold_idx - 1]
    mask = (scores <= abs_threshold).squeeze()

    return mask

def get_pruning_iter2_mask(gaussians, opt):
    with torch.no_grad():
        scores = get_unactivate_opacity(gaussians)
    scores_sorted, _ = torch.sort(scores, 0)
    threshold_idx = int(opt.opacity_admm_threshold2 * len(scores_sorted))
    abs_threshold = scores_sorted[threshold_idx - 1]
    mask = (scores <= abs_threshold).squeeze()
    return mask