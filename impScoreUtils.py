import numpy as np
import torch
from random import randint
from gaussian_renderer import render, render_topk_mask, render_point_time, render_topk_score
from tqdm import tqdm
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import concurrent.futures
import threading
import queue


@torch.no_grad()
def topk_gs_of_pixels_mask(gaussians, scene, pipe, background, related_gs_num):
    viewpoint_stack = scene.getTrainCameras()
    valid_prune_mask = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
    for view in tqdm(viewpoint_stack, desc=f"CalTopKGSOfPixels, K={related_gs_num}"):
        # if view.time != 0.0:  # 相较于ImportantScore3 唯一的区别
        #     continue
        renderTopk_pkg = render_topk_mask(view, gaussians, pipe, background, topk=related_gs_num)
        topk_mask = renderTopk_pkg["topk_mask"]
        valid_prune_mask = torch.logical_or(valid_prune_mask, topk_mask)
    # 计算 valid_prune_mask 中 True 的数量
    num_true = torch.sum(valid_prune_mask).item()
    print(f"Number of True values in valid_prune_mask: {num_true}")
    return valid_prune_mask


@torch.no_grad()
def topk_gs_of_pixels_score(gaussians, scene, pipe, background, related_gs_num):
    viewpoint_stack = scene.getTrainCameras()
    valid_prune_mask = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    for view in tqdm(viewpoint_stack, desc=f"CalTopKGSScoreOfPixels,K={related_gs_num}"):
        # if view.time != 0.0:  # 相较于ImportantScore3 唯一的区别
        #     continue
        renderTopk_pkg = render_topk_score(view, gaussians, pipe, background, topk=related_gs_num)
        topk_scores = renderTopk_pkg["topk_scores"]
        valid_prune_mask += topk_scores
    # 计算 valid_prune_mask 中 True 的数量
    num_true = torch.sum(valid_prune_mask).item()
    print(f"Number of True values in valid_prune_mask: {num_true}")
    return valid_prune_mask


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


def time_0_bleding_weight(gaussians, opt: OptimizationParams, scene, pipe, background, views=None):
    # render 输入数据的所有时间和相机视角,累计高斯点的权重
    # 根据重要性评分（Importance Score）对 3D Gaussians 进行稀疏化（Pruning），以减少不重要的点，提高渲染效率
    imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()  # 存储每个高斯点的重要性评分，初始为 0
    accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()  # 累积每个点在不同视角下的最大投影面积
    if views is None:
        views = scene.getTrainCameras()  # 获取所有训练视角（相机位置）
    total_views = len(views)  # 获取视角总数
    # 从第一个数开始，每隔important_score_4_time_interval个取一个
    # time_samples = views.dataset.image_times[:: opt.important_score_4_time_interval]  #
    # print(f"Time samples: {time_samples}")
    for view in tqdm(views, total=total_views, desc="CalTime0BlendingWeight"):
        # if view.time not in time_samples:  # 相较于ImportantScore3 唯一的区别
        if view.time != 0.0:  # 相较于ImportantScore3 唯一的区别
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

    non_zero_count = torch.count_nonzero(scores)  # 统计非零元素个数
    print(f"Non-zero count: {non_zero_count}")
    print("imp_score return success")
    return scores


def getOpacityScore(gaussians):
    opacity = gaussians._opacity[:, 0]
    scores = opacity
    return scores


def allTimeBledWeight(gaussians, opt: OptimizationParams, scene, pipe, background):
    # render 输入数据的所有时间和相机视角,累计高斯点的权重
    # 根据重要性评分（Importance Score）对 3D Gaussians 进行稀疏化（Pruning），以减少不重要的点，提高渲染效率
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

    non_zero_count = torch.count_nonzero(scores)  # 统计非零元素个数
    print(f"Non-zero count: {non_zero_count}")
    print("imp_score return success")
    return scores


def cal_scores_1(gaussians, opt, scene, pipe, background, topk):
    scores_queue = queue.Queue()
    related_gs_queue = queue.Queue()

    # 定义第一个线程
    def thread1():
        scores = time_0_bleding_weight(gaussians, opt, scene, pipe, background)
        scores_queue.put(scores)  # 把结果放入队列

    # 定义第二个线程
    def thread2():
        related_gs_mask = topk_gs_of_pixels_mask(gaussians, scene, pipe, background, topk)
        related_gs_queue.put(related_gs_mask)  # 把结果放入队列

    # 创建并启动线程
    t1 = threading.Thread(target=thread1)
    t2 = threading.Thread(target=thread2)
    t1.start()
    t2.start()
    # 等待线程完成
    t1.join()
    t2.join()
    # 从队列中获取线程结果
    scores = scores_queue.get()
    related_gs_mask = related_gs_queue.get()
    # 线程完成后，可以安全地使用 scores 和 related_gs_mask
    max_score = torch.max(scores)
    print(f"Max score: {max_score}")
    # scores = zeroTimeBledWeight(gaussians, opt, scene, pipe, background)
    # related_gs_mask = get_related_gs(gaussians, scene, pipe, background, args.related_gs_num)
    max_score = torch.max(scores)
    scores[related_gs_mask] += max_score
    return scores
