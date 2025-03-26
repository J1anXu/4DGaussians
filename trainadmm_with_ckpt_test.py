#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torchvision import transforms

import numpy as np
import random
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, render_with_topk_mask, render_point_time, network_gui
import sys
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
import matplotlib.pyplot as plt

import copy
from admm import ADMM
import wandb
from logger import initialize_logger
import logging


WANDB = False

# 检查文件夹是否存在，如果不存在则创建
DIR = "diff_analysis"
if not os.path.exists(DIR):
    os.makedirs(DIR)

current_time = initialize_logger()
to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

if WANDB:
    wandb.login()


def scene_reconstruction(
    dataset,
    opt: OptimizationParams,
    hyper,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    gaussians: GaussianModel,
    scene: Scene,
    stage,
    tb_writer,
    train_iter,
    timer,
):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            return
        if stage in checkpoint:
            print("checkpoint load!")
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = None
    first_iter += 1
    train_cams = scene.getTrainCameras()

    if not viewpoint_stack and not opt.dataloader:
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)

    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack, batch_size=batch_size, sampler=sampler, num_workers=16, collate_fn=list
            )
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn=list
            )
            random_loader = True
        loader = iter(viewpoint_stack_loader)

    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack, 0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False

    if opt.dataloader and not load_in_memory:
        try:
            viewpoint_cams = next(loader)
        except StopIteration:
            print(f"reset dataloader into random dataloader. iter=[{iteration}]")
            if not random_loader:
                viewpoint_stack_loader = DataLoader(
                    viewpoint_stack, batch_size=opt.batch_size, shuffle=True, num_workers=32, collate_fn=list
                )
                random_loader = True
            loader = iter(viewpoint_stack_loader)

    else:
        idx = 0
        viewpoint_cams = []
        batch_size = 1
        while idx < batch_size:
            view = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            if not viewpoint_stack:
                viewpoint_stack = temp_list.copy()
            viewpoint_cams.append(view)
            idx += 1

    view = viewpoint_cams[0]
    print(f"time: {view.time}")
    render_pkg = render(view, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type)
    image_1 = render_pkg["render"]
    if scene.dataset_type != "PanopticSports":
        gt_image = view.original_image.cuda()
    else:
        gt_image = view["image"].cuda()

    ## 手动定制一个mark
    # _, height, width = gt_image.shape
    # mid_col = width // 2
    # mark_pixels_image = gt_image
    # mark_pixels_image[:] = 0  # 将所有像素值设置为 [0, 0, 0]
    # mark_pixels_image[:, :, mid_col - 5 : mid_col + 5] = 1  # 设置中间10列的像素为 [0, 0, 0]

    count_ones = torch.sum(view.mark_pixels_image == 1).item()
    print(f"Number of related pixels: {count_ones}")

    # 保存一下图像
    mark_pixels_numpy = view.mark_pixels_image.permute(1, 2, 0).cpu().numpy()
    original_image_numpy = view.original_image.permute(1, 2, 0).cpu().numpy()
    if mark_pixels_numpy.max() > 1:
        mark_pixels_numpy = mark_pixels_numpy / 255.0  # 归一化到 [0,1]
    if original_image_numpy.max() > 1:
        original_image_numpy = original_image_numpy / 255.0  # 归一化到 [0,1]
    plt.imsave(os.path.join(DIR, "mark_pixels_image.png"), mark_pixels_numpy)
    plt.imsave(os.path.join(DIR, "gt_image.png"), original_image_numpy)

    with torch.no_grad():
        renderTopk_pkg = render_with_topk_mask(view, gaussians, pipe, background, topk=50)
        topk_mask = renderTopk_pkg["topk_mask"]
        # 统计 True 的数量
        count = torch.sum(topk_mask).item()  # 使用 torch.sum() 计算 True 的数量
        print(f"maskSize =======================: {count}")
        gaussians.prune_points(~topk_mask)
        # 只保留指定高斯,再渲染一次
        render_pkg_2 = render(view, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type)
        image_2 = render_pkg_2["render"]
        image = image_2.detach().cpu().permute(1, 2, 0).numpy()  # 形状变为 (1014, 1352, 3)
        image = (image - image.min()) / (image.max() - image.min())
        plt.imsave(os.path.join(DIR, "pixels_relate_gs.png"), image)

    rendering = image_1
    means3D_final = render_pkg_2["means3D_final"]
    # perspective projection (modified from cuda code)
    xyz = means3D_final + 0

    device = xyz.device  # 获取 xyz 的设备
    # 将 full_proj_transform 移动到与 xyz 相同的设备
    full_proj_transform = view.full_proj_transform.to(device)
    world_view_transform = view.world_view_transform.to(device)
    rgb = SH2RGB(gaussians._features_dc + 0)[:, 0]
    p_hom = torch.matmul(xyz, full_proj_transform[:3]) + full_proj_transform[3:4]
    p_w = 1.0 / (p_hom[:, 3] + 0.0000001)
    p_proj = p_hom[:, :3] * p_w[:, None]
    p_view = torch.matmul(xyz, world_view_transform[:3, :3]) + world_view_transform[3:4, :3]
    mask = p_view[:, 2].cpu().numpy() > 0.2
    point_image = ndc2Pix(p_proj[:, 0], rendering.shape[2]), ndc2Pix(p_proj[:, 1], rendering.shape[1])
    point_image = torch.cat((point_image[0][:, None], point_image[1][:, None]), -1)
    points = point_image.detach().cpu().numpy()[mask]
    colors = rgb.detach().cpu().numpy()[mask]

    # tune point size for better visualization 0.3, 0.3, 1.2
    image_proj = draw_points_on_image(
        points, np.zeros(colors.shape) + [255, 255, 255], rendering.permute(1, 2, 0).detach().cpu().numpy(), size=0.3
    )
    # 创建转换函数
    transform = transforms.ToTensor()

    # 将 PIL 图像转换为 Tensor
    draw_image = transform(image_proj)  # pil_image 是你的 PIL.Image 对象
    # image_proj.save(f"{draw_path}/{idx}.jpg")
    image = draw_image.detach().cpu().permute(1, 2, 0).numpy()  # 形状变为 (1014, 1352, 3)
    image = (image - image.min()) / (image.max() - image.min())
    plt.imsave(os.path.join(DIR, "pic_and_mark_pixel.png"), image)

    print("success")


C0 = 0.28209479177387814


def SH2RGB(sh):
    return sh * C0 + 0.5


def ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def getOpacityScore(gaussians):
    opacity = gaussians._opacity[:, 0]
    scores = opacity
    return scores


from PIL import Image, ImageDraw


def draw_points_on_image(points, colors, image, size=1):
    image[image > 1] = 1
    image[image < 0] = 0
    image = Image.fromarray((image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for point, color in zip(points, colors):
        x = point[0]
        y = point[1]

        r, g, b = color
        draw.ellipse((x - size, y - size, x + size, y + size), fill=(int(r), int(g), int(b)))
    return image


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


def zeroTimeBledWeight(gaussians, opt: OptimizationParams, scene, pipe, background):
    # render 输入数据的所有时间和相机视角,累计高斯点的权重
    # 根据重要性评分（Importance Score）对 3D Gaussians 进行稀疏化（Pruning），以减少不重要的点，提高渲染效率
    imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()  # 存储每个高斯点的重要性评分，初始为 0
    accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()  # 累积每个点在不同视角下的最大投影面积
    views = scene.getTrainCameras()  # 获取所有训练视角（相机位置）
    total_views = len(views)  # 获取视角总数
    # 从第一个数开始，每隔important_score_4_time_interval个取一个
    # time_samples = views.dataset.image_times[:: opt.important_score_4_time_interval]  #
    # print(f"Time samples: {time_samples}")
    for view in tqdm(views, total=total_views, desc="init_blending_weight"):
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


@torch.no_grad()
def get_topk_mask(gaussians, scene, pipe, background):
    viewpoint_stack = scene.getTrainCameras()
    valid_prune_mask = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
    for view in tqdm(viewpoint_stack, desc="top k"):
        # if view.time != 0.0:  # 相较于ImportantScore3 唯一的区别
        #     continue
        renderTopk_pkg = render_with_topk_mask(view, gaussians, pipe, background, topk=20)
        topk_mask = renderTopk_pkg["topk_mask"]
        valid_prune_mask = torch.logical_or(valid_prune_mask, topk_mask)

    # 计算 valid_prune_mask 中 True 的数量
    num_true = torch.sum(valid_prune_mask).item()
    print(f"Number of True values in valid_prune_mask: {num_true}")

    return ~valid_prune_mask


def training(
    dataset,
    hyper,
    opt: OptimizationParams,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    expname,
):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)

    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()
    # 确保目录存在
    hp_path = os.path.join(args.model_path, "opt_params.pth")
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(opt, hp_path)
    scene_reconstruction(
        dataset,
        opt,
        hyper,
        pipe,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        debug_from,
        gaussians,
        scene,
        "coarse",
        tb_writer,
        opt.coarse_iterations,
        timer,
    )
    scene_reconstruction(
        dataset,
        opt,
        hyper,
        pipe,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        debug_from,
        gaussians,
        scene,
        "fine",
        tb_writer,
        opt.iterations,
        timer,
    )


def prepare_output_and_logger(expname):
    if not args.model_path:
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    stage,
    dataset_type,
):
    if tb_writer:
        tb_writer.add_scalar(f"{stage}/train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar(f"{stage}/train_loss_patchestotal_loss", loss.item(), iteration)
        tb_writer.add_scalar(f"{stage}/iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        #
        validation_configs = (
            {
                "name": "test",
                "cameras": [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)],
            },
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, stage=stage, cam_type=dataset_type, *renderArgs)[
                            "render"
                        ],
                        0.0,
                        1.0,
                    )
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(
                                stage + "/" + config["name"] + "_view_{}/render".format(viewpoint.image_name),
                                image[None],
                                global_step=iteration,
                            )
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(
                                    stage + "/" + config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                    gt_image[None],
                                    global_step=iteration,
                                )
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask

                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/" + config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(stage + "/" + config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_scalar(f"{stage}/total_points", scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(
                f"{stage}/deformation_rate",
                scene.gaussians._deformation_table.sum() / scene.gaussians.get_xyz.shape[0],
                iteration,
            )
            tb_writer.add_histogram(
                f"{stage}/scene/motion_histogram",
                scene.gaussians._deformation_accum.mean(dim=-1) / 100,
                iteration,
                max_bins=500,
            )
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        torch.cuda.empty_cache()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    #
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default="None")
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    args.save_iterations.append(args.iterations)

    # 记录参数到日志
    logging.info("Training started with the following arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

    training(
        lp.extract(args),
        hp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.expname,
    )

    # All done
    print("\nTraining complete.")
