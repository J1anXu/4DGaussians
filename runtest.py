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
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import sys
import numpy as np
import random
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, render_with_topk_mask, render_point_time, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
from utils.scene_utils import render_training_image
from impScoreUtils import *
import copy
from admm import ADMM
import wandb
import logging
import time
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Subset
import concurrent.futures
import torch.multiprocessing as mp
from pathlib import Path
import torchvision
import torchvision.transforms.functional as tf
import pprint
from pathlib import Path
import torch.multiprocessing as mp
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from pytorch_msssim import ms_ssim
import torch
import torch.multiprocessing as mp
import imageio
import numpy as np
from scene import Scene
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
import time
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Subset
import concurrent.futures
import torch.multiprocessing as mp
import logging
from os import makedirs
from logger import initialize_logger

current_time = initialize_logger()
DRAW = True  # 是否画出高斯中心
to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)
WANDB=True

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

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    ema_admm_loss_for_log = 0.0
    final_iter = train_iter

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training...")
    first_iter += 1
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
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

    admm_loss = torch.tensor(0)

    count = 0
    if WANDB:
        wandb.init(
            project="train_admm_with_ckpt",
            name=f"{args.log_name}",
            config={
                "opt": vars(opt),
                # "dataset": vars(dataset),
                # "hyper": vars(hyper),
                # "pipe": vars(pipe),
                # "testing_iterations": testing_iterations,
                "saving_iterations": saving_iterations,
                "checkpoint": checkpoint,
                "stage": stage,
                "train_iter": train_iter,
            },
        )

    times = scene.train_camera.dataset.image_times[0 : scene.train_camera.dataset.time_number]
    # scores = getImportantScore4(gaussians, opt, scene, pipe, background)

    scores = None
    related_gs_mask = None

    for iteration in range(first_iter, final_iter + 1):
        if network_gui.conn == None:
            network_gui.try_connect()

        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    count += 1
                    viewpoint_index = (count) % len(video_cams)
                    if (count // (len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time

                    net_image = render(
                        custom_cam,
                        gaussians,
                        pipe,
                        background,
                        scaling_modifer,
                        stage=stage,
                        cam_type=scene.dataset_type,
                    )["render"]

                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(
                        viewpoint_stack, batch_size=opt.batch_size, shuffle=True, num_workers=32, collate_fn=list
                    )
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size:
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
                if not viewpoint_stack:
                    viewpoint_stack = temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx += 1
            if len(viewpoint_cams) == 0:
                continue

        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii, p_diff, time = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["p_diff"],
                render_pkg["time"],
            )
            images.append(image.unsqueeze(0))
            if scene.dataset_type != "PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image = viewpoint_cam["image"].cuda()

            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images, 0)
        gt_image_tensor = torch.cat(gt_images, 0)

        Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        loss = Ll1

        if (
            opt.admm == True
            and iteration > opt.admm_start_iter1
            and iteration % opt.admm_interval == 0
            and iteration <= opt.admm_stop_iter1
        ):
            admm_loss = 0.1 * admm.get_admm_loss(loss)
            loss += admm_loss

        if stage == "fine" and hyper.time_smoothness_weight != 0:
            tv_loss = gaussians.compute_regulation(
                hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight
            )
            loss += tv_loss

        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor, gt_image_tensor)
            loss += opt.lambda_dssim * (1.0 - ssim_loss)

        loss.backward()

        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            ema_admm_loss_for_log = 0.4 * admm_loss.item() + 0.6 * ema_admm_loss_for_log

            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "admm_loss": f"{ema_admm_loss_for_log:.{7}f}",
                        "psnr": f"{psnr_:.{2}f}",
                        "point": f"{total_point}",
                    }
                )
                progress_bar.update(10)
                logging.info(
                    {
                        "Loss": f"{ema_loss_for_log:.5f}",
                        "admm_loss": f"{ema_admm_loss_for_log:.5f}",
                        "psnr": f"{psnr_:.2f}",
                        "point": total_point,  # 直接使用数值，无需字符串格式化
                    }
                )
                if WANDB:
                    wandb.log(
                        {   "iteration":iteration,
                            "loss": round(ema_loss_for_log, 7),
                            "admm_loss": round(ema_admm_loss_for_log, 7),
                            "psnr": round(psnr_.item(), 7),
                            "point": total_point,  # 直接使用数值，无需字符串格式化
                        }
                    )

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                [pipe, background],
                stage,
                scene.dataset_type,
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)

            if dataset.render_process:
                if (
                    (iteration < 1000 and iteration % 10 == 9)
                    or (iteration < 3000 and iteration % 50 == 49)
                    or (iteration < 60000 and iteration % 100 == 99)
                ):
                    # breakpoint()
                    render_training_image(
                        scene,
                        gaussians,
                        [test_cams[iteration % len(test_cams)]],
                        render,
                        pipe,
                        background,
                        stage + "test",
                        iteration,
                        timer.get_elapsed_time(),
                        scene.dataset_type,
                    )
                    render_training_image(
                        scene,
                        gaussians,
                        [train_cams[iteration % len(train_cams)]],
                        render,
                        pipe,
                        background,
                        stage + "train",
                        iteration,
                        timer.get_elapsed_time(),
                        scene.dataset_type,
                    )
                    # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                # total_images.append(to8b(temp_image).transpose(1,2,0))
            if WANDB & iteration > opt.admm_start_iter1 & iteration % opt.admm_interval == 0:
                wandb.log({"opacity_aftet_admm": wandb.Histogram(gaussians.get_opacity.tolist())})

            timer.start()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    abs_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:
                    abs_threshold = opt.opacity_threshold_fine_init - iteration * (
                        opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after
                    ) / (opt.densify_until_iter)
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration * (
                        opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after
                    ) / (opt.densify_until_iter)

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                    and gaussians.get_xyz.shape[0] < 360000
                ):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(
                        densify_threshold,
                        abs_threshold,
                        scene.cameras_extent,
                        size_threshold,
                        5,
                        5,
                        scene.model_path,
                        iteration,
                        stage,
                    )

                if (
                    iteration > opt.pruning_from_iter
                    and iteration % opt.pruning_interval == 0
                    and gaussians.get_xyz.shape[0] > 200000
                ):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, abs_threshold, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()

            elif args.prune_points and iteration == args.simp_iteration1:

                # scores, related_gs_mask = run_tasks_in_parallel(
                #     (time_0_bleding_weight, gaussians, opt, scene, pipe, background),
                #     (topk_gs_of_pixels, gaussians, scene, pipe, background, args.related_gs_num)
                # )
                # max_score = torch.max(scores)
                # scores[related_gs_mask] += max_score
                # print(f"Max score: {max_score}")

                scores = time_0_bleding_weight(gaussians, opt, args, scene, pipe, background)
                # related_gs_mask = topk_gs_of_pixels(gaussians, scene, pipe, background, args.related_gs_num)

                scores_sorted, _ = torch.sort(scores, 0)
                threshold_idx = int(opt.opacity_admm_threshold1 * len(scores_sorted))
                abs_threshold = scores_sorted[threshold_idx - 1]
                mask = (scores <= abs_threshold).squeeze()

                # 获取标记的pixels对应的gs的gs,相关的gs被标记为true
                gaussians.prune_points(mask)

            elif iteration == opt.admm_start_iter1 and opt.admm == True:
                admm = ADMM(gaussians, opt.rho_lr, device="cuda")
                admm.update(opt, update_u=False)
            elif (
                iteration % opt.admm_interval == 0
                and opt.admm == True
                and (iteration > opt.admm_start_iter1 and iteration <= opt.admm_stop_iter1)
            ):
                admm.update(opt)

            if args.prune_points and iteration == args.simp_iteration2:
                scores_2 = getOpacityScore(gaussians)
                scores_sorted_2, _ = torch.sort(scores_2, 0)
                threshold_idx = int(opt.opacity_admm_threshold2 * len(scores_sorted_2))
                abs_threshold = scores_sorted_2[threshold_idx - 1]
                mask_2 = (scores_2 <= abs_threshold).squeeze()
                gaussians.prune_points(mask_2)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + f"_{stage}_" + str(iteration) + ".pth",
                )
    if WANDB:
        wandb.finish()


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


def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)

    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, "{0:05d}".format(count) + ".png"))
            return count, True
        except:
            return count, False

    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)


def ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def draw_points_on_image(points, colors, image, size=1):
    image[image > 1] = 1
    image[image < 0] = 0
    image = Image.fromarray((image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for point, color in zip(points, colors):
        x, y = point
        r, g, b = color
        draw.ellipse((x - size, y - size, x + size, y + size), fill=(int(r), int(g), int(b)))
    return image


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    draw_path = os.path.join(model_path, name, f"ours_{iteration}", "draw")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if DRAW:
        makedirs(draw_path, exist_ok=True)

    render_images = []
    gt_list = []
    render_list = []
    draw_list = []

    print("Point count:", gaussians._xyz.shape[0])
    count = 0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering_res = render(view, gaussians, pipeline, background, cam_type=cam_type)
        rendering = rendering_res["render"]

        if DRAW:
            means3D_final = rendering_res["means3D_final"]
            xyz = means3D_final.clone()
            device = xyz.device
            full_proj_transform = view.full_proj_transform.to(device)
            world_view_transform = view.world_view_transform.to(device)
            rgb = gaussians._features_dc[:, 0]
            p_hom = torch.matmul(xyz, full_proj_transform[:3]) + full_proj_transform[3:4]
            p_w = 1.0 / (p_hom[:, 3] + 1e-7)
            p_proj = p_hom[:, :3] * p_w[:, None]
            p_view = torch.matmul(xyz, world_view_transform[:3, :3]) + world_view_transform[3:4, :3]
            mask = p_view[:, 2].cpu().numpy() > 0.2
            point_image = ndc2Pix(p_proj[:, 0], rendering.shape[2]), ndc2Pix(p_proj[:, 1], rendering.shape[1])
            point_image = torch.cat((point_image[0][:, None], point_image[1][:, None]), -1)
            points = point_image.detach().cpu().numpy()[mask]
            colors = rgb.detach().cpu().numpy()[mask]

            image_proj = draw_points_on_image(
                points,
                np.zeros(colors.shape) + [0, 0, 255],
                rendering.permute(1, 2, 0).detach().cpu().numpy(),
                size=0.3,
            )
            transform = transforms.ToTensor()
            draw_image = transform(image_proj)
            draw_list.append(draw_image)

        render_images.append(to8b(rendering).transpose(1, 2, 0))
        render_list.append(rendering.cpu())

        if name in ["train", "test"]:
            gt = view.original_image[0:3, :, :]
            gt_list.append(gt)

    return render_list, gt_list, draw_list
    # return render_images


def render_worker(gpu_id, dataset, hyperparam, iteration, pipeline, views, mode, result_dict):
    """多进程渲染 Worker 进程"""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type = scene.dataset_type
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)

        render_list, gt_list, draw_list = render_set(
            dataset.model_path, mode, scene.loaded_iter, views, gaussians, pipeline, background, cam_type
        )

    result_dict[gpu_id] = {
        "renders": render_list,
        "gts": gt_list,
        "draws": draw_list,
    }


def ensure_directories_exist(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def parallel_render_sets(dataset, hyperparam, iteration, pipeline, skip_train, skip_test, skip_video):
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs.")

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    manager = mp.Manager()
    result_dict = manager.dict()  # 共享字典存储 GPU 结果
    processes = []

    def split_and_render(mode, views):
        """按 GPU 划分 views，并启动多进程渲染"""
        if len(views) == 0:
            return

        num_views = len(views)
        indices = torch.linspace(0, num_views, num_gpus + 1, dtype=torch.int).tolist()
        split_views = [Subset(views, range(indices[i], indices[i + 1])) for i in range(num_gpus)]

        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=render_worker,
                args=(gpu_id, dataset, hyperparam, scene.loaded_iter, pipeline, split_views[gpu_id], mode, result_dict),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # **汇总所有 GPU 结果并按顺序写入**
        sorted_gpu_ids = sorted(result_dict.keys())  # 确保按 GPU ID 顺序
        final_renders, final_gts, final_draws = [], [], []

        for gpu_id in sorted_gpu_ids:
            final_renders.extend(result_dict[gpu_id]["renders"])
            final_gts.extend(result_dict[gpu_id]["gts"])
            final_draws.extend(result_dict[gpu_id]["draws"])

        render_path = os.path.join(dataset.model_path, mode, f"ours_{scene.loaded_iter}", "renders")
        gts_path = os.path.join(dataset.model_path, mode, f"ours_{scene.loaded_iter}", "gt")
        draw_path = os.path.join(dataset.model_path, mode, f"ours_{scene.loaded_iter}", "draw")
        ensure_directories_exist(render_path, gts_path, draw_path)

        print(f"Writing images to {gts_path}... size = {len(final_gts)}")
        multithread_write(final_gts, gts_path)
        time.sleep(5)

        print(f"Writing images to {render_path}... size = {len(final_renders)}")
        multithread_write(final_renders, render_path)
        time.sleep(5)

        if DRAW:
            print(f"Writing images to {draw_path}... size = {len(final_draws)}")
            multithread_write(final_draws, draw_path)

    # if not skip_train:
    #     print("Starting train set rendering...")
    #     split_and_render("train", scene.getTrainCameras())

    if not skip_test:
        print("Starting test set rendering...")
        split_and_render("test", scene.getTestCameras())

    # if not skip_video:
    #     print("Starting video rendering...")
    #     split_and_render("video", scene.getVideoCameras())

    print("All rendering processes finished.")

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                # Parallelize the metric evaluation across multiple GPUs
                results = parallel_evaluation(renders, gts, num_gpus=torch.cuda.device_count())

                # Compute the final metrics
                ssims = results["SSIM"]
                psnrs = results["PSNR"]
                lpipss = results["LPIPS-vgg"]
                lpipsa = results["LPIPS-alex"]
                ms_ssims = results["MS-SSIM"]
                Dssims = results["D-SSIM"]

                # Print the results
                print(f"Metrics2 {method_dir}")
                print("SSIM : {:>12.8f}".format(ssims))
                print("PSNR : {:>12.8f}".format(psnrs))
                print("LPIPS-vgg: {:>12.8f}".format(lpipss))
                print("LPIPS-alex: {:>12.8f}".format(lpipsa))
                print("MS-SSIM: {:>12.8f}".format(ms_ssims))
                print("D-SSIM: {:>12.8f}".format(Dssims))

                # Logging the results
                logging.info(f"Mertics2 {method_dir}")
                logging.info("Scene: %s", scene_dir)
                logging.info("  SSIM: %.8f", ssims)
                logging.info("  PSNR: %.8f", psnrs)
                logging.info("  LPIPS-vgg: %.8f", lpipss)
                logging.info("  LPIPS-alex: %.8f", lpipsa)
                logging.info("  MS-SSIM: %.8f", ms_ssims)
                logging.info("  D-SSIM: %.8f", Dssims)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            raise e


def parallel_evaluation(renders, gts, num_gpus=8):
    # Split renders and gts into N parts for N GPUs
    splits = len(renders) // num_gpus
    results = mp.Manager().dict()

    # Create processes, one for each GPU
    processes = []
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * splits
        end_idx = (gpu_id + 1) * splits if gpu_id < num_gpus - 1 else len(renders)

        device = torch.device(f"cuda:{gpu_id}")
        p = mp.Process(target=worker, args=(device, renders, gts, start_idx, end_idx, results))
        processes.append(p)
        p.start()

    # Join processes
    for p in processes:
        p.join()

    # Aggregate the results from all GPUs
    aggregated_results = {"SSIM": [], "PSNR": [], "LPIPS-vgg": [], "LPIPS-alex": [], "MS-SSIM": [], "D-SSIM": []}
    lock = mp.Lock()

    # 将结果汇总并计算均值
    for device_results in results.values():
        with lock:  # 加锁，确保线程安全
            for metric in aggregated_results:
                aggregated_results[metric].extend(device_results[metric])

    # 计算均值
    mean_results = {
        metric: sum(map(float, values)) / len(values) if len(values) > 0 else 0
        for metric, values in aggregated_results.items()
    }

    return mean_results


def worker(device, renders, gts, start_idx, end_idx, results):
    ssims = []
    psnrs = []
    lpipss = []
    lpipsa = []
    ms_ssims = []
    Dssims = []

    # Move tensors to the corresponding device (GPU)
    renders = [render.to(device) for render in renders[start_idx:end_idx]]
    gts = [gt.to(device) for gt in gts[start_idx:end_idx]]

    # Calculate metrics
    for idx in tqdm(range(start_idx, end_idx), desc="Evaling", unit="item"):
        ssims.append(ssim(renders[idx - start_idx], gts[idx - start_idx]).item())
        psnrs.append(psnr(renders[idx - start_idx], gts[idx - start_idx]).item())
        lpipss.append(lpips(renders[idx - start_idx], gts[idx - start_idx], net_type="vgg").item())
        ms_ssims.append(ms_ssim(renders[idx - start_idx], gts[idx - start_idx], data_range=1, size_average=True).item())
        lpipsa.append(lpips(renders[idx - start_idx], gts[idx - start_idx], net_type="alex").item())
        Dssims.append((1 - ms_ssims[-1]) / 2)

    # Ensure the results are on CPU and clone the tensors to avoid CUDA issues when passing between processes
    results[device] = {
        "SSIM": ssims,
        "PSNR": psnrs,
        "LPIPS-vgg": lpipss,
        "LPIPS-alex": lpipsa,
        "MS-SSIM": ms_ssims,
        "D-SSIM": Dssims,
    }

    # Move results to CPU to avoid CUDA context sharing issues across processes
    for key, value in results[device].items():
        # Ensure that the values are not CUDA tensors
        results[device][key] = [val.cpu() if isinstance(val, torch.Tensor) else val for val in value]

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
    parser.add_argument("--port", type=int, default=6018)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default="None")
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--log_name", type=str, default="default")
    mp.set_start_method("spawn", force=True)

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


    parallel_render_sets(
        lp.extract(args),
        hp.extract(args),
        args.iteration,
        pp.extract(args),
        args.skip_train,
        args.skip_test,
        args.skip_video,
    )

    args_path = Path(args.model_paths[0]) / "opt_params.pth"

    # 检查文件是否存在
    if args_path.exists():
        # 如果文件存在，加载数据
        hp_data = torch.load(args_path)
        print("Hyperparameters loaded successfully.")
        # 格式化输出
        pretty_data = pprint.pformat(vars(hp_data), indent=2)
        logging.info(f"Loaded data:\n{pretty_data}\n")
    else:
        # 如果文件不存在，打印错误并放弃
        print(f"Error: The file {args_path} does not exist. Skipping...")

    evaluate(args.model_paths)


