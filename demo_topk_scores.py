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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 先设置 GPU 设备

import sys
import numpy as np
import random
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, render_topk_score, render_point_time, network_gui
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
from impScoreUtils import *
import copy
from admm import ADMM
import wandb
from logger import initialize_logger
import logging


WANDB = False


current_time = initialize_logger(log_dir="./log", timezone_str="Etc/GMT-4")
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
            project="trainadmm_with_ckpt",
            name=f"{current_time}",
            config={
                "opt": vars(opt),
                "dataset": vars(dataset),
                "hyper": vars(hyper),
                "pipe": vars(pipe),
                "testing_iterations": testing_iterations,
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

        with torch.no_grad():
            scores = topk_gs_of_pixels_score(gaussians, scene, pipe, background, args.related_gs_num)
            # 假设 related_gs_mask 已经存在
            print("Statistics of scores:")
            print(f"Max: {torch.max(scores).item()}")
            print(f"Min: {torch.min(scores).item()}")
            print(f"Mean: {torch.mean(scores).item()}")
            print(f"Count > 0: {torch.sum(scores > 0).item()}")
            # 保存到本地
            torch.save(scores, "scores.pt")
            print("scores saved to scores.pt")
        break
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
