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
idx = 2
os.environ["CUDA_VISIBLE_DEVICES"] = f"{idx + 2}"  # 先设置 GPU 设备

import sys
import numpy as np
import random
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer_ms import render, render_with_topk_mask, render_point_time, network_gui
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
from utils.tensor_decomposition_utils import decom_tt
import copy
from admm import ADMM
from bw3 import BW
import wandb
import logging
from logger import initialize_logger
WANDB = True
to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# 逐步从 tt_tensor 的 TT 核心恢复 full tensor
def reconstruct_tt_tensor(tt_tensor, original_shape):
    result = tt_tensor.cores[0]
    for core in tt_tensor.cores[1:]:
        result = torch.einsum('aib, bjc -> aijc', result, core).reshape(
            result.shape[0], -1, core.shape[2])
    result = result.squeeze(0)
    return result.view(original_shape)

def reconstruct_from_tt(tt_cores):
    reconstructed_tensor = tt_cores[0]
    for core in tt_cores[1:]:
        reconstructed_tensor = reconstructed_tensor @ core  # 使用张量乘法合并
    return reconstructed_tensor



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
    bw = None
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
    iter_now = first_iter
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


    

    rank_dict = {
        (0, 0): 64,  # level_1=0, level_2=0 使用 rank=4
        (0, 1): 64,  # level_1=0, level_2=1 使用 rank=8
        (0, 2): 48, # level_1=0, level_2=2 使用 rank=12
        (0, 3): 64, # level_1=0, level_2=3 使用 rank=16
        (0, 4): 20, # level_1=0, level_2=4 使用 rank=20
        (0, 5): 20, # level_1=0, level_2=5 使用 rank=18
        (1, 0): 128,  # level_1=1, level_2=0 使用 rank=4
        (1, 1): 128,  # level_1=1, level_2=1 使用 rank=8
        (1, 2): 64, # level_1=1, level_2=2 使用 rank=12
        (1, 3): 128, # level_1=1, level_2=3 使用 rank=16
        (1, 4): 64, # level_1=1, level_2=4 使用 rank=20
        (1, 5): 32, # level_1=1, level_2=5 使用 rank=18
    }
    decom_path = os.path.join(scene.model_path, f"point_cloud/iteration_{iter_now}/decom")
    decom_tt(scene.gaussians,rank_dict,decom_path)

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
        "fine",
        tb_writer,
        opt.coarse_iterations,
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
    parser.add_argument("--log_name", type=str, default="default")
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

    current_time = initialize_logger("Train")

    for arg, value in vars(op.extract(args)).items():
        logging.info(f"{arg}: {value}")
        
    if WANDB:
        wandb.login()
        run = wandb.init(
            project="admm",
            name=f"idx_{idx}_{current_time}",  # 让不同脚本的数据归为一组
            job_type="training",
            config=vars(op.extract(args))
        )
        wandb.define_metric("iteration")  # 将 iteration 作为横坐标

    # 保存 run_id 供后续使用
    with open(f"wandb_run_id_{idx}.txt", "w") as f:
        f.write(run.id)

    output_path = "./output/" + args.expname

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
    

