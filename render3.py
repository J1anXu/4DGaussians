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

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"

import torch
import torch.multiprocessing as mp
import imageio
import numpy as np
from scene import Scene
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer_ms import render_with_error_scores
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer_ms import GaussianModel
import time
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Subset
import concurrent.futures
import torch.multiprocessing as mp
DRAW = True  # 是否画出高斯中心

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)


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
        rendering_res = render_with_error_scores(view, gaussians, pipeline, background, cam_type=cam_type)
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


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering ", args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    parallel_render_sets(
        model.extract(args),
        hyperparam.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.skip_video,
    )
