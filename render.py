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
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  

import imageio
import numpy as np
import torch
from scene import Scene

import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer_ms import GaussianModel
from time import time
import threading
import concurrent.futures
from PIL import Image, ImageDraw
from torchvision import transforms
import torchvision.utils as vutils
DRAW = False # 是否画出高斯中心

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
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

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5
C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0
def SH2RGB(sh):
    return sh * C0 + 0.5
from PIL import Image, ImageDraw
def draw_points_on_image(points, colors, image, size=1):

    image[image>1]=1
    image[image<0]=0
    image = Image.fromarray((image*255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for point, color in zip(points, colors):
        x = point[0]
        y = point[1]
        
        r, g, b = color
        draw.ellipse((x-size,y-size,x+size,y+size), fill=(int(r), int(g), int(b)))
    return image

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    draw_path = os.path.join(model_path, name, "ours_{}".format(iteration), "draw")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if DRAW:
        makedirs(draw_path, exist_ok=True)

    render_images = []
    gt_list = []
    render_list = []
    draw_list = []
    print("point nums:",gaussians._xyz.shape[0])
    count = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()

        rendering_res = render(view, gaussians, pipeline, background, cam_type=cam_type)
        rendering = rendering_res["render"]

        if DRAW:
            means3D_final = rendering_res["means3D_final"]

            # perspective projection (modified from cuda code)
            xyz = means3D_final+0
            device = xyz.device  # 获取 xyz 的设备

            # 将 full_proj_transform 移动到与 xyz 相同的设备
            full_proj_transform = view.full_proj_transform.to(device)
            world_view_transform = view.world_view_transform.to(device)
            rgb = SH2RGB(gaussians._features_dc + 0)[:, 0]
            p_hom = torch.matmul(xyz, full_proj_transform[:3]) + full_proj_transform[3:4]
            p_w = 1.0 / (p_hom[:, 3] + 0.0000001)
            p_proj = p_hom[:, :3] * p_w[:, None]
            p_view = torch.matmul(xyz, world_view_transform[:3,:3])+world_view_transform[3:4, :3]
            mask = p_view[:,2].cpu().numpy()>0.2
            point_image = ndc2Pix(p_proj[:,0], rendering.shape[2]), ndc2Pix(p_proj[:,1], rendering.shape[1])
            point_image = torch.cat((point_image[0][:,None], point_image[1][:,None]), -1)
            points = point_image.detach().cpu().numpy()[mask]
            colors = rgb.detach().cpu().numpy()[mask]

            # tune point size for better visualization 0.3, 0.3, 1.2
            image_proj = draw_points_on_image(points, np.zeros(colors.shape)+[0,0,255], rendering.permute(1,2,0).detach().cpu().numpy(), size=0.3)
            # 创建转换函数
            transform = transforms.ToTensor()

            # 将 PIL 图像转换为 Tensor
            drwa_image = transform(image_proj)  # pil_image 是你的 PIL.Image 对象
            #image_proj.save(f"{draw_path}/{idx}.jpg")
            draw_list.append(drwa_image)

        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)

        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)
        count+=1
        # 防止渲染train的时候显存溢出
        if count>350:
            break


    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    multithread_write(gt_list, gts_path)

    multithread_write(render_list, render_path)

    if DRAW:
        multithread_write(draw_list, draw_path)


    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30)
def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type)
            
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type)
        
        print(f"current model point size : {gaussians.get_xyz.shape[0]}")

if __name__ == "__main__":
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
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)
