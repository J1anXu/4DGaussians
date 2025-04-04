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
from typing_extensions import Literal
idx = 4
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"

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
import wandb

import logging
from logger import initialize_logger

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

                wandb.log({
                    "SSIM": ssims,
                    "PSNR": psnrs,
                    "LPIPS-vgg": lpipss,
                    "LPIPS-alex": lpipsa,
                    "MS-SSIM": ms_ssims,
                    "D-SSIM": Dssims
                })

                wandb.summary["SSIM"] = ssims
                wandb.summary["PSNR"] = psnrs
                wandb.summary["LPIPS-vgg"] = lpipss
                wandb.summary["LPIPS-alex"] = lpipsa
                wandb.summary["MS-SSIM"] = ms_ssims
                wandb.summary["D-SSIM"] = Dssims

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
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    mp.set_start_method("spawn", force=True)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str, default=[])

    args = parser.parse_args()

    args_path = Path(args.model_paths[0]) / "opt_params.pth"
    initialize_logger()

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
        
    # 读取 run.id
    with open(f"wandb_run_id_{idx}.txt", "r") as f:
        run_id = f.read().strip()
        
    wandb.init(
        project="admm", 
        job_type="eval",
        id=run_id, 
        resume="allow"
        )


    evaluate(args.model_paths)
