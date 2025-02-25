import os
from pathlib import Path
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
from concurrent.futures import ThreadPoolExecutor
import logging
from logger import initialize_logger

# Initialize logger
initialize_logger(log_dir='./log', timezone_str="Etc/GMT-4")
logging.basicConfig(level=logging.INFO)

# Helper function to read images
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

# Evaluate function for a given scene
def evaluate_scene(scene_dir):
    try:
        print("Scene:", scene_dir)
        full_dict = {}
        per_view_dict = {}
        full_dict_polytopeonly = {}
        per_view_dict_polytopeonly = {}

        test_dir = Path(scene_dir) / "test"
        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []
            lpipsa = []
            ms_ssims = []
            Dssims = []
            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                ms_ssims.append(ms_ssim(renders[idx], gts[idx], data_range=1, size_average=True))
                lpipsa.append(lpips(renders[idx], gts[idx], net_type='alex'))
                Dssims.append((1 - ms_ssims[-1]) / 2)

            print(f"Scene: {scene_dir}")
            logging.info(f"Scene: {scene_dir}")
            logging.info(f"  SSIM: {torch.tensor(ssims).mean():.5f}")
            logging.info(f"  PSNR: {torch.tensor(psnrs).mean():.5f}")
            logging.info(f"  LPIPS-vgg: {torch.tensor(lpipss).mean():.5f}")
            logging.info(f"  LPIPS-alex: {torch.tensor(lpipsa).mean():.5f}")
            logging.info(f"  MS-SSIM: {torch.tensor(ms_ssims).mean():.5f}")
            logging.info(f"  D-SSIM: {torch.tensor(Dssims).mean():.5f}")

            full_dict[scene_dir][method] = {
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
                "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                "D-SSIM": torch.tensor(Dssims).mean().item()
            }

            per_view_dict[scene_dir][method] = {
                "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                "LPIPS-vgg": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                "LPIPS-alex": {name: lp for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
                "MS-SSIM": {name: ms_ssim for ms_ssim, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                "D-SSIM": {name: Dssim for Dssim, name in zip(torch.tensor(Dssims).tolist(), image_names)}
            }

        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict, fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

    except Exception as e:
        print(f"Unable to compute metrics for model {scene_dir}: {e}")
        logging.error(f"Unable to compute metrics for model {scene_dir}: {e}")

# Evaluate function that handles multiple scenes
def evaluate(model_paths):
    with ThreadPoolExecutor(max_workers=len(model_paths)) as executor:
        executor.map(evaluate_scene, model_paths)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()

    # Run evaluation with multithreading
    evaluate(args.model_paths)
