/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
    return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
  const torch::Tensor& background,
  const torch::Tensor& means3D,
  const torch::Tensor& colors,
  const torch::Tensor& opacity,
  const torch::Tensor& scales,
  const torch::Tensor& rotations,
  const float scale_modifier,
  const torch::Tensor& cov3D_precomp,
  const torch::Tensor& viewmatrix,
  const torch::Tensor& projmatrix,
  const float tan_fovx, 
  const float tan_fovy,
  const int image_height,
  const int image_width,
  const torch::Tensor& sh,
  const int degree,
  const torch::Tensor& campos,
  const bool prefiltered,
  const bool debug,
  const torch::Tensor& image_gt
  )
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Tensor accum_scores_ptr = torch::full({P}, 0, float_opts);
  torch::Tensor accum_weights_ptr = torch::full({P}, 0, float_opts);
  torch::Tensor accum_weights_count = torch::full({P}, 0, int_opts);
  torch::Tensor accum_max_count = torch::full({P}, 0, float_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  

  int rendered = 0;
  if(P != 0)
  {
    int M = 0;
    if(sh.size(0) != 0)
    {
      M = sh.size(1);
    }

    rendered = CudaRasterizer::Rasterizer::forward(
      geomFunc,
      binningFunc,
      imgFunc,
      P, degree, M,
      background.contiguous().data<float>(),
      W, H,
      means3D.contiguous().data<float>(),
      sh.contiguous().data_ptr<float>(),
      colors.contiguous().data<float>(), 
      opacity.contiguous().data<float>(), 
      scales.contiguous().data_ptr<float>(),
      scale_modifier,
      rotations.contiguous().data_ptr<float>(),
      cov3D_precomp.contiguous().data<float>(), 
      viewmatrix.contiguous().data<float>(), 
      projmatrix.contiguous().data<float>(),
      campos.contiguous().data<float>(),
      tan_fovx,
      tan_fovy,
      prefiltered,
      out_color.contiguous().data<float>(),

      accum_scores_ptr.contiguous().data<float>(),  
      accum_weights_ptr.contiguous().data<float>(),  
      accum_weights_count.contiguous().data<int>(),  
      accum_max_count.contiguous().data<float>(),  
      image_gt.contiguous().data<float>(),

      radii.contiguous().data<int>(),
      debug);
  }
  return std::make_tuple(rendered, out_color, accum_weights_ptr, accum_weights_count, accum_max_count, radii, geomBuffer, 
  binningBuffer, imgBuffer, accum_scores_ptr);  
  
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TopKColorGaussiansCUDA(
  const torch::Tensor& background,
  const torch::Tensor& means3D,
  const torch::Tensor& colors,
  const torch::Tensor& opacity,
  const torch::Tensor& scales,
  const torch::Tensor& rotations,
  const float scale_modifier,
  const torch::Tensor& cov3D_precomp,
  const torch::Tensor& viewmatrix,
  const torch::Tensor& projmatrix,
  const float tan_fovx, 
  const float tan_fovy,
  const int image_height,
  const int image_width,
  const torch::Tensor& sh,
  const int degree,
  const torch::Tensor& campos,
  const bool prefiltered,
  const bool debug,

  const int topk,
  const int score_function,
  const torch::Tensor& image_gt,
  const float p_dist_activation_coef,
	const float c_dist_activation_coef
  )
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  auto bool_opts = means3D.options().dtype(torch::kBool);

  torch::Tensor accum_weights_ptr = torch::full({P}, 0, float_opts);
  torch::Tensor accum_weights_count = torch::full({P}, 0, int_opts);
  torch::Tensor accum_max_count = torch::full({P}, 0, float_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  // topk_mask
  torch::Tensor topk_mask = torch::full({P}, false, bool_opts);

  int rendered = 0;
  if(P != 0)
  {
    int M = 0;
    if(sh.size(0) != 0)
    {
      M = sh.size(1);
    }

    rendered = CudaRasterizer::Rasterizer::forwardTopKColor(
      geomFunc,
      binningFunc,
      imgFunc,
      P, degree, M,
      background.contiguous().data<float>(),
      W, H,
      means3D.contiguous().data<float>(),
      sh.contiguous().data_ptr<float>(),
      colors.contiguous().data<float>(), 
      opacity.contiguous().data<float>(), 
      scales.contiguous().data_ptr<float>(),
      scale_modifier,
      rotations.contiguous().data_ptr<float>(),
      cov3D_precomp.contiguous().data<float>(), 
      viewmatrix.contiguous().data<float>(), 
      projmatrix.contiguous().data<float>(),
      campos.contiguous().data<float>(),
      tan_fovx,
      tan_fovy,
      prefiltered,
      out_color.contiguous().data<float>(),

      accum_weights_ptr.contiguous().data<float>(),  
      accum_weights_count.contiguous().data<int>(),  
      accum_max_count.contiguous().data<float>(),  
      

      topk, // 指定需要保留多少个

      score_function,
      image_gt.contiguous().data_ptr<float>(),
      p_dist_activation_coef,
      c_dist_activation_coef,

      topk_mask.contiguous().data<bool>(),  // 数据载体

      radii.contiguous().data<int>(),
      debug
      );
  }

  return std::make_tuple(rendered, out_color, accum_weights_ptr, accum_weights_count, accum_max_count, radii, geomBuffer, 
  binningBuffer, imgBuffer, topk_mask);  
  
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TopKScoreGaussiansCUDA(
  const torch::Tensor& background,
  const torch::Tensor& means3D,
  const torch::Tensor& colors,
  const torch::Tensor& opacity,
  const torch::Tensor& scales,
  const torch::Tensor& rotations,
  const float scale_modifier,
  const torch::Tensor& cov3D_precomp,
  const torch::Tensor& viewmatrix,
  const torch::Tensor& projmatrix,
  const float tan_fovx, 
  const float tan_fovy,
  const int image_height,
  const int image_width,
  const torch::Tensor& sh,
  const int degree,
  const torch::Tensor& campos,
  const bool prefiltered,
  const bool debug,

  const int topk,
  const int score_function,
  const torch::Tensor& image_gt,
  const float p_dist_activation_coef,
	const float c_dist_activation_coef
  )
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Tensor accum_weights_ptr = torch::full({P}, 0, float_opts);
  torch::Tensor accum_weights_count = torch::full({P}, 0, int_opts);
  torch::Tensor accum_max_count = torch::full({P}, 0, float_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  // topk_score
  torch::Tensor topk_score = torch::full({P}, 0, float_opts);
  torch::Tensor score_counts = torch::full({P}, 0, int_opts);

  int rendered = 0;
  if(P != 0)
  {
    int M = 0;
    if(sh.size(0) != 0)
    {
      M = sh.size(1);
    }

    rendered = CudaRasterizer::Rasterizer::forwardTopKScore(
      geomFunc,
      binningFunc,
      imgFunc,
      P, degree, M,
      background.contiguous().data<float>(),
      W, H,
      means3D.contiguous().data<float>(),
      sh.contiguous().data_ptr<float>(),
      colors.contiguous().data<float>(), 
      opacity.contiguous().data<float>(), 
      scales.contiguous().data_ptr<float>(),
      scale_modifier,
      rotations.contiguous().data_ptr<float>(),
      cov3D_precomp.contiguous().data<float>(), 
      viewmatrix.contiguous().data<float>(), 
      projmatrix.contiguous().data<float>(),
      campos.contiguous().data<float>(),
      tan_fovx,
      tan_fovy,
      prefiltered,
      out_color.contiguous().data<float>(),

      accum_weights_ptr.contiguous().data<float>(),  
      accum_weights_count.contiguous().data<int>(),  
      accum_max_count.contiguous().data<float>(),  
      

      topk, // 指定需要保留多少个

      score_function,
      image_gt.contiguous().data_ptr<float>(),
      p_dist_activation_coef,
      c_dist_activation_coef,

      topk_score.contiguous().data<float>(),  // 数据载体
      score_counts.contiguous().data<int>(),  // 数据载体

      radii.contiguous().data<int>(),
      debug
      );
  }

  return std::make_tuple(rendered, out_color, accum_weights_ptr, accum_weights_count, accum_max_count, radii, geomBuffer, 
  binningBuffer, imgBuffer, topk_score);  
  
}







std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor>
RasterizeGaussiansDepthCUDA(
  const torch::Tensor& background,
  const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
  const torch::Tensor& scales,
  const torch::Tensor& rotations,
  const float scale_modifier,
  const torch::Tensor& cov3D_precomp,
  const torch::Tensor& viewmatrix,
  const torch::Tensor& projmatrix,
  const float tan_fovx, 
  const float tan_fovy,
    const int image_height,
    const int image_width,
  const torch::Tensor& sh,
  const int degree,
  const torch::Tensor& campos,
  const bool prefiltered,
  const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_pts = torch::full({3, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);

  torch::Tensor accum_alpha = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor gidx = torch::full({1, H, W}, 0.0, int_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  

  torch::Tensor discriminants = torch::full({1, H, W}, 0.0, float_opts);


  int rendered = 0;
  if(P != 0)
  {
    int M = 0;
    if(sh.size(0) != 0)
    {
      M = sh.size(1);
    }

    rendered = CudaRasterizer::Rasterizer::forward_depth(
      geomFunc,
      binningFunc,
      imgFunc,
      P, degree, M,
      background.contiguous().data<float>(),
      W, H,
      means3D.contiguous().data<float>(),
      sh.contiguous().data_ptr<float>(),
      colors.contiguous().data<float>(), 
      opacity.contiguous().data<float>(), 
      scales.contiguous().data_ptr<float>(),
      scale_modifier,
      rotations.contiguous().data_ptr<float>(),
      cov3D_precomp.contiguous().data<float>(), 
      viewmatrix.contiguous().data<float>(), 
      projmatrix.contiguous().data<float>(),
      campos.contiguous().data<float>(),
      tan_fovx,
      tan_fovy,
      prefiltered,
      out_color.contiguous().data<float>(),
      out_pts.contiguous().data<float>(),

      out_depth.contiguous().data<float>(),
      accum_alpha.contiguous().data<float>(),  
      gidx.contiguous().data<int>(),
      discriminants.contiguous().data<float>(),  

      radii.contiguous().data<int>(),
      debug);
  }

  return std::make_tuple(rendered, out_color, out_pts, out_depth, 
  accum_alpha, gidx, discriminants, radii, geomBuffer, 
  binningBuffer, imgBuffer);  
}  






std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
   const torch::Tensor& background,
  const torch::Tensor& means3D,
  const torch::Tensor& radii,
    const torch::Tensor& colors,
  const torch::Tensor& scales,
  const torch::Tensor& rotations,
  const float scale_modifier,
  const torch::Tensor& cov3D_precomp,
  const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
  const float tan_fovx,
  const float tan_fovy,
    const torch::Tensor& dL_dout_color,
  const torch::Tensor& sh,
  const int degree,
  const torch::Tensor& campos,
  const torch::Tensor& geomBuffer,
  const int R,
  const torch::Tensor& binningBuffer,
  const torch::Tensor& imageBuffer,
  const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {  
  M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
    CudaRasterizer::Rasterizer::backward(P, degree, M, R,
    background.contiguous().data<float>(),
    W, H, 
    means3D.contiguous().data<float>(),
    sh.contiguous().data<float>(),
    colors.contiguous().data<float>(),
    scales.data_ptr<float>(),
    scale_modifier,
    rotations.data_ptr<float>(),
    cov3D_precomp.contiguous().data<float>(),
    viewmatrix.contiguous().data<float>(),
    projmatrix.contiguous().data<float>(),
    campos.contiguous().data<float>(),
    tan_fovx,
    tan_fovy,
    radii.contiguous().data<int>(),
    reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
    reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
    reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
    dL_dout_color.contiguous().data<float>(),
    dL_dmeans2D.contiguous().data<float>(),
    dL_dconic.contiguous().data<float>(),  
    dL_dopacity.contiguous().data<float>(),
    dL_dcolors.contiguous().data<float>(),
    dL_dmeans3D.contiguous().data<float>(),
    dL_dcov3D.contiguous().data<float>(),
    dL_dsh.contiguous().data<float>(),
    dL_dscales.contiguous().data<float>(),
    dL_drotations.contiguous().data<float>(),
    debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
  CudaRasterizer::Rasterizer::markVisible(P,
    means3D.contiguous().data<float>(),
    viewmatrix.contiguous().data<float>(),
    projmatrix.contiguous().data<float>(),
    present.contiguous().data<bool>());
  }
  
  return present;
}