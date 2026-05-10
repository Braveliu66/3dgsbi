// 本文件为 3DGS 预览系统内置 CUDA 扩展代码，来自上游关键运行依赖；保留原许可证，随 worker 镜像编译。
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

#include "spatial.h"
#include "simple_knn.h"

torch::Tensor
distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());

  return means;
}