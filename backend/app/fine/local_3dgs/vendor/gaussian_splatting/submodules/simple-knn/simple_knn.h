// 本文件为 3DGS 预览系统内置 C/C++ 头文件，来自上游关键运行依赖；保留原许可证，随 worker 镜像编译。
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

#ifndef SIMPLEKNN_H_INCLUDED
#define SIMPLEKNN_H_INCLUDED

class SimpleKNN
{
public:
	static void knn(int P, float3* points, float* meanDists);
};

#endif