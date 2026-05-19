#
# 版权所有 (C) 2023, Inria
# GRAPHDECO 研究组, https://team.inria.fr/graphdeco
# 初始化输出目录
#
# 本软件仅可在 LICENSE.md 文件条款下用于
# 非商业、研究和评估用途。
#
# 咨询请联系：george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name


        try:
            # 优先使用外部指定设备，便于在 CPU/GPU 间切换调试。
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # original_image 统一约束在 [0,1]，后续损失计算与渲染输出保持同一数值域。
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # 若有 alpha mask，则直接乘到图像上屏蔽无效区域。
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            # 无 mask 时补全为全 1，不改变原图内容。
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # world_view_transform: 世界坐标 -> 相机坐标
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # projection_matrix: 相机坐标 -> 裁剪空间
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # full_proj_transform: 世界坐标 -> 裁剪空间（一体化矩阵）
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # 相机中心可由 view 矩阵逆变换得到。
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        # MiniCam 不持有图像，仅用于快速渲染或交互预览场景。
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

"""
相机数据结构定义。

本文件提供两类相机对象：
1. Camera   ：训练/评估阶段使用的完整相机，持有图像与投影矩阵；
2. MiniCam  ：轻量相机，仅保存渲染所需矩阵与基础参数。
"""
