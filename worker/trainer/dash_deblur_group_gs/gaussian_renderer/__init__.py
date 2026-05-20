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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import torch.nn.functional as F
from scene.blur_types import BLUR_SHARP, BLUR_MOTION, BLUR_DEFOCUS, normalize_blur_type

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier=1.0, deblur=0, use_pos=False,
           blur_type=None, image_id=None,
           lambda_s=0.01, lambda_p=0.01, max_clamp=1.1, force_original_backend=False ):
    """
    ?????
    
    ?????bg_color????? GPU ??
    """

    # 创建零张量，用于让 PyTorch 返回二维（屏幕空间）均值的梯度。
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if blur_type is not None:
        blur_type = normalize_blur_type(blur_type)

    # 全部完成
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # 若提供预计算的 3D 协方差则直接使用，否则由
    # 全部完成
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else: 
        if not deblur or blur_type == BLUR_SHARP:  # 初始化命令行参数解析器
            scales = pc.get_scaling
            rotations = pc.get_rotation 

            shs = None
            colors_precomp = None
            shs = pc.get_features
            if not force_original_backend and getattr(pipe, "renderer_backend", "original") == "gsplat":
                from gaussian_renderer.backends.gsplat_backend import gsplat_rasterize
                rendered_image, radii, screenspace_points = gsplat_rasterize(
                    viewpoint_camera,
                    pc,
                    bg_color,
                    scaling_modifier=scaling_modifier,
                )
            else:
                rendered_image, radii = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = colors_precomp,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
            
            return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

        else:
            scales = pc.get_scaling 
            rotations = pc.get_rotation
            shs = pc.get_features 
            colors_precomp = None
            _pos = means3D.detach()
            _scales = scales.detach()
            _rotations = rotations.detach()
            _viewdirs = viewpoint_camera.camera_center.repeat(means3D.shape[0], 1)

            if blur_type == BLUR_DEFOCUS:
                if image_id is None:
                    image_id = getattr(viewpoint_camera, "image_id", 0)
                scales_delta, rotations_delta, pos_delta, blur_code = pc.defocus_GTnet(_pos, _scales, _rotations, _viewdirs, image_id)
                scales_delta = torch.clamp(lambda_s * scales_delta + (1-lambda_s), min=1.0, max=max_clamp)
                rotations_delta = torch.clamp(lambda_s * rotations_delta + (1-lambda_s), min=1.0, max=max_clamp)
                transformed_scales = scales * scales_delta
                transformed_rotations = rotations * rotations_delta

                rendered_image, radii = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = colors_precomp,
                    opacities = opacity,
                    scales = transformed_scales,
                    rotations = transformed_rotations,
                    cov3D_precomp = cov3D_precomp)

                delta_reg = (scales_delta - 1.0).abs().mean() + (rotations_delta - 1.0).abs().mean()
                return {"render": rendered_image,
                        "viewspace_points": screenspace_points,
                        "visibility_filter" : radii > 0,
                        "radii": radii,
                        "blur_code": blur_code,
                        "delta_reg": delta_reg}

            if blur_type == BLUR_MOTION:
                if image_id is None:
                    image_id = getattr(viewpoint_camera, "image_id", 0)
                M = pc.motion_GTnet.num_moments
                scales_delta, rotations_delta, pos_delta, blur_code = pc.motion_GTnet(_pos, _scales, _rotations, _viewdirs, image_id)
                scales_delta = torch.clamp(lambda_s * scales_delta + (1-lambda_s), min=1.0, max=max_clamp)
                rotations_delta = torch.clamp(lambda_s * rotations_delta + (1-lambda_s), min=1.0, max=max_clamp)
                pos_delta = lambda_p * pos_delta
                pos_delta = pos_delta.view(-1, 3, M)
                scales_delta = scales_delta.view(-1, 3, M+1)
                rotations_delta = rotations_delta.view(-1, 4, M+1)

                pos = means3D
                transformed_scales = scales * scales_delta[...,-1]
                transformed_rotations = rotations * rotations_delta[...,-1]

                rendered_image, _radii = rasterizer(
                    means3D = pos,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = colors_precomp,
                    opacities = opacity,
                    scales = transformed_scales,
                    rotations = transformed_rotations,
                    cov3D_precomp = cov3D_precomp)

                renders = [rendered_image]
                viewspace_points = [screenspace_points]
                visibility_filter = [_radii > 0]
                radii = [_radii]

                for i in range(M):
                    screenspace_points_i = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
                    try:
                        screenspace_points_i.retain_grad()
                    except:
                        pass
                    transformed_pos = means3D + pos_delta[...,i]
                    transformed_scales = scales * scales_delta[...,i]
                    transformed_rotations = rotations * rotations_delta[...,i]

                    rendered_image, _radii = rasterizer(
                        means3D = transformed_pos,
                        means2D = screenspace_points_i,
                        shs = shs,
                        colors_precomp = colors_precomp,
                        opacities = opacity,
                        scales = transformed_scales,
                        rotations = transformed_rotations,
                        cov3D_precomp = cov3D_precomp)

                    renders.append(rendered_image)
                    viewspace_points.append(screenspace_points_i)
                    visibility_filter.append(_radii > 0)
                    radii.append(_radii)

                render = sum(renders) / len(renders)
                delta_reg = pos_delta.abs().mean() + (scales_delta - 1.0).abs().mean() + (rotations_delta - 1.0).abs().mean()
                return {"render": render,
                    "viewspace_points": viewspace_points,
                    "visibility_filter" : visibility_filter,
                    "radii": radii,
                    "blur_code": blur_code,
                    "delta_reg": delta_reg}

            M = pc.GTnet.num_moments
            scales_delta, rotations_delta, pos_delta = pc.GTnet(_pos, _scales, _rotations, _viewdirs)
            scales_delta = torch.clamp(lambda_s * scales_delta + (1-lambda_s), min=1.0, max=max_clamp)
            rotations_delta = torch.clamp(lambda_s * rotations_delta + (1-lambda_s), min=1.0, max=max_clamp)

            if not use_pos:    # 初始化命令行参数解析器
                transformed_scales = scales * scales_delta
                transformed_rotations = rotations * rotations_delta

                rendered_image, radii = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = colors_precomp,
                    opacities = opacity,
                    scales = transformed_scales,
                    rotations = transformed_rotations,
                    cov3D_precomp = cov3D_precomp)
                
                return {"render": rendered_image,
                        "viewspace_points": screenspace_points,
                        "visibility_filter" : radii > 0,
                        "radii": radii}
            
            elif use_pos:   # 初始化命令行参数解析器
                pos_delta = lambda_p * pos_delta
                pos_delta = pos_delta.view(-1, 3, M)
                scales_delta = scales_delta.view(-1, 3, M+1)
                rotations_delta = rotations_delta.view(-1, 4, M+1)

                pos = means3D
                transformed_scales = scales * scales_delta[...,-1]
                transformed_rotations = rotations * rotations_delta[...,-1]

                rendered_image, _radii = rasterizer(
                    means3D = pos,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = colors_precomp,
                    opacities = opacity,
                    scales = transformed_scales,
                    rotations = transformed_rotations,
                    cov3D_precomp = cov3D_precomp)

                # renders = [torch.clamp(rendered_image, min=0.0, max=1.0)]  # 可选：限制到 [0,1]
                renders = [rendered_image]
                viewspace_points = [screenspace_points]
                visibility_filter = [_radii > 0]
                radii = [_radii]

                for i in range(M):
                    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
                    try:
                        screenspace_points.retain_grad()
                    except:
                        pass
                    means2D = screenspace_points
                    transformed_pos = means3D + pos_delta[...,i]
                    transformed_scales = scales * scales_delta[...,i]
                    transformed_rotations = rotations * rotations_delta[...,i]

                    rendered_image, _radii = rasterizer(
                        means3D = transformed_pos,
                        means2D = means2D,
                        shs = shs,
                        colors_precomp = colors_precomp,
                        opacities = opacity,
                        scales = transformed_scales,
                        rotations = transformed_rotations,
                        cov3D_precomp = cov3D_precomp)

                    # renders.append(torch.clamp(rendered_image, min=0.0, max=1.0))  # 可选：限制到 [0,1]
                    renders.append(rendered_image)
                    viewspace_points.append(screenspace_points)
                    visibility_filter.append(_radii > 0)
                    radii.append(_radii)

                render = sum(renders) / len(renders)

                return {"render": render,
                    "viewspace_points": viewspace_points,
                    "visibility_filter" : visibility_filter,
                    "radii": radii}


