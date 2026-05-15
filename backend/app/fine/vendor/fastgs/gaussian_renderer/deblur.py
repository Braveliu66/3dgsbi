from __future__ import annotations

import math

import torch
from diff_gaussian_rasterization_fastgs import GaussianRasterizationSettings, GaussianRasterizer
from scene.blur_kernel import DeblurState, deblur_transform_regularization, predict_deblur_transforms
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render_fastgs_deblur(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    mult,
    deblur_state: DeblurState,
    scaling_modifier=1.0,
    override_color=None,
    get_flag=None,
    metric_map=None,
):
    """
    FastGS ABI-compatible Deblurring-3DGS training renderer.

    GTnet transforms are used only for the photometric training render; topology
    scoring and export should keep using the normal sharp render_fastgs path.
    """
    if not deblur_state.enabled:
        raise RuntimeError("DeblurState is disabled")
    assert deblur_state.config is not None

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    dc, shs, colors_precomp = _resolve_colors(viewpoint_camera, pc, pipe, override_color)
    scale_delta, rotation_delta, position_delta = predict_deblur_transforms(
        deblur_state,
        means3D,
        scales,
        rotations,
        viewpoint_camera.camera_center,
    )

    if not deblur_state.config.use_position:
        means2D = _screen_space_points(pc)
        rendered_image, radii, accum_metric_counts = _rasterize(
            viewpoint_camera,
            pc,
            pipe,
            bg_color,
            mult,
            means3D=means3D,
            means2D=means2D,
            opacity=opacity,
            scales=scales * scale_delta,
            rotations=rotations * rotation_delta,
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            scaling_modifier=scaling_modifier,
            get_flag=get_flag,
            metric_map=metric_map,
        )
        return {
            "render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter": radii > 0,
            "radii": radii,
            "accum_metric_counts": accum_metric_counts,
            "deblur_regularization": deblur_transform_regularization(scale_delta, rotation_delta, None),
        }

    moments = int(deblur_state.config.num_moments)
    scale_delta = scale_delta.view(-1, 3, moments + 1)
    rotation_delta = rotation_delta.view(-1, 4, moments + 1)
    assert position_delta is not None
    position_delta = position_delta.view(-1, 3, moments)

    renders = []
    first_means2D = None
    radii_accum = None
    metric_counts_accum = None
    for moment in range(moments + 1):
        means2D = _screen_space_points(pc)
        if first_means2D is None:
            first_means2D = means2D
        if moment == 0:
            transformed_means3D = means3D
            transform_index = moments
        else:
            transform_index = moment - 1
            transformed_means3D = means3D + position_delta[..., transform_index]
        rendered_image, radii, accum_metric_counts = _rasterize(
            viewpoint_camera,
            pc,
            pipe,
            bg_color,
            mult,
            means3D=transformed_means3D,
            means2D=means2D,
            opacity=opacity,
            scales=scales * scale_delta[..., transform_index],
            rotations=rotations * rotation_delta[..., transform_index],
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            scaling_modifier=scaling_modifier,
            get_flag=get_flag,
            metric_map=metric_map,
        )
        renders.append(rendered_image)
        radii_accum = radii if radii_accum is None else torch.maximum(radii_accum, radii)
        metric_counts_accum = (
            accum_metric_counts
            if metric_counts_accum is None
            else metric_counts_accum + accum_metric_counts
        )

    return {
        "render": sum(renders) / len(renders),
        "viewspace_points": first_means2D,
        "visibility_filter": radii_accum > 0,
        "radii": radii_accum,
        "accum_metric_counts": metric_counts_accum,
        "deblur_regularization": deblur_transform_regularization(scale_delta, rotation_delta, position_delta),
    }


def _screen_space_points(pc: GaussianModel) -> torch.Tensor:
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass
    return screenspace_points


def _resolve_colors(viewpoint_camera, pc: GaussianModel, pipe, override_color):
    dc = None
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc = pc.get_features_dc
            shs = pc.get_features_rest
    else:
        colors_precomp = override_color
    return dc, shs, colors_precomp


def _rasterize(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    mult,
    *,
    means3D,
    means2D,
    opacity,
    scales,
    rotations,
    dc,
    shs,
    colors_precomp,
    scaling_modifier,
    get_flag,
    metric_map,
):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if metric_map is None:
        metric_map = torch.zeros(
            int(viewpoint_camera.image_height) * int(viewpoint_camera.image_width),
            dtype=torch.int,
            device="cuda",
        )

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
        mult = mult,
        prefiltered=False,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map = metric_map,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    return rasterizer(
        means3D=means3D,
        means2D=means2D,
        dc=dc,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
