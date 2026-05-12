from __future__ import annotations

import math
from typing import Any

import torch


def render_gaussians(
    viewpoint_camera: Any,
    pc: Any,
    pipe: Any,
    bg_color: torch.Tensor,
    *,
    cg_state: Any | None = None,
    current_batch: int = -1,
    is_batched: bool = False,
    scaling_modifier: float = 1.0,
    fastgs_mult: float = 0.5,
    fastgs_get_flag: bool = False,
    fastgs_metric_map: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    from diff_gaussian_rasterization_fastgs import GaussianRasterizationSettings, GaussianRasterizer

    screen_shape = (int(pc.get_xyz.shape[0]), 4)
    screen = torch.zeros(screen_shape, dtype=pc.get_xyz.dtype, requires_grad=True, device=pc.get_xyz.device) + 0
    try:
        screen.retain_grad()
    except Exception:
        pass

    metric_map = fastgs_metric_map
    if metric_map is None:
        metric_map = torch.zeros(int(viewpoint_camera.image_height) * int(viewpoint_camera.image_width), dtype=torch.int32, device=bg_color.device)
    settings = _settings(
        GaussianRasterizationSettings,
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        mult=fastgs_mult,
        prefiltered=False,
        debug=bool(getattr(pipe, "debug", False)),
        get_flag=fastgs_get_flag,
        metric_map=metric_map,
        antialiasing=bool(getattr(pipe, "antialiasing", False)),
        isbatched=is_batched,
        end_transmittance=0.0001,
        enable_timer=bool(getattr(pipe, "enable_timer", False)),
        return_matvec_kernels=bool(getattr(pipe, "return_matvec_kernels", False)),
        enable_error_check=bool(getattr(pipe, "enable_error_check", False)),
    )
    rasterizer = GaussianRasterizer(settings)
    dc, shs, colors_precomp = _colors(viewpoint_camera, pc, pipe)
    output = rasterizer(
        means3D=pc.get_xyz,
        means2D=screen,
        dc=dc,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None,
    )
    image, radii, depth, accum_metric_counts = _unpack(output, fastgs_get_flag=fastgs_get_flag)
    result = {
        "render": image.clamp(0.0, 1.0),
        "viewspace_points": screen,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
    }
    if accum_metric_counts is not None:
        result["accum_metric_counts"] = accum_metric_counts
    return result


def _settings(settings_type: type, **values: Any) -> Any:
    fields = getattr(settings_type, "_fields", ())
    if fields:
        values = {key: value for key, value in values.items() if key in fields}
    return settings_type(**values)


def _colors(viewpoint_camera: Any, pc: Any, pipe: Any) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if not bool(getattr(pipe, "convert_SHs_python", False)):
        return pc.get_features_dc, pc.get_features_rest, None
    from utils.sh_utils import eval_sh

    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    direction = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
    direction = direction / direction.norm(dim=1, keepdim=True)
    colors = torch.clamp_min(eval_sh(pc.active_sh_degree, shs_view, direction) + 0.5, 0.0)
    return None, None, colors


def _unpack(output: tuple[torch.Tensor, ...], *, fastgs_get_flag: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if len(output) == 3:
        if fastgs_get_flag:
            return output[0], output[1], None, output[2]
        return output[0], output[1], None, None
    return output[0], output[1], None, None
