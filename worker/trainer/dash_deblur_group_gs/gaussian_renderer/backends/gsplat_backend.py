import math

import torch

from utils.graphics_utils import fov2focal


def gsplat_rasterize(viewpoint_camera, pc, bg_color, scaling_modifier=1.0):
    try:
        from gsplat import rasterization
    except Exception as exc:
        raise RuntimeError("renderer_backend=gsplat requires the gsplat Python package and CUDA kernels") from exc

    means = pc.get_xyz
    quats = pc.get_rotation
    scales = pc.get_scaling * scaling_modifier
    opacities = pc.get_opacity.squeeze(-1)
    colors = pc.get_features.contiguous()

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).contiguous().unsqueeze(0)
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    fx = fov2focal(viewpoint_camera.FoVx, width)
    fy = fov2focal(viewpoint_camera.FoVy, height)
    K = torch.tensor(
        [[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=means.dtype,
        device=means.device,
    ).unsqueeze(0)

    rendered, _alpha, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=width,
        height=height,
        sh_degree=pc.active_sh_degree,
        packed=False,
        backgrounds=bg_color.reshape(1, 3),
        camera_model="pinhole",
    )

    image = rendered[0].permute(2, 0, 1).contiguous()
    radii = _training_radii(meta.get("radii"), means)
    means2d = _training_means2d(meta.get("means2d"), means)

    if not torch.isfinite(image).all():
        raise RuntimeError("renderer_backend=gsplat produced non-finite pixels")
    if math.prod(image.shape) == 0:
        raise RuntimeError("renderer_backend=gsplat produced an empty image")

    return image, radii, means2d


def _training_radii(radii, means):
    point_count = int(means.shape[0])
    if radii is None:
        return torch.zeros((point_count,), dtype=means.dtype, device=means.device)
    if radii.dim() >= 2 and radii.shape[0] == 1:
        radii = radii[0]
    if radii.dim() >= 2 and radii.shape[-1] == 2:
        radii = radii.max(dim=-1).values
    radii = radii.reshape(-1)
    if radii.numel() != point_count:
        raise RuntimeError(f"renderer_backend=gsplat returned radii for {radii.numel()} entries, expected {point_count}")
    return radii


def _training_means2d(means2d, means):
    point_count = int(means.shape[0])
    if means2d is not None:
        if means2d.dim() >= 3 and means2d.shape[0] == 1:
            means2d = means2d[0]
        if means2d.dim() >= 3 and means2d.shape[-2] == 1:
            means2d = means2d.squeeze(-2)
        if means2d.shape[0] != point_count:
            raise RuntimeError(f"renderer_backend=gsplat returned means2d for {means2d.shape[0]} entries, expected {point_count}")
        if means2d.shape[-1] == 2:
            means2d = torch.cat([means2d, torch.zeros_like(means2d[:, :1])], dim=-1)
        try:
            means2d.retain_grad()
        except RuntimeError:
            pass
        return means2d

    fallback = torch.zeros_like(means, dtype=means.dtype, requires_grad=True, device=means.device) + 0
    try:
        fallback.retain_grad()
    except RuntimeError:
        pass
    return fallback
