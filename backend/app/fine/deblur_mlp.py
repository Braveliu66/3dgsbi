from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from app.fine.option_utils import read_float, read_int


DEBLURRING_3DGS_COMMIT = "e63366b8581c0fde2fda0ab1aea99518da2e2f10"


class FourierEmbedding(nn.Module):
    def __init__(self, input_dims: int, num_freqs: int) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.out_dim = input_dims * (1 + 2 * num_freqs)
        freq_bands = 2.0 ** torch.linspace(0.0, float(num_freqs - 1), steps=num_freqs)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = [inputs]
        for freq in self.freq_bands.to(device=inputs.device, dtype=inputs.dtype):
            encoded.append(torch.sin(inputs * freq))
            encoded.append(torch.cos(inputs * freq))
        return torch.cat(encoded, dim=-1)


def init_linear_weights(module: nn.Module) -> None:
    if not isinstance(module, nn.Linear):
        return
    gain = 0.1 if module.weight.shape[0] in {2, 3} else 1.0
    nn.init.xavier_normal_(module.weight, gain=gain)
    nn.init.constant_(module.bias, 0.0)


class GTnet(nn.Module):
    """Adapted from Deblurring-3DGS scene/blur_kernel.py at the pinned commit."""

    def __init__(
        self,
        *,
        res_pos: int = 3,
        res_view: int = 10,
        num_hidden: int = 3,
        width: int = 64,
        pos_delta: bool = False,
        num_moments: int = 4,
    ) -> None:
        super().__init__()
        self.pos_delta = pos_delta
        self.num_moments = num_moments
        self.embed_pos = FourierEmbedding(3, res_pos)
        self.embed_view = FourierEmbedding(3, res_view)
        input_dim = self.embed_pos.out_dim + self.embed_view.out_dim + 7

        layers: list[nn.Module] = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(max(0, num_hidden - 1)):
            layers.extend([nn.Linear(width, width), nn.ReLU()])
        self.linears = nn.Sequential(*layers)

        if pos_delta:
            self.s = nn.Linear(width, 3 * (num_moments + 1))
            self.r = nn.Linear(width, 4 * (num_moments + 1))
            self.p = nn.Linear(width, 3 * num_moments)
        else:
            self.s = nn.Linear(width, 3)
            self.r = nn.Linear(width, 4)
            self.p = None

        self.apply(init_linear_weights)

    def forward(
        self,
        pos: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        viewdirs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = torch.cat([self.embed_pos(pos), self.embed_view(viewdirs), scales, rotations], dim=-1)
        hidden = self.linears(x)
        pos_delta = self.p(hidden) if self.p is not None else None
        return self.s(hidden), self.r(hidden), pos_delta


@dataclass(slots=True)
class DeblurMLPConfig:
    mode: str
    use_position: bool
    hidden: int = 3
    width: int = 64
    num_moments: int = 4
    lambda_s: float = 0.01
    lambda_p: float = 0.01
    min_clamp: float = 0.9
    max_clamp: float = 1.1
    max_position_delta: float = 0.02
    transform_reg_weight: float = 0.001
    lr: float = 1e-3


@dataclass(slots=True)
class DeblurMLPState:
    config: DeblurMLPConfig | None
    model: GTnet | None = None

    @property
    def enabled(self) -> bool:
        return self.config is not None and self.model is not None

    def metrics(self) -> dict[str, Any]:
        if self.config is None:
            return {
                "deblur_mlp_enabled": False,
                "deblur_algorithm": "disabled",
            }
        return {
            "deblur_mlp_enabled": True,
            "deblur_algorithm": "Deblurring-3DGS_GTnet",
            "deblur_source_commit": DEBLURRING_3DGS_COMMIT,
            "deblur_mlp_mode": self.config.mode,
            "deblur_mlp_use_position_moments": self.config.use_position,
            "deblur_mlp_num_moments": self.config.num_moments,
            "deblur_mlp_lambda_s": self.config.lambda_s,
            "deblur_mlp_lambda_p": self.config.lambda_p,
            "deblur_mlp_min_clamp": self.config.min_clamp,
            "deblur_mlp_max_clamp": self.config.max_clamp,
            "deblur_mlp_max_position_delta": self.config.max_position_delta,
            "deblur_mlp_transform_reg_weight": self.config.transform_reg_weight,
            "deblur_mlp_lr": self.config.lr,
        }


def build_deblur_mlp_state(blur_mode: str, options: dict[str, Any], *, device: torch.device | str) -> DeblurMLPState:
    mode = str(options.get("fine_deblur_mode") or blur_mode or "sharp").lower()
    enabled_value = str(options.get("fine_deblur_enabled", "auto")).lower()
    if enabled_value in {"0", "false", "no", "off"}:
        return DeblurMLPState(None)
    if enabled_value not in {"1", "true", "yes", "on"} and mode == "sharp":
        return DeblurMLPState(None)

    use_position = mode in {"motion", "mixed"}
    explicit_use_position = options.get("fine_deblur_use_position")
    if explicit_use_position is not None:
        use_position = str(explicit_use_position).lower() in {"1", "true", "yes", "on"}

    config = DeblurMLPConfig(
        mode=mode,
        use_position=use_position,
        hidden=read_int(options.get("fine_deblur_hidden"), 3, minimum=1, maximum=8),
        width=read_int(options.get("fine_deblur_width"), 64, minimum=16, maximum=256),
        num_moments=read_int(options.get("fine_deblur_num_moments"), 4, minimum=1, maximum=8),
        lambda_s=read_float(options.get("fine_deblur_lambda_s"), 0.01, minimum=0.0, maximum=0.1),
        lambda_p=read_float(options.get("fine_deblur_lambda_p"), 0.01, minimum=0.0, maximum=0.1),
        min_clamp=read_float(options.get("fine_deblur_min_clamp"), 0.9, minimum=0.5, maximum=1.0),
        max_clamp=read_float(options.get("fine_deblur_max_clamp"), 1.1, minimum=1.0, maximum=1.8),
        max_position_delta=read_float(options.get("fine_deblur_max_position_delta"), 0.02, minimum=0.0, maximum=1.0),
        transform_reg_weight=read_float(options.get("fine_deblur_transform_reg_weight"), 0.001, minimum=0.0, maximum=1.0),
        lr=read_float(options.get("fine_deblur_gtnet_lr"), 1e-3, minimum=1e-6, maximum=1e-1),
    )
    model = GTnet(
        num_hidden=config.hidden,
        width=config.width,
        pos_delta=config.use_position,
        num_moments=config.num_moments,
    ).to(device)
    return DeblurMLPState(config=config, model=model)


def attach_deblur_mlp_optimizer(gaussians: Any, state: DeblurMLPState) -> None:
    if not state.enabled:
        return
    if gaussians.optimizer is None:
        raise RuntimeError("Gaussian optimizer must be initialized before attaching GTnet")
    patch_gaussian_topology_optimizer(gaussians)
    gaussians.optimizer.add_param_group(
        {
            "params": list(state.model.parameters()),
            "lr": state.config.lr,
            "name": "GTnet",
        }
    )


def patch_gaussian_topology_optimizer(gaussians: Any) -> None:
    if bool(getattr(gaussians, "_deblur_mlp_topology_patch", False)):
        return
    original_prune = gaussians._prune_optimizer
    original_cat = gaussians.cat_tensors_to_optimizer

    def without_gtnet_groups(callback: Any, *args: Any, **kwargs: Any) -> Any:
        optimizer = gaussians.optimizer
        skipped = [group for group in optimizer.param_groups if group.get("name") == "GTnet"]
        if not skipped:
            return callback(*args, **kwargs)
        optimizer.param_groups[:] = [group for group in optimizer.param_groups if group.get("name") != "GTnet"]
        try:
            return callback(*args, **kwargs)
        finally:
            optimizer.param_groups.extend(skipped)

    def prune_optimizer(mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return without_gtnet_groups(original_prune, mask)

    def cat_tensors_to_optimizer(tensors_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return without_gtnet_groups(original_cat, tensors_dict)

    gaussians._prune_optimizer = prune_optimizer
    gaussians.cat_tensors_to_optimizer = cat_tensors_to_optimizer
    gaussians._deblur_mlp_topology_patch = True


def predict_deblur_transforms(
    state: DeblurMLPState,
    means3d: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    camera_center: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not state.enabled:
        raise RuntimeError("DeblurMLPState is disabled")
    config = state.config
    viewdirs = camera_center.repeat(means3d.shape[0], 1)
    scale_delta, rotation_delta, position_delta = state.model(
        means3d.detach(),
        scales.detach(),
        rotations.detach(),
        viewdirs,
    )
    scale_delta = torch.clamp(1.0 + config.lambda_s * scale_delta, min=config.min_clamp, max=config.max_clamp)
    rotation_delta = torch.clamp(1.0 + config.lambda_s * rotation_delta, min=config.min_clamp, max=config.max_clamp)
    if position_delta is not None:
        position_delta = torch.clamp(
            config.lambda_p * position_delta,
            min=-config.max_position_delta,
            max=config.max_position_delta,
        )
    return scale_delta, rotation_delta, position_delta



def deblur_transform_regularization(
    scale_delta: torch.Tensor,
    rotation_delta: torch.Tensor,
    position_delta: torch.Tensor | None,
) -> torch.Tensor:
    reg = torch.mean((scale_delta - 1.0) ** 2) + torch.mean((rotation_delta - 1.0) ** 2)
    if position_delta is not None:
        reg = reg + torch.mean(position_delta**2)
    return reg


def render_with_deblur_mlp(viewpoint_camera: Any, pc: Any, pipe: Any, bg_color: torch.Tensor, state: DeblurMLPState) -> dict[str, torch.Tensor]:
    if not state.enabled:
        raise RuntimeError("DeblurMLPState is disabled")

    from diff_gaussian_rasterization import GaussianRasterizer

    rasterizer = GaussianRasterizer(raster_settings=_raster_settings(viewpoint_camera, pc, pipe, bg_color))
    means3d = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs, colors_precomp = _resolve_colors(viewpoint_camera, pc, pipe)
    scale_delta, rotation_delta, position_delta = predict_deblur_transforms(
        state,
        means3d,
        scales,
        rotations,
        viewpoint_camera.camera_center,
    )

    if not state.config.use_position:
        screen = _screen_space_points(means3d)
        output = rasterizer(
            means3D=means3d,
            means2D=screen,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales * scale_delta,
            rotations=rotations * rotation_delta,
            cov3D_precomp=None,
        )
        image, radii = _unpack_rasterizer_output(output)
        return {
            "render": image,
            "viewspace_points": screen,
            "visibility_filter": radii > 0,
            "radii": radii,
            "deblur_regularization": deblur_transform_regularization(scale_delta, rotation_delta, None),
        }

    moments = int(state.config.num_moments)
    position_delta = position_delta.view(-1, 3, moments)
    scale_delta = scale_delta.view(-1, 3, moments + 1)
    rotation_delta = rotation_delta.view(-1, 4, moments + 1)

    renders = []
    radii_accum = None
    visibility_accum = None
    first_screen = None
    for moment in range(moments + 1):
        screen = _screen_space_points(means3d)
        if first_screen is None:
            first_screen = screen
        if moment == 0:
            transformed_pos = means3d
            delta_index = moments
        else:
            delta_index = moment - 1
            transformed_pos = means3d + position_delta[..., delta_index]
        output = rasterizer(
            means3D=transformed_pos,
            means2D=screen,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales * scale_delta[..., delta_index],
            rotations=rotations * rotation_delta[..., delta_index],
            cov3D_precomp=None,
        )
        image, radii = _unpack_rasterizer_output(output)
        renders.append(image)
        radii_accum = radii if radii_accum is None else torch.maximum(radii_accum, radii)
        visible = radii > 0
        visibility_accum = visible if visibility_accum is None else torch.logical_or(visibility_accum, visible)

    return {
        "render": sum(renders) / len(renders),
        "viewspace_points": first_screen,
        "visibility_filter": visibility_accum,
        "radii": radii_accum,
        "deblur_regularization": deblur_transform_regularization(scale_delta, rotation_delta, position_delta),
    }


def _raster_settings(viewpoint_camera: Any, pc: Any, pipe: Any, bg_color: torch.Tensor) -> Any:
    from diff_gaussian_rasterization import GaussianRasterizationSettings

    values = {
        "image_height": int(viewpoint_camera.image_height),
        "image_width": int(viewpoint_camera.image_width),
        "tanfovx": math.tan(viewpoint_camera.FoVx * 0.5),
        "tanfovy": math.tan(viewpoint_camera.FoVy * 0.5),
        "bg": bg_color,
        "scale_modifier": 1.0,
        "viewmatrix": viewpoint_camera.world_view_transform,
        "projmatrix": viewpoint_camera.full_proj_transform,
        "sh_degree": pc.active_sh_degree,
        "campos": viewpoint_camera.camera_center,
        "prefiltered": False,
        "debug": bool(getattr(pipe, "debug", False)),
        "antialiasing": bool(getattr(pipe, "antialiasing", False)),
        "isbatched": False,
        "end_transmittance": 0.0001,
        "enable_timer": bool(getattr(pipe, "enable_timer", False)),
        "return_matvec_kernels": bool(getattr(pipe, "return_matvec_kernels", False)),
        "enable_error_check": bool(getattr(pipe, "enable_error_check", False)),
    }
    fields = getattr(GaussianRasterizationSettings, "_fields", ())
    if fields:
        values = {key: value for key, value in values.items() if key in fields}
    return GaussianRasterizationSettings(**values)


def _screen_space_points(points: torch.Tensor) -> torch.Tensor:
    screen = torch.zeros_like(points, dtype=points.dtype, requires_grad=True, device=points.device) + 0
    try:
        screen.retain_grad()
    except Exception:
        pass
    return screen


def _resolve_colors(viewpoint_camera: Any, pc: Any, pipe: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not bool(getattr(pipe, "convert_SHs_python", False)):
        return pc.get_features, None

    from utils.sh_utils import eval_sh

    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    return None, torch.clamp_min(sh2rgb + 0.5, 0.0)

def _unpack_rasterizer_output(output: object) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output, (tuple, list)):
        raise RuntimeError(f"GaussianRasterizer returned unsupported output type: {type(output)!r}")
    if len(output) < 2:
        raise RuntimeError(f"GaussianRasterizer returned {len(output)} values, expected at least 2")
    return output[0], output[1]