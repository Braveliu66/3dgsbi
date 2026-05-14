from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    def __init__(self, input_dims: int, num_freqs: int) -> None:
        super().__init__()
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
    """GTnet/Fourier embedding adapted from Deblurring-3DGS."""

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
        position_delta = self.p(hidden) if self.p is not None else None
        return self.s(hidden), self.r(hidden), position_delta


@dataclass(slots=True)
class DeblurConfig:
    mode: str = "sharp"
    use_position: bool = False
    hidden: int = 3
    width: int = 64
    num_moments: int = 4
    lambda_s: float = 0.01
    lambda_p: float = 0.01
    min_clamp: float = 1.0
    max_clamp: float = 1.1
    max_position_delta: float = 0.02
    transform_reg_weight: float = 0.001
    lr: float = 1e-3


@dataclass(slots=True)
class DeblurState:
    config: DeblurConfig | None = None
    model: GTnet | None = None

    @property
    def enabled(self) -> bool:
        return self.config is not None and self.model is not None

    def metrics(self) -> dict[str, Any]:
        if not self.enabled:
            return {"deblur_mlp_enabled": False, "deblur_algorithm": "disabled"}
        assert self.config is not None
        return {
            "deblur_mlp_enabled": True,
            "deblur_algorithm": "Deblurring-3DGS_GTnet_fastgs",
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


def build_deblur_state(opt: Any, *, device: torch.device | str = "cuda") -> DeblurState:
    enabled_value = str(getattr(opt, "deblur_enabled", "auto")).strip().lower()
    mode = str(getattr(opt, "deblur_mode", "sharp")).strip().lower()
    if enabled_value in {"0", "false", "no", "off"}:
        return DeblurState()
    if enabled_value not in {"1", "true", "yes", "on"} and mode == "sharp":
        return DeblurState()
    if getattr(opt, "optimizer_type", "default") != "default":
        return DeblurState()

    use_position = mode in {"motion", "mixed"}
    config = DeblurConfig(
        mode=mode,
        use_position=use_position,
        hidden=int(getattr(opt, "deblur_hidden", 3)),
        width=int(getattr(opt, "deblur_width", 64)),
        num_moments=int(getattr(opt, "deblur_num_moments", 4)),
        lambda_s=float(getattr(opt, "deblur_lambda_s", 0.01)),
        lambda_p=float(getattr(opt, "deblur_lambda_p", 0.01)),
        max_clamp=float(getattr(opt, "deblur_max_clamp", 1.1)),
        max_position_delta=float(getattr(opt, "deblur_max_position_delta", 0.02)),
        transform_reg_weight=float(getattr(opt, "deblur_transform_reg_weight", 0.001)),
        lr=float(getattr(opt, "deblur_gtnet_lr", 1e-3)),
    )
    model = GTnet(
        num_hidden=config.hidden,
        width=config.width,
        pos_delta=config.use_position,
        num_moments=config.num_moments,
    ).to(device)
    return DeblurState(config=config, model=model)


def predict_deblur_transforms(
    state: DeblurState,
    means3d: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    camera_center: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not state.enabled:
        raise RuntimeError("DeblurState is disabled")
    assert state.config is not None and state.model is not None
    viewdirs = camera_center.reshape(1, 3).repeat(means3d.shape[0], 1)
    scale_delta, rotation_delta, position_delta = state.model(
        means3d.detach(),
        scales.detach(),
        rotations.detach(),
        viewdirs,
    )
    scale_delta = torch.clamp(
        1.0 + state.config.lambda_s * scale_delta,
        min=state.config.min_clamp,
        max=state.config.max_clamp,
    )
    rotation_delta = torch.clamp(
        1.0 + state.config.lambda_s * rotation_delta,
        min=state.config.min_clamp,
        max=state.config.max_clamp,
    )
    if position_delta is not None:
        position_delta = torch.clamp(
            state.config.lambda_p * position_delta,
            min=-state.config.max_position_delta,
            max=state.config.max_position_delta,
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
