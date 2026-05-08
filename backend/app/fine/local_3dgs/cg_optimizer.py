from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from app.fine.local_3dgs.cg_solver import LocalCGSolver


C0 = 0.28209479177387814
MAX_COLOR = 0.5 / C0
MIN_COLOR = -MAX_COLOR


@dataclass(slots=True)
class ResidualState:
    mse_residuals: torch.Tensor
    ssim_residuals: torch.Tensor
    ssim_derivatives: torch.Tensor


class LocalCGOptimizer:
    def __init__(self, gaussians: Any, options: Any, *, scene_extent: float) -> None:
        self.options = options
        self.fixed_lr = float(getattr(options, "fixed_lr", 0.1))
        self.max_lr = torch.tensor(float(getattr(options, "max_lr", 0.2)), device="cuda")
        self.auto_lr = bool(getattr(options, "auto_lr", False))
        self.scene_extent = float(scene_extent)
        self.batch_size = int(getattr(options, "batch_size", 1))
        width = int(gaussians.cgState.width)
        height = int(gaussians.cgState.height)
        channels = 3
        self.residual_state = ResidualState(
            mse_residuals=torch.zeros((self.batch_size, width * height * channels), dtype=torch.float32, device="cuda"),
            ssim_residuals=torch.zeros((self.batch_size, width * height * channels), dtype=torch.float32, device="cuda"),
            ssim_derivatives=torch.zeros((self.batch_size, width * height * channels), dtype=torch.float32, device="cuda"),
        )
        self.losses = torch.zeros((self.batch_size,), dtype=torch.float32, device="cuda")
        self.solver = LocalCGSolver(
            int(gaussians.get_xyz.shape[0]),
            linear_iter=int(getattr(options, "cg_iter", 8)),
            lambda_reg=float(getattr(options, "regularizer", 0.01)),
            levenberg_type=str(getattr(options, "levenberg_type", "identity")),
        )
        self.solution: torch.Tensor | None = None
        self.optim_iter = 1

    def append_residual(self, image: torch.Tensor, gt_image: torch.Tensor, current_batch: int) -> torch.Tensor:
        with torch.no_grad():
            residual = gt_image - image
            self.residual_state.mse_residuals[current_batch] = residual.permute(1, 2, 0).flatten()
            self.losses[current_batch] = torch.sum(residual * residual)
            return residual

    def linear_solve(self, gaussians: Any, *, return_matvec_kernels: bool = False) -> torch.Tensor:
        from diff_gaussian_rasterization import _RasterizeGaussians

        self.solver.resize(int(gaussians.get_xyz.shape[0]))
        _RasterizeGaussians.set_residual_state(self.residual_state, gaussians.cgState.lambda_dssim)
        _RasterizeGaussians.set_backward_inputs(gaussians.cgState)
        self.solution = self.solver.solve(debug=return_matvec_kernels)
        return self.losses.mean()

    def step(self, gaussians: Any) -> None:
        if self.solution is None:
            return
        with torch.no_grad():
            n = int(gaussians.get_xyz.shape[0])
            param_names = ("opacity", "dc", "xyz", "scale", "rotation")
            param_dims = (1, 3, 3, 3, 4)
            params = (gaussians._opacity, gaussians._features_dc, gaussians._xyz, gaussians._scaling, gaussians._rotation)
            offset = 0
            lr = self.fixed_lr
            if self.auto_lr:
                color_delta = self.solution[n : n + 3 * n].abs().max().clamp_min(1e-8)
                lr = float(torch.minimum(self.max_lr, 1.0 / color_delta).item())
            for name, dim, param in zip(param_names, param_dims, params):
                chunk = self.solution[offset : offset + dim * n].view(dim, n).T.contiguous().view_as(param)
                param.add_(chunk * lr)
                if name == "dc":
                    param.clamp_(min=MIN_COLOR, max=MAX_COLOR)
                elif name == "scale":
                    param.clamp_(max=self.scene_extent)
                offset += dim * n
            self.optim_iter += 1

