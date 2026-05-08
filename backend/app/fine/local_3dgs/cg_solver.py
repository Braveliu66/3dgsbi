from __future__ import annotations

import torch

from app.fine.types import FineFailure


def _rasterize_gaussians():
    try:
        from diff_gaussian_rasterization import _RasterizeGaussians
    except Exception as exc:  # pragma: no cover - worker/runtime dependent
        raise FineFailure("LMRS_MATRIX_FREE_UNAVAILABLE", f"LM-RS rasterizer wrapper unavailable: {exc}") from exc
    for name in ("get_JTv", "get_Diag", "get_JTJv"):
        if not hasattr(_RasterizeGaussians, name):
            raise FineFailure("LMRS_MATRIX_FREE_UNAVAILABLE", f"LM-RS rasterizer symbol missing: {name}")
    return _RasterizeGaussians


class LocalCGSolver:
    def __init__(self, gaussian_count: int, *, linear_iter: int, lambda_reg: float, levenberg_type: str = "identity") -> None:
        if levenberg_type not in {"identity", "diagonal"}:
            raise FineFailure("LMRS_UNSUPPORTED_REGULARIZER", f"Unsupported LM regularizer: {levenberg_type}")
        self.gaussian_count = int(gaussian_count)
        self.size = self.gaussian_count * 14
        self.linear_iter = int(linear_iter)
        self.lambda_reg = float(lambda_reg)
        self.levenberg_type = levenberg_type
        self.x = torch.zeros((self.size,), dtype=torch.float32, device="cuda")
        self.r = torch.zeros_like(self.x)
        self.diag = torch.zeros_like(self.x)
        self.p = torch.zeros_like(self.x)
        self.Ap = torch.zeros_like(self.x)
        self.z = torch.zeros_like(self.x)
        self.inv_diag = torch.zeros_like(self.x)
        self.solution: torch.Tensor | None = None

    def resize(self, gaussian_count: int) -> None:
        if int(gaussian_count) == self.gaussian_count:
            return
        self.__init__(gaussian_count, linear_iter=self.linear_iter, lambda_reg=self.lambda_reg, levenberg_type=self.levenberg_type)

    def set_linear_iter(self, value: int) -> None:
        self.linear_iter = int(value)

    def solve(self, *, debug: bool = False) -> torch.Tensor:
        raster = _rasterize_gaussians()
        self.x.zero_()
        self.r.zero_()
        self.diag.zero_()
        self.Ap.zero_()
        raster.get_JTv(self.r, self.gaussian_count)
        raster.get_Diag(self.diag, self.gaussian_count)
        if debug:
            raster.get_JTJv(self.r, self.Ap, self.gaussian_count)
            self.solution = self.r.clone()
            return self.solution

        if self.levenberg_type == "identity":
            self.diag.add_(self.lambda_reg)
        else:
            self.diag.add_(self.diag * self.lambda_reg).add_(1e-6)
        torch.div(1.0, self.diag.clamp_min(1e-8), out=self.inv_diag)
        torch.mul(self.inv_diag, self.r, out=self.z)
        self.p.copy_(self.z)
        r_dot_z_old = torch.dot(self.r, self.z)
        for _ in range(max(1, self.linear_iter)):
            self.Ap.zero_()
            raster.get_JTJv(self.p, self.Ap, self.gaussian_count)
            if self.levenberg_type == "identity":
                self.Ap.add_(self.p * self.lambda_reg)
            else:
                self.Ap.add_(self.p * self.lambda_reg * self.diag)
            denom = torch.dot(self.p, self.Ap).clamp_min(1e-12)
            alpha = r_dot_z_old / denom
            self.x.add_(self.p, alpha=alpha)
            self.r.add_(self.Ap, alpha=-alpha)
            torch.mul(self.inv_diag, self.r, out=self.z)
            r_dot_z_new = torch.dot(self.r, self.z)
            beta = r_dot_z_new / r_dot_z_old.clamp_min(1e-12)
            self.p.mul_(beta).add_(self.z)
            r_dot_z_old = r_dot_z_new
        self.solution = self.x.clone()
        return self.solution

