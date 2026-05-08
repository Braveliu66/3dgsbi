from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from app.config import PROJECT_ROOT
from app.fine.option_utils import read_float, read_int
from app.fine.types import FineFailure


LMRS_ROOT_ENV = "LMRS_ROOT"
MATRIX_FREE_SYMBOLS = ("get_JTv", "get_Diag", "get_JTJv")


@dataclass(slots=True)
class LmrsPhase:
    optimizer: Any
    pixel_sampler: Any
    camera_sampler: Any
    started_at: int
    iterations: int = 0
    last_loss: float | None = None


class MobileRandomCameraSampler:
    def __init__(self, train_cameras: list[Any]) -> None:
        self.train_cameras = train_cameras
        self.viewpoint_stack: list[Any] = []

    def get_camera(self, current_batch: int) -> Any:
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.train_cameras.copy()
        return self.viewpoint_stack.pop(random.randrange(len(self.viewpoint_stack)))

    def update(self, key: str, value: Any) -> None:
        return None


class MobileUniformPixelSampler:
    def sample(self, current_batch: int, cgState, **kwargs: Any) -> None:
        sample_size = cgState.sample_per_block
        total_tiles = cgState.total_blocks_sampled
        max_pixels = cgState.tile_block_dim[0] * cgState.tile_block_dim[1]
        sampled_pixels = torch.randint(low=0, high=max_pixels, size=(total_tiles, sample_size), device="cuda")
        cgState.state["sampled_pixels"][current_batch] = sampled_pixels
        cgState.state["likelihoods"][current_batch] = 1.0 / float(max_pixels)


def build_lmrs_options(options: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        cg_iter=read_int(options.get("fine_lmrs_cg_iter"), 8, minimum=1, maximum=64),
        regularizer=read_float(options.get("fine_lmrs_regularizer"), 0.01, minimum=1e-8, maximum=10.0),
        batch_size=read_int(options.get("fine_lmrs_batch_size"), 1, minimum=1, maximum=16),
        linear_solver="CG",
        kernel=1,
        ssim_weight=read_float(options.get("fine_lambda_dssim"), 0.2, minimum=0.0, maximum=1.0),
        fixed_lr=read_float(options.get("fine_lmrs_fixed_lr"), 0.1, minimum=1e-6, maximum=10.0),
        likelihood_viz_freq=-1,
        loss_fn="mse",
        end_transmittance=0.0001,
        camera_sampler="mobile_random",
        N_sample_per_tile=read_int(options.get("fine_lmrs_samples_per_tile"), 32, minimum=1, maximum=256),
        tile_block_dimx=16,
        tile_block_dimy=16,
        max_lr=read_float(options.get("fine_lmrs_max_lr"), 0.2, minimum=1e-6, maximum=10.0),
        auto_lr=False,
        sampling_distribution="mobile_uniform",
        disable_scheds=True,
        temperature=1.0,
        levenberg_type="identity",
    )


def initialize_lmrs_phase(
    *,
    gaussians: Any,
    scene: Any,
    opt: SimpleNamespace,
    gn_opt: SimpleNamespace,
    cameras: list[Any],
    output_dir: Path,
    started_at: int,
    cg_optimizer_cls: Any,
) -> LmrsPhase:
    ensure_lmrs_matrix_free_symbols()
    gaussians.optimizer.zero_grad(set_to_none=True)
    gaussians.optimization_method = "cg-gpu"
    gaussians.training_setup(opt, gn_opt)
    gaussians.cgState.set_scene_size(scene)
    optimizer = cg_optimizer_cls(gaussians, gn_opt, str(output_dir), scene.cameras_extent)
    optimizer.solver.set_linear_iter(gn_opt.cg_iter)
    return LmrsPhase(
        optimizer=optimizer,
        pixel_sampler=MobileUniformPixelSampler(),
        camera_sampler=MobileRandomCameraSampler(cameras),
        started_at=started_at,
    )


def ensure_lmrs_matrix_free_symbols() -> None:
    module = __import__("diff_gaussian_rasterization")
    extension = getattr(module, "_C", None)
    missing = [name for name in MATRIX_FREE_SYMBOLS if not hasattr(extension, name)]
    if missing:
        raise FineFailure("LMRS_MATRIX_FREE_UNAVAILABLE", f"LM-RS matrix-free rasterizer symbols missing: {', '.join(missing)}")


def compact_box_status() -> str:
    try:
        module = __import__("diff_gaussian_rasterization")
        if bool(getattr(module, "MOBILEGS_COMPACT_BOX", False)):
            return "mobilegs_lmrs_fastgs_compact_box"
    except Exception:
        pass
    return "mobilegs_lmrs_compact_box_marker_missing"


def resolve_lmrs_root() -> Path:
    candidates = []
    env_value = os.getenv(LMRS_ROOT_ENV)
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(
        [
            PROJECT_ROOT / "backend" / "app" / "fine" / "vendor" / "lm-rs",
            PROJECT_ROOT / "repo-cache" / "lm-rs",
            Path("/opt/lm-rs"),
        ]
    )
    for candidate in candidates:
        if (candidate / "gaussian_renderer").is_dir() and (candidate / "scene").is_dir():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FineFailure("LMRS_RUNTIME_UNAVAILABLE", f"LM-RS source tree not found; searched: {searched}")


def resolve_lm_status(lm_start_iter: int, iterations: int) -> dict[str, Any]:
    if lm_start_iter >= iterations:
        return {"active": False, "start_iter": lm_start_iter, "reason": "LM phase disabled because start_iter >= iterations"}
    try:
        ensure_lmrs_matrix_free_symbols()
    except Exception as exc:
        raise FineFailure("LMRS_MATRIX_FREE_UNAVAILABLE", f"LM-RS Phase 2 requested but rasterizer is unavailable: {exc}") from exc
    return {
        "active": True,
        "backend": "lmrs_matrix_free",
        "start_iter": lm_start_iter,
        "cg_iter": None,
        "reason": None,
    }
