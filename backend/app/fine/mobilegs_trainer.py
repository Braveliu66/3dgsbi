from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Any

import torch

from app.fine.deblur_mlp import attach_deblur_mlp_optimizer, build_deblur_mlp_state, render_with_deblur_mlp
from app.fine.fastgs_policy import FastGSPolicy, vcp_min_opacity
from app.fine.lmrs_runtime import (
    LmrsPhase,
    build_lmrs_options,
    compact_box_status,
    initialize_lmrs_phase,
    resolve_lm_status,
    resolve_lmrs_root,
)
from app.fine.option_utils import read_float, read_int
from app.fine.types import FineFailure
from app.preview.utils import prepend_sys_path


Progress = Callable[[str, int, str], None]


@dataclass(slots=True)
class MobileGSTrainResult:
    ply_path: Path
    iterations: int
    point_count: int
    metrics: dict[str, Any] = field(default_factory=dict)


def train_mobile_3dgs(
    *,
    scene_dir: Path,
    output_dir: Path,
    iterations: int,
    lm_start_iter: int,
    blur_mode: str,
    options: dict[str, Any],
    progress: Progress,
) -> MobileGSTrainResult:
    if not torch.cuda.is_available():
        raise FineFailure("GPU_RESOURCE_UNAVAILABLE", "CUDA GPU is required for fine 3DGS training")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    training_root = resolve_lmrs_root()
    with prepend_sys_path(training_root):
        from gaussian_renderer import render
        from scene import GaussianModel, Scene
        from scene.optim_strategy.cgOptimizer import CGOptimizer
        from scene.step.gauss_newton_step import gauss_newton_step
        from utils.loss_utils import l1_loss, ssim

        opt = build_optimization_options(iterations, options)
        gn_opt = build_lmrs_options(options)
        dataset = SimpleNamespace(
            source_path=str(scene_dir.resolve()),
            model_path=str(output_dir.resolve()),
            images="images",
            depths="",
            resolution=read_int(options.get("fine_train_resolution"), -1, minimum=-1, maximum=4096),
            train_test_exp=False,
            data_device="cuda",
            eval=False,
            sh_degree=3,
            white_background=False,
        )
        pipe = SimpleNamespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
            antialiasing=False,
            enable_timer=False,
            return_matvec_kernels=False,
            enable_error_check=False,
        )
        gaussians = GaussianModel(3, "adam")
        scene = Scene(dataset, gaussians, opt, shuffle=True)
        gaussians.training_setup(opt, gn_opt)
        background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        deblur_state = build_deblur_mlp_state(blur_mode, options, device=background.device)
        attach_deblur_mlp_optimizer(gaussians, deblur_state)
        cameras = scene.getTrainCameras().copy()
        if not cameras:
            raise FineFailure("FINE_CAMERA_LOAD_FAILED", "3DGS scene contains no train cameras")

        policy = FastGSPolicy(
            percentile=read_float(options.get("fine_vcd_percentile"), 0.60, minimum=0.05, maximum=0.95),
            window=read_int(options.get("fine_vcd_window"), 5, minimum=1, maximum=20),
        )
        lm_status = resolve_lm_status(lm_start_iter, iterations)
        if lm_status["active"]:
            lm_status["cg_iter"] = gn_opt.cg_iter
            lm_status["batch_size"] = gn_opt.batch_size
            lm_status["samples_per_tile"] = gn_opt.N_sample_per_tile
        lm_phase: LmrsPhase | None = None
        lm_elapsed = 0.0
        ema_loss = 0.0
        last_l1 = 0.0
        last_ssim = 0.0
        train_stack = cameras.copy()

        for iteration in range(1, iterations + 1):
            in_lm_phase = bool(lm_status["active"] and iteration >= lm_start_iter)
            if not in_lm_phase:
                gaussians.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
            if in_lm_phase:
                if lm_phase is None:
                    lm_phase = initialize_lmrs_phase(
                        gaussians=gaussians,
                        scene=scene,
                        opt=opt,
                        gn_opt=gn_opt,
                        cameras=cameras,
                        output_dir=output_dir,
                        started_at=iteration,
                        cg_optimizer_cls=CGOptimizer,
                    )
                    progress("fine_lmrs_phase_start", 72, f"switched to LM-RS matrix-free CG at iteration {iteration}")
                lm_loss, elapsed = gauss_newton_step(
                    lm_phase.pixel_sampler,
                    lm_phase.camera_sampler,
                    iteration,
                    -1,
                    opt,
                    gaussians,
                    pipe,
                    background,
                    lm_phase.optimizer,
                    gn_opt,
                )
                lm_phase.optimizer.step(gaussians)
                lm_phase.iterations += 1
                lm_phase.last_loss = float(lm_loss)
                lm_elapsed += float(elapsed)
                ema_loss = 0.4 * float(lm_loss) + 0.6 * ema_loss
                last_l1 = float(lm_loss)
                last_ssim = 0.0
            else:
                if not train_stack:
                    train_stack = cameras.copy()
                viewpoint = train_stack.pop(random.randrange(len(train_stack)))
                gt_image = viewpoint.original_image.to("cuda")

                sharp_l1 = None
                if deblur_state.enabled:
                    with torch.no_grad():
                        sharp_pkg = render(viewpoint, gaussians, pipe, background, scaling_modifier=1.0)
                        sharp_l1 = float(l1_loss(sharp_pkg["render"], gt_image).item())

                if deblur_state.enabled:
                    render_pkg = render_with_deblur_mlp(viewpoint, gaussians, pipe, background, deblur_state)
                else:
                    render_pkg = render(viewpoint, gaussians, pipe, background, scaling_modifier=1.0)
                image = render_pkg["render"]
                l1_value = l1_loss(image, gt_image)
                ssim_value = 1.0 - ssim(image, gt_image)
                loss = (1.0 - opt.lambda_dssim) * l1_value + opt.lambda_dssim * ssim_value
                gaussians.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                with torch.no_grad():
                    if iteration < min(opt.densify_until_iter, lm_start_iter):
                        visible = render_pkg["visibility_filter"]
                        gaussians.max_radii2D[visible] = torch.max(gaussians.max_radii2D[visible], render_pkg["radii"][visible])
                        gaussians.add_densification_stats(render_pkg["viewspace_points"], visible)
                        policy.observe(gaussians.get_xyz.shape[0], visible, sharp_l1 if sharp_l1 is not None else float(l1_value.item()))

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                ema_loss = 0.4 * float(loss.item()) + 0.6 * ema_loss
                last_l1 = float(l1_value.item())
                last_ssim = float(ssim_value.item())

                with torch.no_grad():
                    if iteration < min(opt.densify_until_iter, lm_start_iter):
                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            original_accum = policy.apply_vcd_gate(gaussians)
                            min_opacity = vcp_min_opacity(policy)
                            gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity, scene.cameras_extent, size_threshold, "adam")
                            if original_accum is not None:
                                policy.reset_after_topology_change()
                        if iteration % opt.opacity_reset_interval == 0:
                            gaussians.reset_opacity("adam")

            if iteration % max(50, min(500, iterations // 20 or 50)) == 0 or iteration == iterations:
                mapped_progress = 42 + int(iteration / max(1, iterations) * 42)
                progress("fine_mobilegs_train", min(84, mapped_progress), f"trained {iteration}/{iterations} iterations, loss={ema_loss:.5f}")

        scene.save(iterations)
        ply_path = output_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        if not ply_path.exists() or ply_path.stat().st_size <= 0:
            raise FineFailure("FINE_PLY_NOT_FOUND", f"3DGS training did not produce point cloud: {ply_path}")
        elapsed = round(time.monotonic() - started, 3)
        return MobileGSTrainResult(
            ply_path=ply_path,
            iterations=iterations,
            point_count=int(gaussians.get_xyz.shape[0]),
            metrics={
                "training_backend": "mobilegs_adam_lmrs_matrix_free" if lm_phase else "mobilegs_adam",
                "lmrs_root": str(training_root),
                "training_elapsed_seconds": elapsed,
                "final_gaussians": int(gaussians.get_xyz.shape[0]),
                "last_l1_loss": last_l1,
                "last_ssim_loss": last_ssim,
                "ema_loss": ema_loss,
                "deblur_mode": blur_mode,
                **deblur_state.metrics(),
                "lm_optimizer": lm_status,
                "lmrs_phase_iterations": lm_phase.iterations if lm_phase else 0,
                "lmrs_phase_elapsed_seconds": round(lm_elapsed, 3),
                "lmrs_last_loss": lm_phase.last_loss if lm_phase else None,
                "compact_box_rasterizer": compact_box_status(),
                **policy.metrics(),
            },
        )


def build_optimization_options(iterations: int, options: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        iterations=iterations,
        position_lr_init=read_float(options.get("fine_position_lr_init"), 0.00016, minimum=1e-7, maximum=0.01),
        position_lr_final=read_float(options.get("fine_position_lr_final"), 0.0000016, minimum=1e-8, maximum=0.001),
        position_lr_delay_mult=0.01,
        position_lr_max_steps=iterations,
        feature_lr=0.0025,
        opacity_lr=0.025,
        scaling_lr=0.005,
        rotation_lr=0.001,
        exposure_lr_init=0.01,
        exposure_lr_final=0.001,
        exposure_lr_delay_steps=0,
        exposure_lr_delay_mult=0.0,
        percent_dense=0.01,
        lambda_dssim=read_float(options.get("fine_lambda_dssim"), 0.2, minimum=0.0, maximum=1.0),
        densification_interval=read_int(options.get("fine_densification_interval"), 100, minimum=20, maximum=1000),
        opacity_reset_interval=read_int(options.get("fine_opacity_reset_interval"), 3000, minimum=500, maximum=10000),
        densify_from_iter=read_int(options.get("fine_densify_from_iter"), 500, minimum=0, maximum=max(1, iterations)),
        densify_until_iter=read_int(options.get("fine_densify_until_iter"), min(15000, iterations), minimum=1, maximum=iterations),
        densify_grad_threshold=read_float(options.get("fine_densify_grad_threshold"), 0.0002, minimum=1e-6, maximum=0.01),
        lambda_scale=0.0,
        scale_cutoff=read_float(options.get("fine_scale_cutoff"), 154.3, minimum=1.0, maximum=10000.0),
        random_background=False,
        optimizer_type=str(options.get("fine_optimizer_type") or "default"),
    )
