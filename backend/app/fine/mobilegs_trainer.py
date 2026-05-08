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
from app.fine.local_3dgs.cg_optimizer import LocalCGOptimizer
from app.fine.local_3dgs.cg_state import BatchState, CGSolverState
from app.fine.local_3dgs.lmrs_step import RandomCameraSampler, UniformPixelSampler, lmrs_step
from app.fine.local_3dgs.render import render_gaussians
from app.fine.local_3dgs.runtime import local_3dgs_runtime, normalize_render_pkg
from app.fine.option_utils import read_float, read_int
from app.fine.types import FineFailure


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
    with local_3dgs_runtime() as runtime:
        render = runtime.render
        GaussianModel = runtime.GaussianModel
        Scene = runtime.Scene
        l1_loss = runtime.l1_loss
        ssim = runtime.ssim
        opt = build_optimization_options(iterations, options)
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
        gaussians = GaussianModel(3, "default")
        scene = Scene(dataset, gaussians, shuffle=True)
        gaussians.training_setup(opt)
        background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        deblur_state = build_deblur_mlp_state(blur_mode, options, device=background.device)
        attach_deblur_mlp_optimizer(gaussians, deblur_state)
        cameras = scene.getTrainCameras().copy()
        if not cameras:
            raise FineFailure("FINE_CAMERA_LOAD_FAILED", "3DGS scene contains no train cameras")

        policy = FastGSPolicy(options)
        lm_status = resolve_local_lm_status(lm_start_iter, iterations, blur_mode, deblur_state.enabled, options)
        lm_optimizer: LocalCGOptimizer | None = None
        lm_pixel_sampler: UniformPixelSampler | None = None
        lm_camera_sampler: RandomCameraSampler | None = None
        lm_iterations = 0
        lm_elapsed = 0.0
        lm_last_loss: float | None = None
        ema_loss = 0.0
        last_l1 = 0.0
        last_ssim = 0.0
        train_stack = cameras.copy()

        def standard_render(viewpoint, model, pipeline, bg):
            return normalize_render_pkg(render(viewpoint, model, pipeline, bg, scaling_modifier=1.0), int(model.get_xyz.shape[0]))

        def training_render(viewpoint, model, pipeline, bg):
            if deblur_state.enabled:
                return normalize_render_pkg(render_with_deblur_mlp(viewpoint, model, pipeline, bg, deblur_state), int(model.get_xyz.shape[0]))
            return standard_render(viewpoint, model, pipeline, bg)

        def lmrs_render(viewpoint, model, pipeline, bg, cg_state, batch_index):
            return normalize_render_pkg(
                render_gaussians(viewpoint, model, pipeline, bg, cg_state=cg_state, current_batch=batch_index, is_batched=True),
                int(model.get_xyz.shape[0]),
            )

        for iteration in range(1, iterations + 1):
            in_lm_phase = bool(lm_status["active"] and iteration >= lm_start_iter)
            if not in_lm_phase:
                gaussians.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
            if in_lm_phase:
                if lm_optimizer is None:
                    lm_options = build_lmrs_options(options)
                    gaussians.cgState = CGSolverState(int(gaussians.get_xyz.shape[0]), lm_options)
                    gaussians.cgState.set_scene_size(scene)
                    gaussians.batchState = BatchState.create(gaussians.cgState.batch_size, int(gaussians.get_xyz.shape[0]))
                    lm_optimizer = LocalCGOptimizer(gaussians, lm_options, scene_extent=scene.cameras_extent)
                    lm_pixel_sampler = UniformPixelSampler()
                    lm_camera_sampler = RandomCameraSampler(cameras)
                    progress("fine_lmrs_phase_start", 72, f"switched to local LM-RS matrix-free CG at iteration {iteration}")
                lm_loss, elapsed = lmrs_step(
                    pixel_sampler=lm_pixel_sampler,
                    camera_sampler=lm_camera_sampler,
                    gaussians=gaussians,
                    pipe=pipe,
                    background=background,
                    optimizer=lm_optimizer,
                    render_fn=lmrs_render,
                )
                lm_iterations += 1
                lm_elapsed += elapsed
                lm_last_loss = lm_loss
                ema_loss = 0.4 * lm_loss + 0.6 * ema_loss
                last_l1 = lm_loss
                last_ssim = 0.0
                if iteration % max(50, min(500, iterations // 20 or 50)) == 0 or iteration == iterations:
                    mapped_progress = 42 + int(iteration / max(1, iterations) * 42)
                    progress("fine_mobilegs_train", min(84, mapped_progress), f"trained {iteration}/{iterations} iterations, lm_loss={ema_loss:.5f}")
                continue

            if not train_stack:
                train_stack = cameras.copy()
            viewpoint = train_stack.pop(random.randrange(len(train_stack)))
            gt_image = viewpoint.original_image.to("cuda")

            render_pkg = training_render(viewpoint, gaussians, pipe, background)
            image = render_pkg["render"]
            l1_value = l1_loss(image, gt_image)
            ssim_value = 1.0 - ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * l1_value + opt.lambda_dssim * ssim_value
            gaussians.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            with torch.no_grad():
                if iteration < opt.densify_until_iter:
                    visible = render_pkg["visibility_filter"]
                    gaussians.max_radii2D[visible] = torch.max(gaussians.max_radii2D[visible], render_pkg["radii"][visible])
                    gaussians.add_densification_stats(render_pkg["viewspace_points"], visible)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            ema_loss = 0.4 * float(loss.item()) + 0.6 * ema_loss
            last_l1 = float(l1_value.item())
            last_ssim = float(ssim_value.item())

            with torch.no_grad():
                if iteration < opt.densify_until_iter:
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        policy.update_multiview_scores(
                            cameras=cameras,
                            gaussians=gaussians,
                            pipe=pipe,
                            background=background,
                            render_fn=training_render if deblur_state.enabled else standard_render,
                            ssim_fn=ssim,
                        )
                        original_accum = policy.apply_densification_gate(gaussians)
                        min_opacity = vcp_min_opacity(policy)
                        before = int(gaussians.get_xyz.shape[0])
                        gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity, scene.cameras_extent, size_threshold, render_pkg["radii"])
                        policy.observe_gaussian_count(gaussians)
                        if original_accum is not None or int(gaussians.get_xyz.shape[0]) != before:
                            policy.reset_after_topology_change()
                    if iteration % opt.opacity_reset_interval == 0:
                        gaussians.reset_opacity()
                elif iteration > 15_000 and iteration < iterations and iteration % 3000 == 0:
                    policy.update_multiview_scores(
                        cameras=cameras,
                        gaussians=gaussians,
                        pipe=pipe,
                        background=background,
                        render_fn=training_render if deblur_state.enabled else standard_render,
                        ssim_fn=ssim,
                    )
                    policy.apply_final_prune(gaussians)
                    policy.observe_gaussian_count(gaussians)

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
                "training_backend": "local_3dgs_adam_fastgs_deblur" if deblur_state.enabled else "local_3dgs_adam_fastgs",
                "local_3dgs_root": str(runtime.root),
                "training_elapsed_seconds": elapsed,
                "final_gaussians": int(gaussians.get_xyz.shape[0]),
                "last_l1_loss": last_l1,
                "last_ssim_loss": last_ssim,
                "ema_loss": ema_loss,
                "deblur_mode": blur_mode,
                **deblur_state.metrics(),
                "lm_optimizer": lm_status,
                "lmrs_phase_iterations": lm_iterations,
                "lmrs_phase_elapsed_seconds": round(lm_elapsed, 3),
                "lmrs_last_loss": lm_last_loss,
                "lmrs_cg_iter": read_int(options.get("fine_lmrs_cg_iter"), 8, minimum=1, maximum=64) if lm_iterations else None,
                "fastergs_backend": "local_algorithm_equivalent_no_cuda_kernel_fusion",
                "requested_algorithms": ["AMB3R", "Deblurring-3DGS", "FastGS", "FasterGS", "LM-RS"],
                "effective_algorithms": effective_algorithms(deblur_state.enabled, lm_iterations, policy),
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


def resolve_local_lm_status(
    lm_start_iter: int,
    iterations: int,
    blur_mode: str,
    deblur_enabled: bool,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not read_bool((options or {}).get("fine_lmrs_enabled"), False):
        return {
            "active": False,
            "start_iter": lm_start_iter,
            "reason": "LM-RS temporarily isolated due to unstable local backend",
        }
    if lm_start_iter >= iterations:
        return {"active": False, "start_iter": lm_start_iter, "reason": "LM phase disabled because start_iter >= iterations"}
    if deblur_enabled or blur_mode in {"motion", "defocus", "mixed"}:
        return {
            "active": False,
            "start_iter": lm_start_iter,
            "reason": "LM-RS disabled for blur/deblur scenes until Deblur-aware residual and Jacobian are implemented locally",
        }
    return {"active": True, "backend": "lmrs_local_matrix_free", "start_iter": lm_start_iter, "reason": None}


def read_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return fallback


def build_lmrs_options(options: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        cg_iter=read_int(options.get("fine_lmrs_cg_iter"), 8, minimum=1, maximum=64),
        regularizer=read_float(options.get("fine_lmrs_regularizer"), 0.01, minimum=1e-8, maximum=10.0),
        batch_size=read_int(options.get("fine_lmrs_batch_size"), 1, minimum=1, maximum=16),
        kernel=1,
        ssim_weight=read_float(options.get("fine_lmrs_lambda_dssim"), 0.0, minimum=0.0, maximum=1.0),
        fixed_lr=read_float(options.get("fine_lmrs_fixed_lr"), 0.1, minimum=1e-6, maximum=10.0),
        max_lr=read_float(options.get("fine_lmrs_max_lr"), 0.2, minimum=1e-6, maximum=10.0),
        auto_lr=False,
        loss_fn="mse",
        sampling_distribution="mobile_uniform",
        N_sample_per_tile=read_int(options.get("fine_lmrs_samples_per_tile"), 32, minimum=1, maximum=256),
        tile_block_dimx=16,
        tile_block_dimy=16,
        temperature=1.0,
        levenberg_type=str(options.get("fine_lmrs_levenberg_type") or "identity"),
    )


def effective_algorithms(deblur_enabled: bool, lm_iterations: int, policy: FastGSPolicy) -> list[str]:
    algorithms = [
        "AMB3R",
        "Deblurring-3DGS_GTnet" if deblur_enabled else "Deblurring-3DGS_disabled",
        "FastGS_local_multiview_score",
    ]
    if lm_iterations > 0:
        algorithms.append("LM-RS_local_matrix_free")
    if policy.cuda_metric_calls > 0:
        algorithms.append("FastGS_cuda_metric_counts")
    if policy.compact_box_available:
        algorithms.append("FasterGS_compact_box_cuda")
    return algorithms
