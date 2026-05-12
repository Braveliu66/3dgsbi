from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Any

import torch

from app.fine.deblur_mlp import attach_deblur_mlp_optimizer, build_deblur_mlp_state, render_with_deblur_mlp
from app.fine.edgs_init import EDGSDenseInit, make_edgs_cfg
from app.fine.fastgs_policy import FastGSPolicy
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
            data_device=str(options.get("fine_data_device") or "cpu"),
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
        edgs_metrics = initialize_edgs_if_enabled(gaussians, scene, opt, options)
        gaussians.training_setup(opt)
        background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        deblur_state = build_deblur_mlp_state(blur_mode, options, device=background.device)
        attach_deblur_mlp_optimizer(gaussians, deblur_state)
        deblur_warmup_iters = read_int(options.get("fine_deblur_warmup_iters"), 3000, minimum=0, maximum=iterations)
        deblur_xyz_lr_scale = read_float(options.get("fine_deblur_xyz_lr_scale"), 0.1, minimum=0.001, maximum=1.0)
        deblur_activated = False
        if deblur_state.enabled:
            deblur_state.model.requires_grad_(False)
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

        def standard_render(viewpoint, model, pipeline, bg, **render_options):
            return normalize_render_pkg(render_gaussians(viewpoint, model, pipeline, bg, **render_options), int(model.get_xyz.shape[0]))

        def training_render(viewpoint, model, pipeline, bg, *, use_deblur: bool = True, **render_options):
            if deblur_state.enabled and use_deblur:
                return normalize_render_pkg(render_with_deblur_mlp(viewpoint, model, pipeline, bg, deblur_state), int(model.get_xyz.shape[0]))
            return standard_render(viewpoint, model, pipeline, bg, **render_options)

        def lmrs_render(viewpoint, model, pipeline, bg, cg_state, batch_index):
            return normalize_render_pkg(
                render_gaussians(viewpoint, model, pipeline, bg, cg_state=cg_state, current_batch=batch_index, is_batched=True),
                int(model.get_xyz.shape[0]),
            )

        for iteration in range(1, iterations + 1):
            in_lm_phase = bool(lm_status["active"] and iteration >= lm_start_iter)
            deblur_active = bool(deblur_state.enabled and iteration > deblur_warmup_iters)
            if not in_lm_phase:
                gaussians.update_learning_rate(iteration)
            if deblur_active and not deblur_activated:
                deblur_state.model.requires_grad_(True)
                deblur_activated = True
            if deblur_active:
                scale_xyz_learning_rate(gaussians, deblur_xyz_lr_scale)
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
                    progress("fine_gaussian_train", min(84, mapped_progress), f"trained {iteration}/{iterations} iterations, lm_loss={ema_loss:.5f}")
                continue

            if not train_stack:
                train_stack = cameras.copy()
            viewpoint = train_stack.pop(random.randrange(len(train_stack)))
            gt_image = viewpoint.original_image.to("cuda")

            render_pkg = training_render(viewpoint, gaussians, pipe, background, use_deblur=deblur_active)
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
                            render_fn=training_render if deblur_active else standard_render,
                            ssim_fn=ssim,
                        )
                        changed = policy.apply_fastgs_densify_and_prune(
                            gaussians,
                            opt=opt,
                            scene_extent=scene.cameras_extent,
                            size_threshold=size_threshold,
                            radii=render_pkg["radii"],
                        )
                        policy.observe_gaussian_count(gaussians)
                        if changed:
                            policy.reset_after_topology_change()
                    if iteration % opt.opacity_reset_interval == 0:
                        gaussians.reset_opacity()
                elif iteration > 15_000 and iteration < iterations and iteration % 3000 == 0:
                    policy.update_multiview_scores(
                        cameras=cameras,
                        gaussians=gaussians,
                        pipe=pipe,
                        background=background,
                        render_fn=training_render if deblur_activated else standard_render,
                        ssim_fn=ssim,
                    )
                    policy.apply_final_prune(gaussians)
                    policy.observe_gaussian_count(gaussians)

            if iteration % max(50, min(500, iterations // 20 or 50)) == 0 or iteration == iterations:
                mapped_progress = 42 + int(iteration / max(1, iterations) * 42)
                progress("fine_gaussian_train", min(84, mapped_progress), f"trained {iteration}/{iterations} iterations, loss={ema_loss:.5f}")

        with torch.no_grad():
            policy.update_multiview_scores(
                cameras=cameras,
                gaussians=gaussians,
                pipe=pipe,
                background=background,
                render_fn=training_render if deblur_activated else standard_render,
                ssim_fn=ssim,
            )
            policy.apply_final_prune(gaussians)
            policy.observe_gaussian_count(gaussians)

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
                "training_backend": "litevggt_initialized_local_3dgs_fastgs_deblur" if deblur_state.enabled else "litevggt_initialized_local_3dgs_fastgs",
                "local_3dgs_root": str(runtime.root),
                "raster_backend": "diff_gaussian_rasterization",
                "target_raster_backend": "gsplat",
                "data_device": dataset.data_device,
                "training_elapsed_seconds": elapsed,
                "final_gaussians": int(gaussians.get_xyz.shape[0]),
                "last_l1_loss": last_l1,
                "last_ssim_loss": last_ssim,
                "ema_loss": ema_loss,
                "deblur_mode": blur_mode,
                "deblur_warmup_iters": deblur_warmup_iters,
                "deblur_xyz_lr_scale": deblur_xyz_lr_scale,
                "deblur_activated_after_warmup": deblur_activated,
                **deblur_state.metrics(),
                **edgs_metrics,
                "lm_optimizer": lm_status,
                "lmrs_phase_iterations": lm_iterations,
                "lmrs_phase_elapsed_seconds": round(lm_elapsed, 3),
                "lmrs_last_loss": lm_last_loss,
                "lmrs_cg_iter": read_int(options.get("fine_lmrs_cg_iter"), 8, minimum=1, maximum=64) if lm_iterations else None,
                "fastergs_backend": "compact_box_cuda_if_available",
                "requested_algorithms": ["LiteVGGT", "Deblurring-3DGS", "FastGS", "gsplat"],
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
        feature_lr=read_float(options.get("fine_lowfeature_lr"), 0.0025, minimum=1e-6, maximum=0.1),
        highfeature_lr=read_float(options.get("fine_highfeature_lr"), 0.005, minimum=1e-6, maximum=0.1),
        lowfeature_lr=read_float(options.get("fine_lowfeature_lr"), 0.0025, minimum=1e-6, maximum=0.1),
        opacity_lr=0.025,
        scaling_lr=0.005,
        rotation_lr=0.001,
        exposure_lr_init=0.01,
        exposure_lr_final=0.001,
        exposure_lr_delay_steps=0,
        exposure_lr_delay_mult=0.0,
        percent_dense=read_float(options.get("fine_percent_dense", options.get("fine_dense")), 0.001, minimum=1e-5, maximum=0.1),
        dense=read_float(options.get("fine_dense", options.get("fine_percent_dense")), 0.001, minimum=1e-5, maximum=0.1),
        lambda_dssim=read_float(options.get("fine_lambda_dssim"), 0.2, minimum=0.0, maximum=1.0),
        densification_interval=read_int(options.get("fine_densification_interval"), 100, minimum=20, maximum=1000),
        opacity_reset_interval=read_int(options.get("fine_opacity_reset_interval"), 3000, minimum=500, maximum=10000),
        densify_from_iter=read_int(options.get("fine_densify_from_iter"), 500, minimum=0, maximum=max(1, iterations)),
        densify_until_iter=read_int(options.get("fine_densify_until_iter"), min(15000, iterations), minimum=1, maximum=iterations),
        densify_grad_threshold=read_float(options.get("fine_densify_grad_threshold"), 0.0002, minimum=1e-6, maximum=0.01),
        grad_thresh=read_float(options.get("fine_fastgs_grad_thresh"), 0.0002, minimum=1e-6, maximum=0.01),
        grad_abs_thresh=read_float(options.get("fine_fastgs_grad_abs_thresh"), 0.0012, minimum=1e-6, maximum=0.1),
        lambda_scale=0.0,
        scale_cutoff=read_float(options.get("fine_scale_cutoff"), 154.3, minimum=1.0, maximum=10000.0),
        random_background=False,
        optimizer_type=str(options.get("fine_optimizer_type") or "default"),
    )


def initialize_edgs_if_enabled(gaussians: Any, scene: Any, opt: SimpleNamespace, options: dict[str, Any]) -> dict[str, Any]:
    if not read_bool(options.get("fine_edgs_enabled"), True):
        return {
            "edgs_enabled": False,
            "densification_disabled_by_edgs": False,
        }

    cfg = make_edgs_cfg(
        matches_per_ref=read_int(options.get("fine_edgs_matches_per_ref"), 15_000, minimum=1_000, maximum=50_000),
        nns_per_ref=read_int(options.get("fine_edgs_nns_per_ref"), 3, minimum=1, maximum=8),
        num_refs=read_optional_int(options.get("fine_edgs_num_refs")),
        scene=scene,
        roma_model=str(options.get("fine_edgs_roma_model") or "outdoor"),
        roma_coarse_res=read_roma_resolution_option(
            options.get("fine_edgs_roma_coarse_res"),
            560,
            minimum=224,
            maximum=1344,
            require_multiple_14=True,
        ),
        roma_upsample_res=read_roma_resolution_option(
            options.get("fine_edgs_roma_upsample_res"),
            864,
            minimum=224,
            maximum=2048,
            require_multiple_14=False,
        ),
        roma_sample_thresh=read_float(options.get("fine_edgs_roma_sample_thresh"), 0.05, minimum=0.0, maximum=1.0),
        roma_sample_mode=str(options.get("fine_edgs_roma_sample_mode") or "threshold_balanced"),
        roma_symmetric=read_bool(options.get("fine_edgs_roma_symmetric"), True),
        roma_use_custom_corr=read_bool(options.get("fine_edgs_roma_use_custom_corr"), True),
        roma_upsample_preds=read_bool(options.get("fine_edgs_roma_upsample_preds"), True),
        roma_with_padding=read_bool(options.get("fine_edgs_roma_with_padding"), False),
        max_points=read_int(options.get("fine_edgs_max_points"), 500_000, minimum=10_000, maximum=2_000_000),
        reprojection_error=read_float(options.get("fine_edgs_reprojection_error"), 4.0, minimum=0.5, maximum=32.0),
    )
    before = int(gaussians.get_xyz.shape[0])
    edgs = EDGSDenseInit(device="cuda", roma_model_name=cfg.roma_model)
    try:
        edgs.initialize(gaussians, scene, cfg)
    except FineFailure as exc:
        if read_bool(options.get("fine_edgs_required"), False):
            raise
        return {
            "edgs_enabled": False,
            "edgs_failed": True,
            "edgs_failure_code": exc.code,
            "edgs_failure_reason": exc.message,
            "edgs_sparse_gaussians_before": before,
            "densification_disabled_by_edgs": False,
        }
    opt.densify_until_iter = 0
    return {
        **edgs.last_metrics,
        "edgs_sparse_gaussians_before": before,
        "densification_disabled_by_edgs": True,
    }


def read_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "auto", "none"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_roma_resolution_option(
    value: Any,
    fallback: int | tuple[int, int],
    *,
    minimum: int,
    maximum: int,
    require_multiple_14: bool,
) -> int | tuple[int, int]:
    if value is None or str(value).strip().lower() in {"", "auto", "none"}:
        return fallback
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            width, height = int(value[0]), int(value[1])
            return (
                normalize_roma_resolution_dim(height, minimum, maximum, require_multiple_14),
                normalize_roma_resolution_dim(width, minimum, maximum, require_multiple_14),
            )
        except (TypeError, ValueError):
            return fallback

    normalized = str(value).strip().lower().replace("*", "x").replace(",", "x")
    parts = [part for part in normalized.split("x") if part]
    try:
        if len(parts) == 1:
            return normalize_roma_resolution_dim(int(parts[0]), minimum, maximum, require_multiple_14)
        if len(parts) == 2:
            width, height = int(parts[0]), int(parts[1])
            return (
                normalize_roma_resolution_dim(height, minimum, maximum, require_multiple_14),
                normalize_roma_resolution_dim(width, minimum, maximum, require_multiple_14),
            )
    except (TypeError, ValueError):
        return fallback
    return fallback


def normalize_roma_resolution_dim(value: int, minimum: int, maximum: int, require_multiple_14: bool) -> int:
    value = max(minimum, min(maximum, int(value)))
    if require_multiple_14:
        value = max(14, int(round(value / 14.0)) * 14)
    return value


def scale_xyz_learning_rate(gaussians: Any, multiplier: float) -> None:
    optimizer = getattr(gaussians, "optimizer", None)
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        if group.get("name") == "xyz":
            group["lr"] *= multiplier
            return


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
    return {
        "active": False,
        "start_iter": lm_start_iter,
        "reason": "LM-RS temporarily isolated due to unstable local backend",
    }


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
        "LiteVGGT_initialization",
        "Deblurring-3DGS_GTnet" if deblur_enabled else "Deblurring-3DGS_disabled",
        "FastGS_official_metric_map" if policy.official_metric_calls > 0 else "FastGS_local_multiview_score",
        "diff_gaussian_rasterization_active",
        "gsplat_target_backend",
    ]
    if lm_iterations > 0:
        algorithms.append("LM-RS_local_matrix_free")
    if policy.cuda_metric_calls > 0:
        algorithms.append("FastGS_cuda_metric_counts")
    if policy.compact_box_available:
        algorithms.append("FasterGS_compact_box_cuda")
    return algorithms
