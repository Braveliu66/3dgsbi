from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Any

import torch

from app.fine.deblur_mlp import attach_deblur_mlp_optimizer, build_deblur_mlp_state, render_with_deblur_mlp
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
    blur_registry: dict[str, dict[str, Any]] | None,
    options: dict[str, Any],
    progress: Progress,
) -> MobileGSTrainResult:
    if not torch.cuda.is_available():
        raise FineFailure("GPU_RESOURCE_UNAVAILABLE", "CUDA GPU is required for fine 3DGS training")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    progress("fine_mobilegs_setup", 42, f"[mobilegs] output directory ready: {output_dir}")
    with local_3dgs_runtime() as runtime:
        GaussianModel = runtime.GaussianModel
        Scene = runtime.Scene
        l1_loss = runtime.l1_loss
        ssim = runtime.ssim
        opt = build_optimization_options(iterations, options)
        progress("fine_mobilegs_runtime", 42, f"[mobilegs] local 3DGS runtime loaded from {runtime.root}")
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
        progress(
            "fine_mobilegs_dataset",
            42,
            (
                "[mobilegs] dataset configured: "
                f"source={dataset.source_path}, images={dataset.images}, resolution={dataset.resolution}, "
                f"data_device={dataset.data_device}, sh_degree={dataset.sh_degree}"
            ),
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
        progress("fine_mobilegs_scene_loading", 42, "[mobilegs] loading COLMAP-compatible scene and initial Gaussian cloud")
        scene = Scene(dataset, gaussians, shuffle=True)

        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        gaussians.save_ply(debug_dir / "init_gaussians.ply")
        progress(
            "fine_mobilegs_scene_ready",
            43,
            (
                "[mobilegs] scene loaded: "
                f"train_cameras={len(scene.getTrainCameras())}, initial_gaussians={gaussian_count(gaussians)}, "
                f"scene_extent={float(scene.cameras_extent):.6f}, debug_init={debug_dir / 'init_gaussians.ply'}"
            ),
        )

        progress(
            "fine_optimizer_setup",
            44,
            (
                "[mobilegs] optimizer options: "
                f"iterations={iterations}, lambda_dssim={opt.lambda_dssim}, "
                f"position_lr={opt.position_lr_init}->{opt.position_lr_final}, "
                f"densify_from={opt.densify_from_iter}, densify_until={opt.densify_until_iter}, "
                f"densification_interval={opt.densification_interval}, opacity_reset_interval={opt.opacity_reset_interval}"
            ),
        )
        gaussians.training_setup(opt)
        background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        cameras = scene.getTrainCameras().copy()
        if not cameras:
            raise FineFailure("FINE_CAMERA_LOAD_FAILED", "3DGS scene contains no train cameras")
        blur_registry = blur_registry or {}
        force_deblur = read_bool(options.get("fine_deblur_enabled"), False)
        if force_deblur:
            blurred_train_cameras = len(cameras)
            clear_train_cameras = 0
            deblur_options = options
            deblur_mode = blur_mode if blur_mode != "sharp" else str(options.get("fine_deblur_mode") or "motion")
        else:
            blurred_train_cameras = sum(1 for camera in cameras if is_blurred_view(camera, blur_registry))
            clear_train_cameras = max(0, len(cameras) - blurred_train_cameras)
            deblur_options = options if blurred_train_cameras > 0 else {**options, "fine_deblur_enabled": "false"}
            deblur_mode = blur_mode if blurred_train_cameras > 0 else "sharp"
        deblur_state = build_deblur_mlp_state(deblur_mode, deblur_options, device=background.device)
        attach_deblur_mlp_optimizer(gaussians, deblur_state)
        deblur_warmup_iters = resolve_deblur_warmup(iterations, options, deblur_state.enabled)
        deblur_xyz_lr_scale = read_float(options.get("fine_deblur_xyz_lr_scale"), 0.1, minimum=0.001, maximum=1.0)
        deblur_activated = False
        deblur_densify_disabled = False
        deblur_photometric_views = 0
        last_deblur_reg = 0.0
        if deblur_state.enabled:
            deblur_state.model.requires_grad_(False)
        progress(
            "fine_deblur_setup",
            45,
            (
                "[mobilegs] DeblurMLP setup: "
                f"enabled={deblur_state.enabled}, mode={deblur_mode}, warmup_iters={deblur_warmup_iters}, "
                f"xyz_lr_scale={deblur_xyz_lr_scale}, blurred_train_cameras={blurred_train_cameras}, "
                f"clear_train_cameras={clear_train_cameras}"
            ),
        )

        policy = FastGSPolicy(options)
        lm_status = resolve_local_lm_status(lm_start_iter, iterations, blur_mode, deblur_state.enabled, options)
        progress(
            "fine_lmrs_status",
            45,
            (
                "[mobilegs] LM-RS status: "
                f"active={lm_status['active']}, start_iter={lm_status['start_iter']}, reason={lm_status['reason']}"
            ),
        )
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
        log_interval = training_log_interval(iterations)
        progress(
            "fine_gaussian_train",
            46,
            (
                "[mobilegs] training loop started: "
                f"iterations={iterations}, cameras={len(cameras)}, gaussians={gaussian_count(gaussians)}, "
                f"progress_interval={log_interval}"
            ),
        )

        def topology_render(viewpoint, model, pipeline, bg, **render_options):
            return normalize_render_pkg(render_gaussians(viewpoint, model, pipeline, bg, **render_options), int(model.get_xyz.shape[0]))

        def should_deblur_view(viewpoint) -> bool:
            if force_deblur:
                return True
            return is_blurred_view(viewpoint, blur_registry)

        def photometric_render(viewpoint, model, pipeline, bg, *, use_deblur: bool = True, **render_options):
            if deblur_state.enabled and use_deblur and should_deblur_view(viewpoint):
                return normalize_render_pkg(render_with_deblur_mlp(viewpoint, model, pipeline, bg, deblur_state), int(model.get_xyz.shape[0]))
            return topology_render(viewpoint, model, pipeline, bg, **render_options)

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
                base_xyz_lr = optimizer_lr_value(gaussians, "xyz")
            if deblur_active and not deblur_activated:
                deblur_state.model.requires_grad_(True)
                deblur_activated = True
                deblur_densify_disabled = True
                progress(
                    "fine_deblur_active",
                    min(84, 42 + int(iteration / max(1, iterations) * 42)),
                    f"[mobilegs] DeblurMLP activated after warmup at iteration {iteration}; xyz_lr={optimizer_lr(gaussians, 'xyz')}",
                )
            if deblur_active and not in_lm_phase and base_xyz_lr is not None:
                set_xyz_learning_rate(gaussians, base_xyz_lr * deblur_xyz_lr_scale)
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
                progress(
                    "fine_sh_degree_update",
                    min(84, 42 + int(iteration / max(1, iterations) * 42)),
                    f"[mobilegs] increased spherical harmonics degree at iteration {iteration}",
                )
            if in_lm_phase:
                if lm_optimizer is None:
                    lm_options = build_lmrs_options(options)
                    gaussians.cgState = CGSolverState(int(gaussians.get_xyz.shape[0]), lm_options)
                    gaussians.cgState.set_scene_size(scene)
                    gaussians.batchState = BatchState.create(gaussians.cgState.batch_size, int(gaussians.get_xyz.shape[0]))
                    lm_optimizer = LocalCGOptimizer(gaussians, lm_options, scene_extent=scene.cameras_extent)
                    lm_pixel_sampler = UniformPixelSampler()
                    lm_camera_sampler = RandomCameraSampler(cameras)
                    progress(
                        "fine_lmrs_phase_start",
                        72,
                        (
                            "[mobilegs] switched to local LM-RS matrix-free CG: "
                            f"iteration={iteration}, cg_iter={lm_options.cg_iter}, batch_size={lm_options.batch_size}, "
                            f"samples_per_tile={lm_options.N_sample_per_tile}"
                        ),
                    )
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
                if iteration % log_interval == 0 or iteration == iterations:
                    mapped_progress = 42 + int(iteration / max(1, iterations) * 42)
                    progress(
                        "fine_gaussian_train",
                        min(84, mapped_progress),
                        (
                            "[mobilegs] LM-RS iteration checkpoint: "
                            f"iteration={iteration}/{iterations}, lm_loss={ema_loss:.5f}, "
                            f"last_lm_loss={lm_last_loss:.5f}, gaussians={gaussian_count(gaussians)}, lm_elapsed={lm_elapsed:.3f}s"
                        ),
                    )
                continue

            if not train_stack:
                train_stack = cameras.copy()
            viewpoint = train_stack.pop(random.randrange(len(train_stack)))
            gt_image = viewpoint.original_image.to("cuda")

            deblur_view_active = bool(deblur_active and should_deblur_view(viewpoint))
            render_pkg = photometric_render(viewpoint, gaussians, pipe, background, use_deblur=deblur_active)
            image = render_pkg["render"]
            l1_value = l1_loss(image, gt_image)
            ssim_value = 1.0 - ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * l1_value + opt.lambda_dssim * ssim_value
            if deblur_view_active:
                deblur_reg = render_pkg.get("deblur_regularization")
                if deblur_reg is not None and deblur_state.config is not None:
                    loss = loss + deblur_state.config.transform_reg_weight * deblur_reg
                    last_deblur_reg = float(deblur_reg.detach().item())
                deblur_photometric_views += 1
            gaussians.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            with torch.no_grad():
                can_densify = bool(iteration < opt.densify_until_iter and not deblur_active)
                if can_densify:
                    visible = render_pkg["visibility_filter"]
                    gaussians.max_radii2D[visible] = torch.max(gaussians.max_radii2D[visible], render_pkg["radii"][visible])
                    gaussians.add_densification_stats(render_pkg["viewspace_points"], visible)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            ema_loss = 0.4 * float(loss.item()) + 0.6 * ema_loss
            last_l1 = float(l1_value.item())
            last_ssim = float(ssim_value.item())

            with torch.no_grad():
                if can_densify:
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        policy.update_multiview_scores(
                            cameras=cameras,
                            gaussians=gaussians,
                            pipe=pipe,
                            background=background,
                            render_fn=topology_render,
                            metric_render_fn=topology_render,
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
                        progress(
                            "fine_fastgs_topology",
                            min(84, 42 + int(iteration / max(1, iterations) * 42)),
                            (
                                "[mobilegs] FastGS densify/prune step: "
                                f"iteration={iteration}, changed={changed}, gaussians={gaussian_count(gaussians)}, "
                                f"size_threshold={size_threshold}, ema_loss={ema_loss:.5f}"
                            ),
                        )
                        if changed:
                            policy.reset_after_topology_change()
                    if iteration % opt.opacity_reset_interval == 0:
                        gaussians.reset_opacity()
                        progress(
                            "fine_opacity_reset",
                            min(84, 42 + int(iteration / max(1, iterations) * 42)),
                            f"[mobilegs] opacity reset at iteration {iteration}, gaussians={gaussian_count(gaussians)}",
                        )
                elif policy.late_prune_enabled and iteration > 15_000 and iteration < iterations and iteration % 3000 == 0:
                    policy.update_multiview_scores(
                        cameras=cameras,
                        gaussians=gaussians,
                        pipe=pipe,
                        background=background,
                        render_fn=topology_render,
                        metric_render_fn=topology_render,
                        ssim_fn=ssim,
                    )
                    policy.apply_final_prune(gaussians, min_opacity=policy.final_prune_min_opacity)
                    policy.observe_gaussian_count(gaussians)
                    progress(
                        "fine_fastgs_final_prune",
                        min(84, 42 + int(iteration / max(1, iterations) * 42)),
                        f"[mobilegs] late FastGS prune at iteration {iteration}, gaussians={gaussian_count(gaussians)}",
                    )

            if iteration % log_interval == 0 or iteration == iterations:
                mapped_progress = 42 + int(iteration / max(1, iterations) * 42)
                progress(
                    "fine_gaussian_train",
                    min(84, mapped_progress),
                    (
                        "[mobilegs] training checkpoint: "
                        f"iteration={iteration}/{iterations}, loss={ema_loss:.5f}, l1={last_l1:.5f}, "
                        f"ssim_loss={last_ssim:.5f}, gaussians={gaussian_count(gaussians)}, "
                        f"deblur_active={deblur_active}, deblur_view={deblur_view_active}, xyz_lr={optimizer_lr(gaussians, 'xyz')}"
                    ),
                )

        progress("fine_fastgs_final_prune", 85, f"[mobilegs] running final multiview scoring and prune, gaussians_before={gaussian_count(gaussians)}")
        with torch.no_grad():
            policy.update_multiview_scores(
                cameras=cameras,
                gaussians=gaussians,
                pipe=pipe,
                background=background,
                render_fn=topology_render,
                metric_render_fn=topology_render,
                ssim_fn=ssim,
            )
            policy.apply_final_prune(gaussians, min_opacity=policy.final_prune_min_opacity)
            policy.observe_gaussian_count(gaussians)
        progress("fine_gaussian_save", 86, f"[mobilegs] saving Gaussian scene at iteration {iterations}, gaussians={gaussian_count(gaussians)}")

        scene.save(iterations)
        ply_path = output_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        if not ply_path.exists() or ply_path.stat().st_size <= 0:
            raise FineFailure("FINE_PLY_NOT_FOUND", f"3DGS training did not produce point cloud: {ply_path}")
        elapsed = round(time.monotonic() - started, 3)
        progress(
            "fine_gaussian_train_done",
            87,
            (
                "[mobilegs] training complete: "
                f"ply={ply_path}, elapsed={elapsed}s, final_gaussians={gaussian_count(gaussians)}, "
                f"ema_loss={ema_loss:.5f}, last_l1={last_l1:.5f}, last_ssim_loss={last_ssim:.5f}"
            ),
        )
        scene_backend = str(options.get("_fine_scene_backend") or "unknown")
        return MobileGSTrainResult(
            ply_path=ply_path,
            iterations=iterations,
            point_count=int(gaussians.get_xyz.shape[0]),
            metrics={
                "training_backend": f"{scene_backend}_initialized_local_3dgs_fastgs_deblur" if deblur_state.enabled else f"{scene_backend}_initialized_local_3dgs_fastgs",
                "local_3dgs_root": str(runtime.root),
                "raster_backend": "diff_gaussian_rasterization_fastgs",
                "target_raster_backend": "diff_gaussian_rasterization_fastgs",
                "data_device": dataset.data_device,
                "training_elapsed_seconds": elapsed,
                "final_gaussians": int(gaussians.get_xyz.shape[0]),
                "last_l1_loss": last_l1,
                "last_ssim_loss": last_ssim,
                "ema_loss": ema_loss,
                "deblur_mode": deblur_mode,
                "deblur_warmup_iters": deblur_warmup_iters,
                "deblur_xyz_lr_scale": deblur_xyz_lr_scale,
                "deblur_activated_after_warmup": deblur_activated,
                "deblur_blurred_train_cameras": blurred_train_cameras,
                "deblur_clear_train_cameras": clear_train_cameras,
                "deblur_photometric_views": deblur_photometric_views,
                "deblur_densify_disabled_after_activation": deblur_densify_disabled,
                "deblur_last_transform_regularization": last_deblur_reg,
                **deblur_state.metrics(),
                "lm_optimizer": lm_status,
                "lmrs_phase_iterations": lm_iterations,
                "lmrs_phase_elapsed_seconds": round(lm_elapsed, 3),
                "lmrs_last_loss": lm_last_loss,
                "lmrs_cg_iter": read_int(options.get("fine_lmrs_cg_iter"), 8, minimum=1, maximum=64) if lm_iterations else None,
                "fastergs_backend": "compact_box_cuda_if_available",
                "requested_algorithms": ["LiteVGGT", "Deblurring-3DGS", "FastGS"],
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


def training_log_interval(iterations: int) -> int:
    return max(50, min(500, iterations // 20 or 50))


def gaussian_count(gaussians: Any) -> int:
    return int(gaussians.get_xyz.shape[0])


def optimizer_lr(gaussians: Any, group_name: str) -> str:
    value = optimizer_lr_value(gaussians, group_name)
    return f"{float(value):.8f}" if value is not None else "unavailable"


def optimizer_lr_value(gaussians: Any, group_name: str) -> float | None:
    optimizer = getattr(gaussians, "optimizer", None)
    if optimizer is None:
        return None
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            value = group.get("lr")
            return float(value) if value is not None else None
    return None


def set_xyz_learning_rate(gaussians: Any, value: float) -> None:
    optimizer = getattr(gaussians, "optimizer", None)
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        if group.get("name") == "xyz":
            group["lr"] = float(value)
            return


def resolve_deblur_warmup(iterations: int, options: dict[str, Any], deblur_enabled: bool) -> int:
    if not deblur_enabled:
        return iterations
    explicit = options.get("fine_deblur_warmup_iters")
    fallback = min(3000, max(0, iterations // 3))
    warmup = read_int(explicit, fallback, minimum=0, maximum=max(0, iterations)) if explicit is not None else fallback
    return min(warmup, max(0, iterations - 1))


def is_blurred_view(camera: Any, blur_registry: dict[str, dict[str, Any]]) -> bool:
    entry = blur_registry_entry(camera, blur_registry)
    return bool(entry and entry.get("blurred"))


def blur_registry_entry(camera: Any, blur_registry: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    image_name = str(getattr(camera, "image_name", "") or "")
    candidates = [image_name]
    path = Path(image_name)
    if path.suffix:
        candidates.append(path.name)
        candidates.append(path.stem)
    else:
        candidates.extend([f"{image_name}.jpg", f"{image_name}.png"])
    for key in candidates:
        if key in blur_registry:
            return blur_registry[key]
    return None


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
        "diff_gaussian_rasterization_fastgs_active",
    ]
    if lm_iterations > 0:
        algorithms.append("LM-RS_local_matrix_free")
    if policy.cuda_metric_calls > 0:
        algorithms.append("FastGS_cuda_metric_counts")
    if policy.compact_box_available:
        algorithms.append("FasterGS_compact_box_cuda")
    return algorithms
