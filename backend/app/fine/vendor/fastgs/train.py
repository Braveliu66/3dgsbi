#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import json
import os, random, time
from dataclasses import replace
from random import randint
from pathlib import Path
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fastgs, network_gui_ws
from gaussian_renderer.deblur import render_fastgs_deblur
import sys
_BACKEND_ROOT = Path(__file__).resolve().parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.fine.fastgs_defaults import (
    COLMAP_MIN_SPARSE_POINTS,
    FASTGS_DEBLUR_EXTRA_POINTS_TARGET,
    FASTGS_DEBLUR_EXTRA_POINTS_WEAK_TARGET,
    FASTGS_FINAL_PRUNE_ENABLED,
    FASTGS_ITERATIONS,
    FASTGS_LATE_PRUNE_ENABLED,
    FASTGS_LATE_PRUNE_FROM_ITER,
    FASTGS_LATE_PRUNE_INTERVAL,
    FASTGS_LATE_PRUNE_UNTIL_ITER,
    FASTGS_VCP_BLUR_PROTECT_WEIGHT,
)
from app.fine.deblur_schedule import (
    DeblurFastGSSchedule,
    PROFILES,
    auto_detect_scene_profile,
)
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.blur_kernel import compute_blur_indicator
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    deblur_state = gaussians.create_deblur_net(opt)
    blur_registry = load_blur_registry(getattr(opt, "deblur_blur_registry", ""))
    registry_match_metrics = summarize_blur_registry_matches(scene.getTrainCameras(), blur_registry)
    print(
        "[deblur] registry matched "
        f"{registry_match_metrics['matched_cameras']}/{registry_match_metrics['camera_count']} "
        f"clear_runtime_cameras={registry_match_metrics['clear_runtime_cameras']} "
        f"blur_runtime_cameras={registry_match_metrics['blur_runtime_cameras']}"
    )
    if deblur_state.enabled:
        opt.deblur_warmup_iters = min(max(0, int(opt.deblur_warmup_iters)), max(0, int(opt.iterations) - 1))
    deblur_schedule = build_deblur_training_schedule(opt, deblur_state)
    print(
        "[DeblurSchedule] "
        f"enabled={deblur_schedule['enabled']} "
        f"deblur_loss=({deblur_schedule['deblur_loss_from']},{deblur_schedule['deblur_loss_until']}) "
        f"densifyA=({deblur_schedule['densify_a_from']},{deblur_schedule['densify_a_until']}) "
        f"densifyB=({deblur_schedule['densify_b_from']},{deblur_schedule['densify_b_until']}) "
        f"opacity_reset_until={deblur_schedule['opacity_reset_until']} "
        f"late_prune=({deblur_schedule['late_prune_from']},{deblur_schedule['late_prune_until']})"
    )
    camera_centers = [
        cam.camera_center.detach()
        for cam in scene.getTrainCameras()
        if hasattr(cam, "camera_center")
    ]
    cameras_xyz = torch.stack(camera_centers) if camera_centers else None
    scene_profile_name = auto_detect_scene_profile(
        sfm_sparse_points=int(gaussians.get_xyz.shape[0]),
        sfm_registered_images=len(scene.getTrainCameras()),
        cameras_xyz=cameras_xyz,
        scene_extent=float(scene.cameras_extent),
        frontend_hint=getattr(opt, "scene_type", "auto"),
    )
    scene_profile = PROFILES.get(scene_profile_name, PROFILES["indoor_full"])
    scene_profile = replace(
        scene_profile,
        max_prune_fraction_per_step=float(
            getattr(opt, "fastgs_late_prune_max_fraction", scene_profile.max_prune_fraction_per_step)
        ),
    )
    runtime_schedule = DeblurFastGSSchedule(
        profile=scene_profile,
        total_iterations=int(opt.iterations),
        warmup_end=int(opt.deblur_warmup_iters),
        deblur_phase_end=int(getattr(opt, "fastgs_late_prune_from_iter", int(opt.iterations * 0.6))),
        consolidate_end=int(getattr(opt, "deblur_sharp_refine_from_iter", int(opt.iterations * 0.8))),
        sharp_refine_start=int(getattr(opt, "deblur_sharp_refine_from_iter", int(opt.iterations * 0.8))),
        num_blurred_frames=sum(1 for item in unique_blur_registry_items(blur_registry) if item.get("blurred") and not item.get("rejected")),
        num_total_frames=len(scene.getTrainCameras()),
    )
    print(f"[PROFILE] Using scene profile: {scene_profile.name} ({scene_profile_name})")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # record time
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    ema_loss_for_log = 0.0
    deblur_photometric_views = 0
    deblur_clear_train_cameras = 0
    deblur_final_pruned = False
    deblur_extra_points_added = 0
    deblur_extra_densify_steps = 0
    deblur_mandatory_extra_points_added = 0
    deblur_mandatory_extra_points_done = False
    sharp_score_sample_count = 0
    sharp_score_skipped_steps = 0
    topology_blur_stats_skipped = 0
    late_prune_steps = 0
    late_prune_removed = 0
    late_prune_deferred = 0
    late_prune_scale_candidates = 0
    final_prune_metrics = {}
    last_deblur_reg = None
    log_interval = 200
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", miniters=log_interval)
    progress_bar_last_iter = first_iter
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    for iteration in range(first_iter, opt.iterations + 1):

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render_fastgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()
        
        base_xyz_lr = gaussians.update_learning_rate(iteration)
        sharp_refine_enabled = option_bool(
            getattr(opt, "deblur_sharp_refine_enabled", "true"),
            True,
        )
        sharp_refine_from_iter = int(
            getattr(opt, "deblur_sharp_refine_from_iter", int(opt.iterations * 0.8))
        )
        sharp_refine_active = bool(sharp_refine_enabled and iteration >= sharp_refine_from_iter)

        # Keep GTnet as the training-time blur renderer through final refine.
        # Export and quality checks still use the normal sharp FastGS renderer.
        deblur_loss_active = schedule_deblur_loss_active(iteration, deblur_schedule, deblur_state)
        if iteration == sharp_refine_from_iter and deblur_state.enabled:
            gaussians.freeze_deblur_mlp()
            print(f"[ITER {iteration}] Switching to frozen GTnet sharp refine mode")
        if sharp_refine_active and base_xyz_lr is not None:
            set_xyz_learning_rate(gaussians, base_xyz_lr * 0.2)
        elif deblur_loss_active and base_xyz_lr is not None:
            set_xyz_learning_rate(gaussians, base_xyz_lr * float(opt.deblur_xyz_lr_scale))

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)
        if deblur_state.enabled and not deblur_loss_active:
            train_cameras = scene.getTrainCameras().copy()
            if train_cameras:
                viewpoint_cam = train_cameras[randint(0, len(train_cameras) - 1)]
        if sharp_refine_active and option_bool(
            getattr(opt, "deblur_sharp_refine_clear_only", "true"),
            True,
        ):
            train_cameras = scene.getTrainCameras().copy()
            if train_cameras:
                viewpoint_cam = train_cameras[randint(0, len(train_cameras) - 1)]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        deblur_view_active = bool(deblur_loss_active and is_deblur_view(viewpoint_cam, blur_registry, opt))
        if deblur_view_active:
            render_pkg = render_fastgs_deblur(viewpoint_cam, gaussians, pipe, bg, opt.mult, deblur_state)
        else:
            render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
            if deblur_loss_active:
                deblur_clear_train_cameras += 1
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        if deblur_view_active and not sharp_refine_active:
            deblur_reg = render_pkg.get("deblur_regularization")
            if deblur_reg is not None and deblur_state.config is not None:
                loss = loss + deblur_state.config.transform_reg_weight * deblur_reg
                last_deblur_reg = float(deblur_reg.detach().item())
        if deblur_view_active:
            deblur_photometric_views += 1
        loss.backward()
        schedule_ctrl = runtime_schedule.step(iteration, float(loss.detach().item()))

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % log_interval == 0 or iteration == opt.iterations:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "DeblurLoss": str(deblur_loss_active)})
                progress_bar.update(iteration - progress_bar_last_iter)
                progress_bar_last_iter = iteration
                print(
                    f"[ITER {iteration}] loss={ema_loss_for_log:.7f} "
                    f"deblur_loss_active={deblur_loss_active} deblur_view={deblur_view_active}"
                )
            if iteration == opt.iterations:
                progress_bar.close()

            iter_time = iter_start.elapsed_time(iter_end)
            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_fastgs, (pipe, background, opt.mult))
            if (iteration in saving_iterations):
                final_prune_enabled = option_bool(
                    getattr(opt, "fastgs_final_prune_enabled", None),
                    FASTGS_FINAL_PRUNE_ENABLED,
                )
                if iteration == opt.iterations and final_prune_enabled and not deblur_final_pruned:
                    camlist = sample_sharp_score_cameras(scene, blur_registry, opt)
                    pruning_score = None
                    if camlist:
                        sharp_score_sample_count += len(camlist)
                        _, pruning_score = compute_gaussian_score_fastgs(
                            camlist,
                            gaussians,
                            pipe,
                            bg,
                            opt,
                            score_renderer="sharp",
                            score_purpose="vcp",
                        )
                        blur_indicator = compute_blur_indicator(
                            deblur_state,
                            gaussians.get_xyz,
                            gaussians.get_scaling,
                            gaussians.get_rotation,
                            [camera.camera_center for camera in camlist],
                        )
                    else:
                        sharp_score_skipped_steps += 1
                        blur_indicator = None
                    final_max_world_scale = scale_limit_from_extent(
                        scene.cameras_extent,
                        getattr(opt, "fastgs_final_prune_max_world_scale_ratio", 0.0),
                    )
                    final_prune_metrics = gaussians.final_prune_fastgs(
                        min_opacity = opt.fastgs_final_prune_min_opacity,
                        pruning_score = pruning_score,
                        score_thresh = opt.fastgs_final_prune_score_thresh,
                        max_world_scale = final_max_world_scale,
                        blur_indicator = blur_indicator,
                        blur_protect_weight = getattr(opt, "fastgs_vcp_blur_protect_weight", FASTGS_VCP_BLUR_PROTECT_WEIGHT),
                    )
                    deblur_final_pruned = True
                    print(f"\n[ITER {iteration}] FastGS final prune complete {final_prune_metrics}")
                elif iteration == opt.iterations and not final_prune_enabled:
                    final_prune_metrics = {"enabled": False, "removed": 0}
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            optim_start.record()
            
            # Densification
            can_update_topology = schedule_densify_active(iteration, deblur_schedule)
            if can_update_topology:
                topology_sharp_only = option_bool(
                    getattr(opt, "deblur_topology_sharp_only", "false"),
                    False,
                )
                topology_stats_active = not (topology_sharp_only and deblur_view_active)
                if topology_stats_active:
                    # VCD uses the active score renderer.  Legacy sharp-only mode
                    # can still skip blurred-view topology stats via the option.
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                else:
                    topology_blur_stats_skipped += 1

                deblur_extra_densified = False
                enable_deblur_extra_points = option_bool(
                    getattr(opt, "deblur_extra_points_enabled", "false"),
                    False,
                )
                enable_mandatory_extra_points = option_bool(
                    getattr(opt, "deblur_extra_points_mandatory", "false"),
                    False,
                )
                if (
                    enable_deblur_extra_points
                    and enable_mandatory_extra_points
                    and not deblur_mandatory_extra_points_done
                    and deblur_state.enabled
                    and not sharp_refine_active
                    and iteration > opt.deblur_warmup_iters
                    and iteration % opt.densification_interval == 0
                ):
                    weak_init = int(gaussians.get_xyz.shape[0]) < COLMAP_MIN_SPARSE_POINTS
                    target_count = int(
                        getattr(
                            opt,
                            "deblur_extra_points_weak_target" if weak_init else "deblur_extra_points_target",
                            FASTGS_DEBLUR_EXTRA_POINTS_WEAK_TARGET if weak_init else FASTGS_DEBLUR_EXTRA_POINTS_TARGET,
                        )
                    )
                    added = gaussians.densify_deblur_seed_points(
                        target_count=target_count,
                        extent=scene.cameras_extent,
                        args=opt,
                    )
                    deblur_mandatory_extra_points_done = True
                    if added:
                        deblur_extra_points_added += int(added)
                        deblur_mandatory_extra_points_added += int(added)
                        deblur_extra_densify_steps += 1
                        deblur_extra_densified = True
                        print(
                            f"\n[ITER {iteration}] Deblur mandatory extra points "
                            f"target={target_count} added={added}"
                        )
                if (
                    enable_deblur_extra_points
                    and deblur_view_active
                    and not topology_sharp_only
                    and not sharp_refine_active
                    and iteration > opt.deblur_warmup_iters
                    and iteration < min(int(opt.densify_until_iter), 15_000)
                    and iteration % opt.densification_interval == 0
                ):
                    added = gaussians.densify_deblur_extra_points(
                        extent=scene.cameras_extent,
                        radii=radii,
                        args=opt,
                    )
                    if added:
                        deblur_extra_points_added += int(added)
                        deblur_extra_densify_steps += 1
                        deblur_extra_densified = True

                if (
                    not sharp_refine_active
                    and topology_stats_active
                    and not deblur_extra_densified
                    and iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_prune_from_iter = int(getattr(opt, "fastgs_size_prune_from_iter", 3000))
                    size_prune_max_screen_size = int(getattr(opt, "fastgs_size_prune_max_screen_size", 12))
                    size_threshold = size_prune_max_screen_size if iteration > size_prune_from_iter else None
                    camlist = sample_sharp_score_cameras(scene, blur_registry, opt)

                    # VCD grows topology from the same blurred observation model
                    # used by the active training loss, while VCP stays sharp.
                    if camlist:
                        sharp_score_sample_count += len(camlist)
                        vcd_score_renderer = (
                            "deblur"
                            if deblur_loss_active and deblur_state.enabled and not topology_sharp_only
                            else "sharp"
                        )
                        importance_score, pruning_score = compute_gaussian_score_fastgs(
                            camlist,
                            gaussians,
                            pipe,
                            bg,
                            opt,
                            DENSIFY=True,
                            score_renderer=vcd_score_renderer,
                            score_purpose="vcd",
                            deblur_state=deblur_state if vcd_score_renderer == "deblur" else None,
                        )
                        if vcd_score_renderer == "deblur":
                            _, pruning_score = compute_gaussian_score_fastgs(
                                camlist,
                                gaussians,
                                pipe,
                                bg,
                                opt,
                                score_renderer="sharp",
                                score_purpose="vcp",
                            )
                        gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                                                    min_opacity = 0.005, 
                                                    extent = scene.cameras_extent, 
                                                    radii=radii,
                                                    args = opt,
                                                    importance_score = importance_score,
                                                    pruning_score = pruning_score)
                    else:
                        sharp_score_skipped_steps += 1

                if (
                    schedule_opacity_reset_active(iteration, deblur_schedule)
                    and (
                        iteration % opt.opacity_reset_interval == 0
                        or (dataset.white_background and iteration == opt.densify_from_iter)
                    )
                ):
                    gaussians.reset_opacity()

            # The multiview consistent pruning of fastgs.
            # In this stage, the model converge basically. So we can prune more aggressively without degrading rendering quality.
            # You can check the rendering results of 20K iterations in arxiv version (https://arxiv.org/abs/2511.04283), the rendering quality is already very good.
            late_prune_enabled = option_bool(getattr(opt, "fastgs_late_prune_enabled", None), FASTGS_LATE_PRUNE_ENABLED)
            late_prune_interval = max(1, int(getattr(opt, "fastgs_late_prune_interval", FASTGS_LATE_PRUNE_INTERVAL)))
            late_prune_from_iter = int(getattr(opt, "fastgs_late_prune_from_iter", FASTGS_LATE_PRUNE_FROM_ITER))
            late_prune_until_iter = int(getattr(opt, "fastgs_late_prune_until_iter", FASTGS_LATE_PRUNE_UNTIL_ITER))
            if (
                late_prune_enabled
                and iteration >= sharp_refine_from_iter
                and schedule_late_prune_active(iteration, deblur_schedule)
                and (iteration - late_prune_from_iter) % late_prune_interval == 0
                and iteration >= late_prune_from_iter
                and iteration < late_prune_until_iter
            ):
                camlist = sample_sharp_score_cameras(scene, blur_registry, opt)

                if camlist:
                    sharp_score_sample_count += len(camlist)
                    _, pruning_score = compute_gaussian_score_fastgs(
                        camlist,
                        gaussians,
                        pipe,
                        bg,
                        opt,
                        score_renderer="sharp",
                        score_purpose="vcp",
                    )
                    blur_indicator = compute_blur_indicator(
                        deblur_state,
                        gaussians.get_xyz,
                        gaussians.get_scaling,
                        gaussians.get_rotation,
                        [camera.camera_center for camera in camlist],
                    )
                    late_max_world_scale = scale_limit_from_extent(
                        scene.cameras_extent,
                        getattr(opt, "fastgs_late_prune_max_world_scale_ratio", 0.0),
                    )
                    raw_prune_mask, late_metrics = gaussians.final_prune_mask_fastgs(
                        min_opacity = opt.fastgs_late_prune_min_opacity,
                        pruning_score = pruning_score,
                        score_thresh = opt.fastgs_late_prune_score_thresh,
                        max_world_scale = late_max_world_scale,
                        use_score = scene_profile.name != "indoor",
                        use_scale = scene_profile.name != "indoor",
                        blur_indicator = blur_indicator,
                        blur_protect_weight = getattr(opt, "fastgs_vcp_blur_protect_weight", FASTGS_VCP_BLUR_PROTECT_WEIGHT),
                    )
                    before_prune_count = int(gaussians.get_xyz.shape[0])
                    safe_prune_mask = runtime_schedule.safe_prune_mask(
                        iteration,
                        gaussians,
                        raw_prune_mask,
                        schedule_ctrl.get("prune_mode", "adaptive"),
                    )
                    raw_removed = int(late_metrics.get("removed", 0))
                    safe_removed = int(torch.count_nonzero(safe_prune_mask).detach().item())
                    if safe_removed:
                        gaussians.prune_points(safe_prune_mask)
                    gaussians.tmp_radii = None
                    torch.cuda.empty_cache()
                    after_prune_count = int(gaussians.get_xyz.shape[0])
                    remove_fraction = safe_removed / max(before_prune_count, 1)
                    late_metrics["raw_removed"] = raw_removed
                    late_metrics["removed"] = safe_removed
                    late_metrics["deferred"] = max(0, raw_removed - safe_removed)
                    late_metrics["before_count"] = before_prune_count
                    late_metrics["after_count"] = after_prune_count
                    late_metrics["remove_fraction"] = remove_fraction
                    late_prune_steps += 1
                    late_prune_removed += int(late_metrics.get("removed", 0))
                    late_prune_deferred += int(late_metrics.get("deferred", 0))
                    late_prune_scale_candidates += int(late_metrics.get("scale_candidates", 0))
                    print(f"\n[ITER {iteration}] FastGS late prune {late_metrics}")
                else:
                    sharp_score_skipped_steps += 1
        
            # Optimization step
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # record time
            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

    # scene.save(iteration)
    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Training time: {total_time}")
    final_metrics = collect_final_metrics(
        dataset.model_path,
        opt.iterations,
        scene,
        gaussians,
        pipe,
        background,
        opt.mult,
        total_time,
    )
    if deblur_state.enabled and deblur_photometric_views == 0:
        raise RuntimeError(
            "Deblur was enabled, but no training view used deblur render. "
            "Check deblur warmup, sharp refine, and total iteration settings."
        )
    write_deblur_metrics(
        dataset.model_path,
        gaussians,
        deblur_state,
        blur_registry,
        registry_match_metrics,
        deblur_photometric_views,
        deblur_clear_train_cameras,
        deblur_final_pruned,
        deblur_extra_points_added,
        deblur_mandatory_extra_points_added,
        deblur_extra_densify_steps,
        sharp_score_sample_count,
        sharp_score_skipped_steps,
        topology_blur_stats_skipped,
        late_prune_steps,
        late_prune_removed,
        late_prune_deferred,
        late_prune_scale_candidates,
        final_prune_metrics,
        last_deblur_reg,
        opt,
        deblur_schedule,
        final_metrics,
    )


def load_blur_registry(path):
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        print(f"[deblur] blur registry not found: {path}")
        return {}
    except Exception as exc:
        print(f"[deblur] failed to read blur registry {path}: {exc}")
        return {}
    if isinstance(payload, dict) and "frames" in payload and isinstance(payload["frames"], dict):
        return normalize_blur_registry(payload["frames"])
    if isinstance(payload, dict):
        return normalize_blur_registry(payload)
    return {}


def norm_image_key(name):
    if name is None:
        return ""
    normalized = str(name).replace("\\", "/").split("/")[-1]
    return os.path.splitext(normalized)[0].lower()


def normalize_blur_registry(registry):
    normalized = {}
    for key, value in registry.items():
        if not isinstance(value, dict):
            continue
        aliases = {
            norm_image_key(key),
            norm_image_key(value.get("training_image")),
            norm_image_key(value.get("training_stem")),
            norm_image_key(value.get("source_image")),
        }
        for alias in aliases:
            if alias:
                normalized[alias] = value
    return normalized


def blur_registry_entry(viewpoint_camera, blur_registry):
    key = norm_image_key(getattr(viewpoint_camera, "image_name", ""))
    return blur_registry.get(key)


def is_deblur_view(viewpoint_camera, blur_registry, opt):
    if not option_bool(getattr(opt, "deblur_blurred_views_only", "false"), False):
        return True
    item = blur_registry_entry(viewpoint_camera, blur_registry)
    if not isinstance(item, dict):
        return False
    return bool(item.get("blurred") and not item.get("rejected"))


def summarize_blur_registry_matches(cameras, blur_registry):
    camera_keys = {norm_image_key(getattr(camera, "image_name", "")) for camera in cameras}
    camera_keys.discard("")
    matched = camera_keys & set(blur_registry.keys())
    clear = 0
    blur = 0
    for key in matched:
        item = blur_registry.get(key)
        if not isinstance(item, dict) or item.get("rejected"):
            continue
        if item.get("blurred"):
            blur += 1
        else:
            clear += 1
    return {
        "camera_count": len(camera_keys),
        "registry_keys": len(blur_registry),
        "matched_cameras": len(matched),
        "clear_runtime_cameras": clear,
        "blur_runtime_cameras": blur,
    }


def unique_blur_registry_items(blur_registry):
    seen = set()
    items = []
    for key, item in blur_registry.items():
        if not isinstance(item, dict):
            continue
        identity = norm_image_key(item.get("training_image")) or norm_image_key(item.get("training_stem")) or key
        if identity in seen:
            continue
        seen.add(identity)
        items.append(item)
    return items


def sample_sharp_score_cameras(scene, _blur_registry, opt):
    cameras = scene.getTrainCameras().copy()
    if not cameras:
        return []
    return sampling_cameras(cameras, opt.fastgs_sample_cameras)


def set_xyz_learning_rate(gaussians, lr):
    for param_group in gaussians.optimizer.param_groups:
        if param_group.get("name") == "xyz":
            param_group["lr"] = lr
            return lr
    return None


def option_bool(value, default=False):
    if value in {None, ""}:
        return bool(default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def scale_limit_from_extent(extent, ratio):
    ratio = float(ratio or 0.0)
    if ratio <= 0:
        return None
    return float(extent) * ratio


def clamp_int(value, minimum, maximum):
    minimum = int(minimum)
    maximum = int(maximum)
    if maximum < minimum:
        return maximum
    return max(minimum, min(int(value), maximum))


def build_deblur_training_schedule(opt, deblur_state):
    total = max(1, int(opt.iterations))
    enabled = bool(deblur_state.enabled)
    auto = option_bool(getattr(opt, "deblur_auto_schedule", "true"), True)
    warmup = int(getattr(opt, "deblur_warmup_iters", 7000))

    if not enabled or not auto:
        return {
            "enabled": False,
            "profile": "manual",
            "deblur_loss_from": warmup,
            "deblur_loss_until": total,
            "densify_a_from": 0,
            "densify_a_until": min(int(getattr(opt, "densify_until_iter", total)), total),
            "densify_b_from": -1,
            "densify_b_until": -1,
            "opacity_reset_until": total + 1,
            "late_prune_from": int(getattr(opt, "fastgs_late_prune_from_iter", total + 1)),
            "late_prune_until": int(getattr(opt, "fastgs_late_prune_until_iter", total + 1)),
        }

    if enabled and auto:
        warmup = clamp_int(warmup, 1000, max(1, total - 1))

    deblur_loss_from = warmup
    deblur_loss_until = total
    densify_a_from = int(getattr(opt, "densify_from_iter", 500))
    densify_a_until = min(
        int(getattr(opt, "densify_until_iter", total)),
        total,
    )
    enable_late_densify = option_bool(getattr(opt, "deblur_late_densify_enabled", "false"), False)
    if enabled and auto and enable_late_densify:
        densify_b_from = clamp_int(round(total * 0.55), warmup + 1, total)
        densify_b_until = clamp_int(round(total * 0.85), densify_b_from + 1, total)
    else:
        densify_b_from = -1
        densify_b_until = -1

    late_prune_from = clamp_int(
        int(getattr(opt, "fastgs_late_prune_from_iter", round(total * 0.70))),
        warmup + 1,
        total,
    )
    late_prune_until = clamp_int(
        int(getattr(opt, "fastgs_late_prune_until_iter", total)),
        late_prune_from + 1,
        total,
    )
    opacity_reset_until = warmup if enabled and auto else total

    opt.deblur_warmup_iters = warmup
    opt.densify_from_iter = densify_a_from
    opt.densify_until_iter = densify_b_until if densify_b_until > 0 else densify_a_until
    opt.fastgs_late_prune_from_iter = late_prune_from
    opt.fastgs_late_prune_until_iter = late_prune_until
    
    return {
        "enabled": enabled and auto,
        "profile": str(getattr(opt, "deblur_schedule_profile", "quality")).strip().lower(),
        "deblur_loss_from": deblur_loss_from,
        "deblur_loss_until": deblur_loss_until,
        "densify_a_from": densify_a_from,
        "densify_a_until": densify_a_until,
        "densify_b_from": densify_b_from,
        "densify_b_until": densify_b_until,
        "opacity_reset_until": opacity_reset_until,
        "late_prune_from": late_prune_from,
        "late_prune_until": late_prune_until,
    }


def schedule_deblur_loss_active(iteration, schedule, deblur_state):
    if not deblur_state.enabled:
        return False
    return int(schedule["deblur_loss_from"]) < iteration <= int(schedule["deblur_loss_until"])


def schedule_densify_active(iteration, schedule):
    in_first_densify = int(schedule["densify_a_from"]) <= iteration < int(schedule["densify_a_until"])
    in_second_densify = int(schedule["densify_b_from"]) <= iteration < int(schedule["densify_b_until"])
    return bool(in_first_densify or in_second_densify)


def schedule_opacity_reset_active(iteration, schedule):
    return iteration < int(schedule["opacity_reset_until"])


def schedule_late_prune_active(iteration, schedule):
    return int(schedule["late_prune_from"]) <= iteration < int(schedule["late_prune_until"])


def collect_final_metrics(model_path, iteration, scene, gaussians, pipe, background, mult, training_seconds):
    metrics = {
        "training_time_seconds": round(float(training_seconds), 3),
        "training_time_minutes": round(float(training_seconds) / 60.0, 3),
        "final_gaussians": int(gaussians.get_xyz.shape[0]),
    }
    metrics.update(storage_metrics(model_path, iteration))

    try:
        metrics.update(evaluate_render_quality(scene, gaussians, pipe, background, mult))
    except Exception as exc:
        metrics["final_quality_metrics_error"] = str(exc)
        print(f"[FINAL METRICS][WARN] failed to compute quality metrics: {exc}")

    try:
        speed_cameras = select_final_eval_cameras(scene)[1]
        metrics.update(measure_render_speed(speed_cameras, gaussians, pipe, background, mult))
    except Exception as exc:
        metrics["final_render_speed_error"] = str(exc)
        print(f"[FINAL METRICS][WARN] failed to compute render speed: {exc}")

    print_final_metrics(metrics)
    return metrics


def select_final_eval_cameras(scene):
    test_cameras = scene.getTestCameras()
    if test_cameras:
        return "test", test_cameras
    return "train", scene.getTrainCameras()


def evaluate_render_quality(scene, gaussians, pipe, background, mult):
    split, cameras = select_final_eval_cameras(scene)
    if not cameras:
        return {
            "final_eval_split": split,
            "final_eval_images": 0,
            "final_psnr": None,
            "final_ssim": None,
            "final_lpips": None,
        }

    from lpipsPyTorch.modules.lpips import LPIPS

    lpips_model = LPIPS(net_type="vgg").to("cuda").eval()
    l1_total = 0.0
    psnr_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    with torch.no_grad():
        for viewpoint in cameras:
            image = torch.clamp(render_fastgs(viewpoint, gaussians, pipe, background, mult)["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            image_batch = image.unsqueeze(0)
            gt_batch = gt_image.unsqueeze(0)
            l1_total += float(l1_loss(image, gt_image).mean().item())
            psnr_total += float(psnr(image_batch, gt_batch).mean().item())
            ssim_total += float(fast_ssim(image_batch, gt_batch).mean().item())
            lpips_total += float(lpips_model(image_batch, gt_batch).mean().item())

    count = len(cameras)
    torch.cuda.empty_cache()
    return {
        "final_eval_split": split,
        "final_eval_images": count,
        "final_l1": round(l1_total / count, 6),
        "final_psnr": round(psnr_total / count, 6),
        "final_ssim": round(ssim_total / count, 6),
        "final_lpips": round(lpips_total / count, 6),
    }


def measure_render_speed(cameras, gaussians, pipe, background, mult):
    if not cameras:
        return {
            "final_render_frames": 0,
            "final_render_fps": None,
            "final_render_time_ms": None,
        }

    with torch.no_grad():
        render_fastgs(cameras[0], gaussians, pipe, background, mult)["render"]
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for viewpoint in cameras:
            render_fastgs(viewpoint, gaussians, pipe, background, mult)["render"]
        end.record()
        torch.cuda.synchronize()

    elapsed_ms = float(start.elapsed_time(end))
    frame_count = len(cameras)
    render_time_ms = elapsed_ms / max(frame_count, 1)
    fps = 1000.0 / render_time_ms if render_time_ms > 0 else None
    return {
        "final_render_frames": frame_count,
        "final_render_total_ms": round(elapsed_ms, 3),
        "final_render_time_ms": round(render_time_ms, 3),
        "final_render_fps": round(fps, 3) if fps is not None else None,
    }


def storage_metrics(model_path, iteration):
    metrics = {
        "final_ply_path": None,
        "final_ply_bytes": None,
        "final_ply_mb": None,
        "final_model_dir_bytes": None,
        "final_model_dir_mb": None,
    }
    if not model_path:
        return metrics

    final_ply = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if os.path.exists(final_ply):
        ply_bytes = os.path.getsize(final_ply)
        metrics["final_ply_path"] = final_ply
        metrics["final_ply_bytes"] = int(ply_bytes)
        metrics["final_ply_mb"] = round(ply_bytes / (1024.0 * 1024.0), 3)

    total_bytes = 0
    for root, _dirs, files in os.walk(model_path):
        for filename in files:
            path = os.path.join(root, filename)
            try:
                total_bytes += os.path.getsize(path)
            except OSError:
                pass
    metrics["final_model_dir_bytes"] = int(total_bytes)
    metrics["final_model_dir_mb"] = round(total_bytes / (1024.0 * 1024.0), 3)
    return metrics


def print_final_metrics(metrics):
    print(
        "[FINAL METRICS] "
        f"split={metrics.get('final_eval_split')} "
        f"images={metrics.get('final_eval_images')} "
        f"PSNR={metrics.get('final_psnr')} "
        f"SSIM={metrics.get('final_ssim')} "
        f"LPIPS={metrics.get('final_lpips')} "
        f"FPS={metrics.get('final_render_fps')} "
        f"render_ms={metrics.get('final_render_time_ms')} "
        f"final_ply_mb={metrics.get('final_ply_mb')} "
        f"model_dir_mb={metrics.get('final_model_dir_mb')} "
        f"training_time_s={metrics.get('training_time_seconds')}"
    )


def write_deblur_metrics(
    model_path,
    gaussians,
    deblur_state,
    blur_registry,
    registry_match_metrics,
    deblur_photometric_views,
    deblur_clear_train_cameras,
    deblur_final_pruned,
    deblur_extra_points_added,
    deblur_mandatory_extra_points_added,
    deblur_extra_densify_steps,
    sharp_score_sample_count,
    sharp_score_skipped_steps,
    topology_blur_stats_skipped,
    late_prune_steps,
    late_prune_removed,
    late_prune_deferred,
    late_prune_scale_candidates,
    final_prune_metrics,
    last_deblur_reg,
    opt,
    deblur_schedule=None,
    final_metrics=None,
):
    if not model_path:
        return
    registry_items = unique_blur_registry_items(blur_registry)
    training_blur_frames = sum(1 for item in registry_items if item.get("blurred") and not item.get("rejected"))
    rejected_blur_frames = sum(1 for item in registry_items if item.get("blurred") and item.get("rejected"))
    clear_train_cameras = sum(1 for item in registry_items if not item.get("blurred") and not item.get("rejected"))
    metrics = {
        "deblur_enabled": bool(deblur_state.enabled),
        "deblur_mode": getattr(opt, "deblur_mode", "sharp"),
        "deblur_warmup_iters": getattr(opt, "deblur_warmup_iters", None),
        "deblur_xyz_lr_scale": getattr(opt, "deblur_xyz_lr_scale", None),
        "deblur_extra_points_enabled": getattr(opt, "deblur_extra_points_enabled", None),
        "deblur_sharp_refine_enabled": getattr(opt, "deblur_sharp_refine_enabled", None),
        "deblur_sharp_refine_from_iter": getattr(opt, "deblur_sharp_refine_from_iter", None),
        "deblur_sharp_refine_clear_only": getattr(opt, "deblur_sharp_refine_clear_only", None),
        "deblur_topology_sharp_only": getattr(opt, "deblur_topology_sharp_only", None),
        "deblur_training_blur_frames": training_blur_frames,
        "deblur_rejected_blur_frames": rejected_blur_frames,
        "deblur_clear_train_cameras": clear_train_cameras,
        "deblur_runtime_clear_train_cameras": registry_match_metrics.get("clear_runtime_cameras", 0),
        "deblur_runtime_blur_train_cameras": registry_match_metrics.get("blur_runtime_cameras", 0),
        "deblur_registry_camera_count": registry_match_metrics.get("camera_count", 0),
        "deblur_registry_key_count": registry_match_metrics.get("registry_keys", 0),
        "deblur_registry_matched_cameras": registry_match_metrics.get("matched_cameras", 0),
        "deblur_clear_render_views": deblur_clear_train_cameras,
        "deblur_photometric_views": deblur_photometric_views,
        "deblur_extra_points_added": deblur_extra_points_added,
        "deblur_mandatory_extra_points_added": deblur_mandatory_extra_points_added,
        "deblur_extra_densify_steps": deblur_extra_densify_steps,
        "sharp_score_sample_count": sharp_score_sample_count,
        "sharp_score_skipped_steps": sharp_score_skipped_steps,
        "topology_blur_stats_skipped": topology_blur_stats_skipped,
        "late_prune_steps": late_prune_steps,
        "late_prune_removed": late_prune_removed,
        "late_prune_deferred": late_prune_deferred,
        "late_prune_scale_candidates": late_prune_scale_candidates,
        "deblur_final_prune_uses_sharp_score": True,
        "fastgs_final_prune_enabled": getattr(opt, "fastgs_final_prune_enabled", None),
        "deblur_final_pruned": deblur_final_pruned,
        "final_prune_metrics": final_prune_metrics,
        "fastgs_size_prune_from_iter": getattr(opt, "fastgs_size_prune_from_iter", None),
        "fastgs_size_prune_max_screen_size": getattr(opt, "fastgs_size_prune_max_screen_size", None),
        "fastgs_size_prune_max_world_scale_ratio": getattr(opt, "fastgs_size_prune_max_world_scale_ratio", None),
        "fastgs_late_prune_max_world_scale_ratio": getattr(opt, "fastgs_late_prune_max_world_scale_ratio", None),
        "fastgs_late_prune_max_fraction": getattr(opt, "fastgs_late_prune_max_fraction", None),
        "fastgs_final_prune_max_world_scale_ratio": getattr(opt, "fastgs_final_prune_max_world_scale_ratio", None),
        "fastgs_vcd_blend_alpha": getattr(opt, "fastgs_vcd_blend_alpha", None),
        "fastgs_vcd_score_thresh": getattr(opt, "fastgs_vcd_score_thresh", None),
        "fastgs_vcp_blur_protect_weight": getattr(opt, "fastgs_vcp_blur_protect_weight", None),
        "deblur_last_transform_regularization": last_deblur_reg,
        "deblur_schedule": deblur_schedule or {},
        "final_gaussians": int(gaussians.get_xyz.shape[0]),
        **(final_metrics or {}),
        **deblur_state.metrics(),
    }
    path = os.path.join(model_path, "fastgs_deblur_metrics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"[deblur] metrics written: {path} {metrics}")
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[FASTGS_ITERATIONS])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[FASTGS_ITERATIONS])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[FASTGS_ITERATIONS])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.websockets
    )

    # All done
    print("\nTraining complete.")
