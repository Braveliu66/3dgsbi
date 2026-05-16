from __future__ import annotations

import os
import re
import json
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.config import get_settings
from app.fine.fastgs_defaults import (
    FASTGS_DATA_DEVICE,
    FASTGS_DEBLUR_AUTO_SCHEDULE,
    FASTGS_DEBLUR_BLURRED_VIEWS_ONLY,
    FASTGS_DEBLUR_ENABLED,
    FASTGS_DEBLUR_EXTRA_POINTS_ENABLED,
    FASTGS_DEBLUR_GTNET_LR,
    FASTGS_DEBLUR_HIDDEN,
    FASTGS_DEBLUR_LAMBDA_P,
    FASTGS_DEBLUR_LAMBDA_S,
    FASTGS_DEBLUR_LATE_DENSIFY_ENABLED,
    FASTGS_DEBLUR_MAX_CLAMP,
    FASTGS_DEBLUR_MAX_POSITION_DELTA,
    FASTGS_DEBLUR_MODE,
    FASTGS_DEBLUR_NUM_MOMENTS,
    FASTGS_DEBLUR_SCHEDULE_PROFILE,
    FASTGS_DEBLUR_SHARP_REFINE_CLEAR_ONLY,
    FASTGS_DEBLUR_SHARP_REFINE_ENABLED,
    FASTGS_DEBLUR_SHARP_REFINE_FROM_ITER,
    FASTGS_DEBLUR_TOPOLOGY_SHARP_ONLY,
    FASTGS_DEBLUR_TRANSFORM_REG_WEIGHT,
    FASTGS_DEBLUR_WARMUP_ITERS,
    FASTGS_DEBLUR_WIDTH,
    FASTGS_DEBLUR_XYZ_LR_SCALE,
    FASTGS_DENSE,
    FASTGS_DENSIFICATION_INTERVAL,
    FASTGS_DENSIFY_FROM_ITER,
    FASTGS_DENSIFY_GRAD_THRESHOLD,
    FASTGS_DENSIFY_UNTIL_ITER,
    FASTGS_FEATURE_LR,
    FASTGS_FINAL_PRUNE_MAX_WORLD_SCALE_RATIO,
    FASTGS_FINAL_PRUNE_MIN_OPACITY,
    FASTGS_FINAL_PRUNE_SCORE_THRESH,
    FASTGS_GRAD_ABS_THRESH,
    FASTGS_GRAD_THRESH,
    FASTGS_HIGHFEATURE_LR,
    FASTGS_LAMBDA_DSSIM,
    FASTGS_LATE_PRUNE_ENABLED,
    FASTGS_LATE_PRUNE_FROM_ITER,
    FASTGS_LATE_PRUNE_INTERVAL,
    FASTGS_LATE_PRUNE_MAX_WORLD_SCALE_RATIO,
    FASTGS_LATE_PRUNE_MIN_OPACITY,
    FASTGS_LATE_PRUNE_SCORE_THRESH,
    FASTGS_LATE_PRUNE_UNTIL_ITER,
    FASTGS_LOSS_THRESH,
    FASTGS_LOWFEATURE_LR,
    FASTGS_MULT,
    FASTGS_OPACITY_LR,
    FASTGS_OPACITY_RESET_INTERVAL,
    FASTGS_OPTIMIZER_TYPE,
    FASTGS_PERCENT_DENSE,
    FASTGS_POSITION_LR_DELAY_MULT,
    FASTGS_POSITION_LR_FINAL,
    FASTGS_POSITION_LR_INIT,
    FASTGS_RESOLUTION,
    FASTGS_ROTATION_LR,
    FASTGS_SAMPLE_CAMERAS,
    FASTGS_SCALING_LR,
    FASTGS_SHFEATURE_LR,
    FASTGS_SIZE_PRUNE_FROM_ITER,
    FASTGS_SIZE_PRUNE_MAX_SCREEN_SIZE,
    FASTGS_SIZE_PRUNE_MAX_WORLD_SCALE_RATIO,
)
from app.fine.option_utils import read_float, read_int
from app.fine.types import FineFailure


Progress = Callable[[str, int, str], None]


@dataclass(slots=True)
class OfficialFastGSTrainResult:
    ply_path: Path
    iterations: int
    metrics: dict[str, Any]


def train_official_fastgs_big(
    *,
    scene_dir: Path,
    output_dir: Path,
    iterations: int,
    options: dict[str, Any],
    progress: Progress,
) -> OfficialFastGSTrainResult:
    vendor_root = _fastgs_vendor_root()
    _require_fastgs_vendor(vendor_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    iterations = read_int((options or {}).get("fine_iterations"), iterations, minimum=5_000, maximum=60_000)
    options = options or {}
    schedule = resolve_fastgs_schedule(iterations)
    densification_interval = read_int(options.get("fine_densification_interval"), schedule["densification_interval"], minimum=1, maximum=10_000)
    data_device = str(options.get("fine_data_device") or FASTGS_DATA_DEVICE).strip().lower()
    if data_device not in {"cpu", "cuda"}:
        raise FineFailure("UNSUPPORTED_FASTGS_DATA_DEVICE", f"Unsupported FastGS data_device: {data_device}")
    position_lr_init = read_float(options.get("fine_position_lr_init"), FASTGS_POSITION_LR_INIT, minimum=1e-8, maximum=1.0)
    position_lr_final = read_float(options.get("fine_position_lr_final"), FASTGS_POSITION_LR_FINAL, minimum=1e-9, maximum=1.0)
    position_lr_delay_mult = read_float(options.get("fine_position_lr_delay_mult"), FASTGS_POSITION_LR_DELAY_MULT, minimum=0.0, maximum=1.0)
    position_lr_max_steps = read_int(options.get("fine_position_lr_max_steps"), schedule["position_lr_max_steps"], minimum=1, maximum=100_000)
    feature_lr = read_float(options.get("fine_feature_lr"), FASTGS_FEATURE_LR, minimum=1e-7, maximum=1.0)
    shfeature_lr = read_float(options.get("fine_shfeature_lr"), FASTGS_SHFEATURE_LR, minimum=1e-7, maximum=1.0)
    opacity_lr = read_float(options.get("fine_opacity_lr"), FASTGS_OPACITY_LR, minimum=1e-7, maximum=1.0)
    scaling_lr = read_float(options.get("fine_scaling_lr"), FASTGS_SCALING_LR, minimum=1e-7, maximum=1.0)
    rotation_lr = read_float(options.get("fine_rotation_lr"), FASTGS_ROTATION_LR, minimum=1e-7, maximum=1.0)
    percent_dense = read_float(options.get("fine_percent_dense"), FASTGS_PERCENT_DENSE, minimum=0.0, maximum=1.0)
    grad_thresh = read_float(_first_option(options, "fine_grad_thresh", "fine_fastgs_grad_thresh"), FASTGS_GRAD_THRESH, minimum=1e-7, maximum=0.1)
    grad_abs_thresh = read_float(_first_option(options, "fine_grad_abs_thresh", "fine_fastgs_grad_abs_thresh"), FASTGS_GRAD_ABS_THRESH, minimum=1e-7, maximum=0.1)
    densify_grad_threshold = read_float(options.get("fine_densify_grad_threshold"), FASTGS_DENSIFY_GRAD_THRESHOLD, minimum=1e-7, maximum=0.1)
    dense = read_float(options.get("fine_dense"), FASTGS_DENSE, minimum=0.0, maximum=1.0)
    mult = read_float(options.get("fine_mult"), FASTGS_MULT, minimum=0.01, maximum=10.0)
    loss_thresh = read_float(options.get("fine_fastgs_loss_thresh"), FASTGS_LOSS_THRESH, minimum=0.0, maximum=1.0)
    sample_cameras = read_int(options.get("fine_fastgs_sample_cameras"), FASTGS_SAMPLE_CAMERAS, minimum=1, maximum=32)
    lambda_dssim = read_float(options.get("fine_lambda_dssim"), FASTGS_LAMBDA_DSSIM, minimum=0.0, maximum=1.0)
    highfeature_lr = _optional_float(options, "fine_highfeature_lr", fallback=FASTGS_HIGHFEATURE_LR, minimum=1e-7, maximum=1.0)
    lowfeature_lr = _optional_float(options, "fine_lowfeature_lr", fallback=FASTGS_LOWFEATURE_LR, minimum=1e-7, maximum=1.0)
    resolution = _optional_int(options, "fine_train_resolution", fallback=min(settings.fine_image_max_side, FASTGS_RESOLUTION), minimum=1, maximum=16_384)
    densify_from_iter = read_int(options.get("fine_densify_from_iter"), schedule["densify_from_iter"], minimum=0, maximum=100_000)
    densify_until_iter = read_int(options.get("fine_densify_until_iter"), schedule["densify_until_iter"], minimum=0, maximum=100_000)
    opacity_reset_interval = read_int(options.get("fine_opacity_reset_interval"), schedule["opacity_reset_interval"], minimum=1, maximum=100_000)
    size_prune_from_iter = read_int(options.get("fine_fastgs_size_prune_from_iter"), FASTGS_SIZE_PRUNE_FROM_ITER, minimum=0, maximum=100_000)
    size_prune_max_screen_size = read_int(options.get("fine_fastgs_size_prune_max_screen_size"), FASTGS_SIZE_PRUNE_MAX_SCREEN_SIZE, minimum=1, maximum=10_000)
    size_prune_max_world_scale_ratio = read_float(options.get("fine_fastgs_size_prune_max_world_scale_ratio"), FASTGS_SIZE_PRUNE_MAX_WORLD_SCALE_RATIO, minimum=0.0, maximum=1.0)
    late_prune_enabled = _bool_string(options.get("fine_fastgs_late_prune_enabled"), FASTGS_LATE_PRUNE_ENABLED)
    late_prune_interval = read_int(options.get("fine_fastgs_late_prune_interval"), schedule["late_prune_interval"], minimum=1, maximum=100_000)
    late_prune_from_iter = read_int(options.get("fine_fastgs_late_prune_from_iter"), schedule["late_prune_from_iter"], minimum=0, maximum=100_000)
    late_prune_until_iter = read_int(options.get("fine_fastgs_late_prune_until_iter"), schedule["late_prune_until_iter"], minimum=0, maximum=100_000)
    late_prune_min_opacity = read_float(options.get("fine_fastgs_late_prune_min_opacity"), FASTGS_LATE_PRUNE_MIN_OPACITY, minimum=0.001, maximum=0.2)
    late_prune_score_thresh = read_float(options.get("fine_fastgs_late_prune_score_thresh"), FASTGS_LATE_PRUNE_SCORE_THRESH, minimum=0.5, maximum=1.0)
    late_prune_max_world_scale_ratio = read_float(options.get("fine_fastgs_late_prune_max_world_scale_ratio"), FASTGS_LATE_PRUNE_MAX_WORLD_SCALE_RATIO, minimum=0.0, maximum=1.0)
    final_prune_min_opacity = read_float(options.get("fine_fastgs_final_prune_min_opacity"), FASTGS_FINAL_PRUNE_MIN_OPACITY, minimum=0.001, maximum=0.2)
    final_prune_score_thresh = read_float(options.get("fine_fastgs_final_prune_score_thresh"), FASTGS_FINAL_PRUNE_SCORE_THRESH, minimum=0.5, maximum=1.0)
    final_prune_max_world_scale_ratio = read_float(options.get("fine_fastgs_final_prune_max_world_scale_ratio"), FASTGS_FINAL_PRUNE_MAX_WORLD_SCALE_RATIO, minimum=0.0, maximum=1.0)
    deblur_enabled = _choice_string(options.get("fine_deblur_enabled"), FASTGS_DEBLUR_ENABLED, {"auto", "true", "false"})
    deblur_mode = _choice_string(options.get("fine_deblur_mode"), FASTGS_DEBLUR_MODE, {"sharp", "defocus", "motion", "mixed"})
    deblur_blur_registry = str(options.get("fine_deblur_blur_registry") or "").strip()
    deblur_auto_schedule = _choice_string(options.get("fine_deblur_auto_schedule"), FASTGS_DEBLUR_AUTO_SCHEDULE, {"true", "false"})
    deblur_schedule_profile = _choice_string(
        options.get("fine_deblur_schedule_profile"),
        FASTGS_DEBLUR_SCHEDULE_PROFILE,
        {"quality", "balanced", "fast"},
    )
    deblur_late_densify_enabled = _choice_string(
        options.get("fine_deblur_late_densify_enabled"),
        FASTGS_DEBLUR_LATE_DENSIFY_ENABLED,
        {"true", "false"},
    )
    deblur_warmup_iters = read_int(options.get("fine_deblur_warmup_iters"), schedule["deblur_warmup_iters"], minimum=0, maximum=max(0, iterations - 1))
    deblur_extra_points_enabled = _choice_string(options.get("fine_deblur_extra_points_enabled"), FASTGS_DEBLUR_EXTRA_POINTS_ENABLED, {"true", "false"})
    deblur_sharp_refine_enabled = _choice_string(options.get("fine_deblur_sharp_refine_enabled"), FASTGS_DEBLUR_SHARP_REFINE_ENABLED, {"true", "false"})
    deblur_sharp_refine_from_iter = read_int(options.get("fine_deblur_sharp_refine_from_iter"), FASTGS_DEBLUR_SHARP_REFINE_FROM_ITER, minimum=0, maximum=max(0, iterations - 1))
    deblur_sharp_refine_clear_only = _choice_string(options.get("fine_deblur_sharp_refine_clear_only"), FASTGS_DEBLUR_SHARP_REFINE_CLEAR_ONLY, {"true", "false"})
    deblur_topology_sharp_only = _choice_string(options.get("fine_deblur_topology_sharp_only"), FASTGS_DEBLUR_TOPOLOGY_SHARP_ONLY, {"true", "false"})
    deblur_num_moments = read_int(options.get("fine_deblur_num_moments"), FASTGS_DEBLUR_NUM_MOMENTS, minimum=1, maximum=8)
    deblur_gtnet_lr = read_float(options.get("fine_deblur_gtnet_lr"), FASTGS_DEBLUR_GTNET_LR, minimum=1e-6, maximum=0.1)
    deblur_hidden = read_int(options.get("fine_deblur_hidden"), FASTGS_DEBLUR_HIDDEN, minimum=1, maximum=8)
    deblur_width = read_int(options.get("fine_deblur_width"), FASTGS_DEBLUR_WIDTH, minimum=16, maximum=256)
    deblur_lambda_s = read_float(options.get("fine_deblur_lambda_s"), FASTGS_DEBLUR_LAMBDA_S, minimum=0.0, maximum=0.1)
    deblur_lambda_p = read_float(options.get("fine_deblur_lambda_p"), FASTGS_DEBLUR_LAMBDA_P, minimum=0.0, maximum=0.1)
    deblur_max_clamp = read_float(options.get("fine_deblur_max_clamp"), FASTGS_DEBLUR_MAX_CLAMP, minimum=1.0, maximum=1.8)
    deblur_max_position_delta = read_float(options.get("fine_deblur_max_position_delta"), FASTGS_DEBLUR_MAX_POSITION_DELTA, minimum=0.0, maximum=1.0)
    deblur_transform_reg_weight = read_float(options.get("fine_deblur_transform_reg_weight"), FASTGS_DEBLUR_TRANSFORM_REG_WEIGHT, minimum=0.0, maximum=1.0)
    deblur_xyz_lr_scale = read_float(options.get("fine_deblur_xyz_lr_scale"), FASTGS_DEBLUR_XYZ_LR_SCALE, minimum=0.0, maximum=1.0)
    deblur_blurred_views_only = _choice_string(options.get("fine_deblur_blurred_views_only"), FASTGS_DEBLUR_BLURRED_VIEWS_ONLY, {"true", "false"})

    command = [
        sys.executable,
        str(vendor_root / "train.py"),
        "-s",
        str(scene_dir.resolve()),
        "-i",
        "images",
        "-m",
        str(output_dir.resolve()),
        "--iterations",
        str(iterations),
        "--save_iterations",
        str(iterations),
        "--test_iterations",
        str(iterations),
        "--checkpoint_iterations",
        str(iterations),
        "--densification_interval",
        str(densification_interval),
        "--opacity_reset_interval",
        str(opacity_reset_interval),
        "--densify_from_iter",
        str(densify_from_iter),
        "--densify_until_iter",
        str(densify_until_iter),
        "--fastgs_size_prune_from_iter",
        str(size_prune_from_iter),
        "--fastgs_size_prune_max_screen_size",
        str(size_prune_max_screen_size),
        "--fastgs_size_prune_max_world_scale_ratio",
        str(size_prune_max_world_scale_ratio),
        "--densify_grad_threshold",
        str(densify_grad_threshold),
        "--optimizer_type",
        FASTGS_OPTIMIZER_TYPE,
        "--data_device",
        data_device,
        "--position_lr_init",
        str(position_lr_init),
        "--position_lr_final",
        str(position_lr_final),
        "--position_lr_delay_mult",
        str(position_lr_delay_mult),
        "--position_lr_max_steps",
        str(position_lr_max_steps),
        "--feature_lr",
        str(feature_lr),
        "--shfeature_lr",
        str(shfeature_lr),
        "--opacity_lr",
        str(opacity_lr),
        "--scaling_lr",
        str(scaling_lr),
        "--rotation_lr",
        str(rotation_lr),
        "--percent_dense",
        str(percent_dense),
        "--loss_thresh",
        str(loss_thresh),
        "--fastgs_sample_cameras",
        str(sample_cameras),
        "--grad_thresh",
        str(grad_thresh),
        "--grad_abs_thresh",
        str(grad_abs_thresh),
        "--dense",
        str(dense),
        "--mult",
        str(mult),
        "--lambda_dssim",
        str(lambda_dssim),
        "--deblur_enabled",
        deblur_enabled,
        "--deblur_mode",
        deblur_mode,
        "--deblur_auto_schedule",
        deblur_auto_schedule,
        "--deblur_schedule_profile",
        deblur_schedule_profile,
        "--deblur_late_densify_enabled",
        deblur_late_densify_enabled,
        "--deblur_warmup_iters",
        str(deblur_warmup_iters),
        "--deblur_extra_points_enabled",
        deblur_extra_points_enabled,
        "--deblur_sharp_refine_enabled",
        deblur_sharp_refine_enabled,
        "--deblur_sharp_refine_from_iter",
        str(deblur_sharp_refine_from_iter),
        "--deblur_sharp_refine_clear_only",
        deblur_sharp_refine_clear_only,
        "--deblur_topology_sharp_only",
        deblur_topology_sharp_only,
        "--deblur_num_moments",
        str(deblur_num_moments),
        "--deblur_gtnet_lr",
        str(deblur_gtnet_lr),
        "--deblur_hidden",
        str(deblur_hidden),
        "--deblur_width",
        str(deblur_width),
        "--deblur_lambda_s",
        str(deblur_lambda_s),
        "--deblur_lambda_p",
        str(deblur_lambda_p),
        "--deblur_max_clamp",
        str(deblur_max_clamp),
        "--deblur_max_position_delta",
        str(deblur_max_position_delta),
        "--deblur_transform_reg_weight",
        str(deblur_transform_reg_weight),
        "--deblur_xyz_lr_scale",
        str(deblur_xyz_lr_scale),
        "--deblur_blurred_views_only",
        deblur_blurred_views_only,
        "--fastgs_final_prune_min_opacity",
        str(final_prune_min_opacity),
        "--fastgs_final_prune_score_thresh",
        str(final_prune_score_thresh),
        "--fastgs_final_prune_max_world_scale_ratio",
        str(final_prune_max_world_scale_ratio),
        "--fastgs_late_prune_enabled",
        late_prune_enabled,
        "--fastgs_late_prune_interval",
        str(late_prune_interval),
        "--fastgs_late_prune_from_iter",
        str(late_prune_from_iter),
        "--fastgs_late_prune_until_iter",
        str(late_prune_until_iter),
        "--fastgs_late_prune_min_opacity",
        str(late_prune_min_opacity),
        "--fastgs_late_prune_score_thresh",
        str(late_prune_score_thresh),
        "--fastgs_late_prune_max_world_scale_ratio",
        str(late_prune_max_world_scale_ratio),
    ]
    if deblur_blur_registry:
        command.extend(["--deblur_blur_registry", deblur_blur_registry])
    if highfeature_lr is not None:
        command.extend(["--highfeature_lr", str(highfeature_lr)])
    if lowfeature_lr is not None:
        command.extend(["--lowfeature_lr", str(lowfeature_lr)])
    if resolution is not None:
        command.extend(["-r", str(resolution)])

    log_path = output_dir / "fastgs_train.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(vendor_root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )

    progress("fine_fastgs_train_start", 44, f"training official FastGS-Big for {iterations} iterations")
    tail: deque[str] = deque(maxlen=80)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(" ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=str(vendor_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line_number, line in enumerate(process.stdout, start=1):
            log_file.write(line)
            tail.append(line.rstrip())
            if _is_progress_line(line):
                progress("fine_fastgs_training", 70, line.strip()[:220] or "FastGS-Big training")
        return_code = process.wait()

    if return_code != 0:
        raise FineFailure("FASTGS_TRAIN_FAILED", "\n".join(tail) or f"FastGS train.py exited with {return_code}")

    ply_path = _find_final_ply(output_dir, iterations)
    if ply_path is None:
        raise FineFailure("FASTGS_PLY_NOT_FOUND", f"FastGS did not create point_cloud.ply under {output_dir / 'point_cloud'}")

    metrics = {
        "training_backend": "official_fastgs_big",
        "fastgs_vendor_root": str(vendor_root),
        "fastgs_mode": "big",
        "iterations": iterations,
        "data_device": data_device,
        "optimizer_type": FASTGS_OPTIMIZER_TYPE,
        "densification_interval": densification_interval,
        "opacity_reset_interval": opacity_reset_interval,
        "fastgs_size_prune_from_iter": size_prune_from_iter,
        "fastgs_size_prune_max_screen_size": size_prune_max_screen_size,
        "fastgs_size_prune_max_world_scale_ratio": size_prune_max_world_scale_ratio,
        "densify_from_iter": densify_from_iter,
        "densify_until_iter": densify_until_iter,
        "densify_grad_threshold": densify_grad_threshold,
        "position_lr_init": position_lr_init,
        "position_lr_final": position_lr_final,
        "position_lr_delay_mult": position_lr_delay_mult,
        "position_lr_max_steps": position_lr_max_steps,
        "feature_lr": feature_lr,
        "shfeature_lr": shfeature_lr,
        "opacity_lr": opacity_lr,
        "scaling_lr": scaling_lr,
        "rotation_lr": rotation_lr,
        "percent_dense": percent_dense,
        "loss_thresh": loss_thresh,
        "fastgs_sample_cameras": sample_cameras,
        "grad_thresh": grad_thresh,
        "grad_abs_thresh": grad_abs_thresh,
        "dense": dense,
        "mult": mult,
        "lambda_dssim": lambda_dssim,
        "fastgs_final_prune_min_opacity": final_prune_min_opacity,
        "fastgs_final_prune_score_thresh": final_prune_score_thresh,
        "fastgs_final_prune_max_world_scale_ratio": final_prune_max_world_scale_ratio,
        "fastgs_late_prune_enabled": late_prune_enabled,
        "fastgs_late_prune_interval": late_prune_interval,
        "fastgs_late_prune_from_iter": late_prune_from_iter,
        "fastgs_late_prune_until_iter": late_prune_until_iter,
        "fastgs_late_prune_min_opacity": late_prune_min_opacity,
        "fastgs_late_prune_score_thresh": late_prune_score_thresh,
        "fastgs_late_prune_max_world_scale_ratio": late_prune_max_world_scale_ratio,
        "deblur_enabled": deblur_enabled,
        "deblur_mode": deblur_mode,
        "deblur_blur_registry": deblur_blur_registry or None,
        "deblur_auto_schedule": deblur_auto_schedule,
        "deblur_schedule_profile": deblur_schedule_profile,
        "deblur_late_densify_enabled": deblur_late_densify_enabled,
        "deblur_warmup_iters": deblur_warmup_iters,
        "deblur_extra_points_enabled": deblur_extra_points_enabled,
        "deblur_sharp_refine_enabled": deblur_sharp_refine_enabled,
        "deblur_sharp_refine_from_iter": deblur_sharp_refine_from_iter,
        "deblur_sharp_refine_clear_only": deblur_sharp_refine_clear_only,
        "deblur_topology_sharp_only": deblur_topology_sharp_only,
        "deblur_num_moments": deblur_num_moments,
        "deblur_gtnet_lr": deblur_gtnet_lr,
        "deblur_hidden": deblur_hidden,
        "deblur_width": deblur_width,
        "deblur_lambda_s": deblur_lambda_s,
        "deblur_lambda_p": deblur_lambda_p,
        "deblur_max_clamp": deblur_max_clamp,
        "deblur_max_position_delta": deblur_max_position_delta,
        "deblur_transform_reg_weight": deblur_transform_reg_weight,
        "deblur_xyz_lr_scale": deblur_xyz_lr_scale,
        "deblur_blurred_views_only": deblur_blurred_views_only,
        "final_ply_bytes": ply_path.stat().st_size,
        "fastgs_log_path": str(log_path),
    }
    if highfeature_lr is not None:
        metrics["highfeature_lr"] = highfeature_lr
    if lowfeature_lr is not None:
        metrics["lowfeature_lr"] = lowfeature_lr
    if resolution is not None:
        metrics["resolution"] = resolution
    deblur_metrics_path = output_dir / "fastgs_deblur_metrics.json"
    if deblur_metrics_path.exists():
        try:
            metrics.update(json.loads(deblur_metrics_path.read_text(encoding="utf-8")))
            metrics["fastgs_deblur_metrics_path"] = str(deblur_metrics_path)
        except Exception:
            metrics["fastgs_deblur_metrics_path"] = str(deblur_metrics_path)
    return OfficialFastGSTrainResult(ply_path=ply_path, iterations=iterations, metrics=metrics)


def _fastgs_vendor_root() -> Path:
    configured = os.getenv("FASTGS_VENDOR_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parent / "vendor" / "fastgs"


def resolve_fastgs_schedule(iterations: int) -> dict[str, int]:
    iterations = max(1, int(iterations))
    return {
        "densification_interval": FASTGS_DENSIFICATION_INTERVAL,
        "densify_from_iter": min(FASTGS_DENSIFY_FROM_ITER, max(0, iterations - 1)),
        "densify_until_iter": min(FASTGS_DENSIFY_UNTIL_ITER, iterations),
        "opacity_reset_interval": FASTGS_OPACITY_RESET_INTERVAL,
        "late_prune_interval": FASTGS_LATE_PRUNE_INTERVAL,
        "late_prune_from_iter": min(FASTGS_LATE_PRUNE_FROM_ITER, iterations),
        "late_prune_until_iter": min(FASTGS_LATE_PRUNE_UNTIL_ITER, iterations),
        "deblur_warmup_iters": min(max(0, FASTGS_DEBLUR_WARMUP_ITERS), max(0, iterations - 1)),
        "position_lr_max_steps": iterations,
    }


def _require_fastgs_vendor(vendor_root: Path) -> None:
    missing = [
        str(path)
        for path in (
            vendor_root / "train.py",
            vendor_root / "gaussian_renderer",
            vendor_root / "scene",
            vendor_root / "utils",
        )
        if not path.exists()
    ]
    if missing:
        raise FineFailure("FASTGS_VENDOR_MISSING", f"FastGS vendor source is incomplete: {', '.join(missing)}")


def _optional_float(
    options: dict[str, Any],
    key: str,
    *,
    fallback: float | None,
    minimum: float,
    maximum: float,
) -> float | None:
    if key not in options and fallback is None:
        return None
    return read_float(options.get(key), fallback if fallback is not None else minimum, minimum=minimum, maximum=maximum)


def _optional_int(options: dict[str, Any], key: str, *, fallback: int | None = None, minimum: int, maximum: int) -> int | None:
    if key not in options or options.get(key) in {None, ""}:
        return fallback
    return read_int(options.get(key), fallback if fallback is not None else minimum, minimum=minimum, maximum=maximum)


def _choice_string(value: Any, fallback: str, allowed: set[str]) -> str:
    normalized = str(value if value not in {None, ""} else fallback).strip().lower()
    return normalized if normalized in allowed else fallback


def _bool_string(value: Any, fallback: bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    normalized = str(value if value not in {None, ""} else fallback).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return "true"
    if normalized in {"0", "false", "no", "off"}:
        return "false"
    return "true" if fallback else "false"


def _first_option(options: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in options and options.get(key) not in {None, ""}:
            return options.get(key)
    return None


def _is_progress_line(line: str) -> bool:
    lowered = line.lower()
    return any(token in lowered for token in ("training progress", "[iter", "saving gaussians", "training complete"))


def _find_final_ply(output_dir: Path, iterations: int) -> Path | None:
    exact = output_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if exact.exists() and exact.stat().st_size > 0:
        return exact

    point_cloud_dir = output_dir / "point_cloud"
    if not point_cloud_dir.exists():
        return None
    candidates = []
    for path in point_cloud_dir.glob("iteration_*/point_cloud.ply"):
        match = re.search(r"iteration_(\d+)", str(path.parent.name))
        if path.stat().st_size > 0 and match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]
