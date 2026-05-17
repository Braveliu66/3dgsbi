from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.fine.fastgs_defaults import (
    COLMAP_GUIDED_MATCHING,
    COLMAP_MATCHER,
    COLMAP_MAX_IMAGE_SIZE,
    COLMAP_MIN_REGISTERED_RATIO,
    COLMAP_MIN_SPARSE_POINTS,
    COLMAP_SIFT_EDGE_THRESHOLD,
    COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE,
    COLMAP_SIFT_DOMAIN_SIZE_POOLING,
    COLMAP_SIFT_MATCH_MAX_RATIO,
    COLMAP_SIFT_MAX_NUM_FEATURES,
    COLMAP_SIFT_PEAK_THRESHOLD,
    COLMAP_TARGET_SPARSE_POINTS,
    COLMAP_THREADS,
    DEFAULT_FINE_SCENE_PROFILE,
    FINE_IMAGE_MAX_SIDE,
    FINE_SCENE_PROFILE_MAX_SIDES,
    FASTGS_DEBLUR_BLURRED_VIEWS_ONLY,
    FASTGS_DEBLUR_ENABLED,
    FASTGS_DEBLUR_MODE,
    FASTGS_RESOLUTION,
)
from app.fine.colmap_cli import build_colmap_cli_scene, build_fastgs_chunks, merge_gaussian_ply_chunks
from app.fine.option_utils import read_float, read_int
from app.fine.official_fastgs_big_trainer import OfficialFastGSTrainResult, train_official_fastgs_big
from app.fine.preprocess import build_pycolmap_scene, prepare_fine_images
from app.fine.types import FineContext, FineFailure, FineResult
from app.fine.viewer_meta import read_ply_xyz_bounds, write_far_noise_filtered_ply, write_final_viewer_meta_json, write_scaled_viewer_ply
from app.preview.io.spz import convert_ply_to_spz
from app.preview.types import PreviewFailure
from app.preview.utils import image_files


PIPELINE_NAME = "official_fastgs_big"

SOURCE_COMMITS_FINE = {
    "FastGS": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
    "diff_gaussian_rasterization_fastgs": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
    "simple-knn": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
    "fused-ssim": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
    "pycolmap": "3.12.6",
}

def run_fine_pipeline(ctx: FineContext) -> FineResult:
    settings = get_settings()
    pipeline = normalize_fine_pipeline(ctx.pipeline)
    print(
        "[fine-runner] start "
        f"task_id={ctx.task_id} project_id={ctx.project_id} requested_pipeline={ctx.pipeline} "
        f"normalized_pipeline={pipeline} input_dir={ctx.input_dir} work_dir={ctx.work_dir} "
        f"source_version={ctx.source_version} options={_format_for_log(ctx.options)}",
        flush=True,
    )
    if pipeline != PIPELINE_NAME:
        raise FineFailure("UNSUPPORTED_FINE_PIPELINE", f"Unsupported fine pipeline: {ctx.pipeline}")
    if read_bool(ctx.options.get("fine_edgs_enabled"), False):
        raise FineFailure("UNSUPPORTED_FINE_OPTION", "EDGS/RoMA dense initialization has been removed from this worker image")
    assert_runtime_ready()

    image_count = len(image_files(ctx.input_dir))
    print(
        "[fine-runner] input summary "
        f"input_dir={ctx.input_dir} image_count={image_count} first_images={_first_image_names(ctx.input_dir)}",
        flush=True,
    )
    if image_count < 8:
        raise FineFailure("INSUFFICIENT_IMAGES", "FastGS-Big fine reconstruction requires at least 8 images")

    iterations = read_int(ctx.options.get("fine_iterations"), settings.fine_iterations, minimum=5_000, maximum=60_000)
    fine_scene_profile = resolve_fine_scene_profile(ctx.options)
    profile_image_max_side = FINE_SCENE_PROFILE_MAX_SIDES[fine_scene_profile]
    default_image_max_side = min(profile_image_max_side, settings.fine_image_max_side, FINE_IMAGE_MAX_SIDE)
    image_max_side = read_int(ctx.options.get("fine_image_max_side"), default_image_max_side, minimum=512, maximum=FINE_IMAGE_MAX_SIDE)
    reject_ratio = read_float(ctx.options.get("fine_blur_reject_ratio"), 0.10, minimum=0.0, maximum=0.45)
    colmap_features = read_int(ctx.options.get("fine_sift_max_num_features"), COLMAP_SIFT_MAX_NUM_FEATURES, minimum=1024, maximum=65_536)
    colmap_max_size = read_int(ctx.options.get("fine_colmap_max_image_size"), COLMAP_MAX_IMAGE_SIZE, minimum=512, maximum=4_096)
    colmap_threads = read_int(ctx.options.get("fine_colmap_threads"), COLMAP_THREADS, minimum=1, maximum=32)
    colmap_matcher = str(ctx.options.get("fine_colmap_matcher") or COLMAP_MATCHER).strip().lower()
    colmap_sift_peak_threshold = read_float(ctx.options.get("fine_colmap_sift_peak_threshold"), COLMAP_SIFT_PEAK_THRESHOLD, minimum=0.0001, maximum=0.1)
    colmap_sift_edge_threshold = read_float(ctx.options.get("fine_colmap_sift_edge_threshold"), COLMAP_SIFT_EDGE_THRESHOLD, minimum=1.0, maximum=100.0)
    colmap_estimate_affine_shape = read_bool(ctx.options.get("fine_colmap_estimate_affine_shape"), COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE)
    colmap_domain_size_pooling = read_bool(ctx.options.get("fine_colmap_domain_size_pooling"), COLMAP_SIFT_DOMAIN_SIZE_POOLING)
    colmap_guided_matching = read_bool(ctx.options.get("fine_colmap_guided_matching"), COLMAP_GUIDED_MATCHING)
    colmap_match_max_ratio = read_float(ctx.options.get("fine_colmap_sift_match_max_ratio"), COLMAP_SIFT_MATCH_MAX_RATIO, minimum=0.1, maximum=1.0)
    min_sparse_points = read_int(ctx.options.get("fine_sfm_min_sparse_points"), COLMAP_MIN_SPARSE_POINTS, minimum=0, maximum=1_000_000)
    min_registered_ratio = _optional_float(
        ctx.options,
        "fine_min_registered_ratio",
        fallback=COLMAP_MIN_REGISTERED_RATIO,
        minimum=0.30,
        maximum=0.95,
    )

    ctx_progress(ctx, "fine_blur_analysis", 20, "analyzing image sharpness and filtering lowest quality frames")
    print(
        "[fine-runner] blur analysis start "
        f"reject_ratio={reject_ratio} min_images=3 input_dir={ctx.input_dir}",
        flush=True,
    )
    train_input_dir, blur = prepare_fine_images(
        ctx.input_dir,
        ctx.work_dir / "fine_input",
        reject_ratio=reject_ratio,
        min_images=3,
    )
    print(
        "[fine-runner] blur analysis complete "
        f"train_input_dir={train_input_dir} blur_metrics={_format_for_log(blur.metrics())}",
        flush=True,
    )
    blur_mode = FASTGS_DEBLUR_MODE
    print(
        "[fine-runner] resolved training params "
        f"iterations={iterations} blur_mode={blur_mode} fine_scene_profile={fine_scene_profile} image_max_side={image_max_side} "
        f"sfm_backend={ctx.options.get('fine_sfm_backend') or 'colmap_cli'} colmap_features={colmap_features} "
        f"colmap_max_size={colmap_max_size} colmap_threads={colmap_threads} colmap_matcher={colmap_matcher}",
        flush=True,
    )

    scene_dir = ctx.work_dir / "fine_scene"
    output_dir = ctx.work_dir / "fine_fastgs_big"
    scene_result = build_scene(
        ctx,
        train_input_dir,
        scene_dir,
        colmap_features,
        colmap_max_size,
        colmap_threads,
        matcher=colmap_matcher,
        sift_peak_threshold=colmap_sift_peak_threshold,
        sift_edge_threshold=colmap_sift_edge_threshold,
        estimate_affine_shape=colmap_estimate_affine_shape,
        domain_size_pooling=colmap_domain_size_pooling,
        guided_matching=colmap_guided_matching,
        match_max_ratio=colmap_match_max_ratio,
        min_sparse_points=min_sparse_points,
        min_registered_ratio=min_registered_ratio,
    )
    print(
        "[fine-runner] scene build complete "
        f"backend={scene_result.backend} scene_dir={scene_result.scene_dir} "
        f"image_count={scene_result.image_count} registered_images={scene_result.registered_images} "
        f"point_count={scene_result.point_count} metrics={_format_for_log(scene_result.metrics)}",
        flush=True,
    )

    ctx_progress(ctx, "fine_gaussian_train_start", 42, f"training official FastGS-Big with {scene_result.backend} initialization")
    blur_registry_path = output_dir / "blur_frame_registry.json"
    blur_registry_path.parent.mkdir(parents=True, exist_ok=True)
    blur_registry_path.write_text(
        json.dumps({"frames": blur.per_frame_blur}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    resolved_deblur_mode, deblur_mode_source = resolve_fine_deblur_mode(ctx.options, blur.mode, blur.per_frame_blur)
    train_options = {
        **ctx.options,
        "_fine_scene_backend": scene_result.backend,
        "fine_scene_profile": fine_scene_profile,
        "fine_image_max_side": image_max_side,
        "fine_deblur_mode": resolved_deblur_mode,
        "fine_deblur_mode_source": deblur_mode_source,
        "fine_deblur_blur_registry": str(blur_registry_path),
        "fine_deblur_blurred_views_only": ctx.options.get("fine_deblur_blurred_views_only") or FASTGS_DEBLUR_BLURRED_VIEWS_ONLY,
        "fine_train_resolution": ctx.options.get("fine_train_resolution")
        or scene_result.metrics.get("fastgs_policy_resolution")
        or min(image_max_side, FASTGS_RESOLUTION),
    }
    if "fine_data_device" not in ctx.options and scene_result.metrics.get("fastgs_policy_data_device"):
        train_options["fine_data_device"] = scene_result.metrics["fastgs_policy_data_device"]
    print(
        "[fine-runner] gaussian training start "
        f"scene_dir={scene_result.scene_dir} output_dir={output_dir} iterations={iterations} "
        f"blur_mode={blur_mode} blur_registry={blur_registry_path} train_options={_format_for_log(train_options)}",
        flush=True,
    )

    train_result = train_fastgs_with_optional_chunks(ctx, scene_result, output_dir, iterations, train_options)
    print(
        "[fine-runner] gaussian training complete "
        f"ply_path={train_result.ply_path} metrics={_format_for_log(train_result.metrics)}",
        flush=True,
    )

    ply_path = Path(train_result.ply_path)
    if not ply_path.exists() or ply_path.stat().st_size <= 0:
        raise FineFailure("ARTIFACT_NOT_FOUND", f"Fine runner did not create non-empty PLY: {ply_path}")

    filtered_ply = ctx.work_dir / "final_filtered.ply"
    far_noise_metrics = write_far_noise_filtered_ply(ply_path, filtered_ply, profile=fine_scene_profile)

    ctx.final_ply.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(filtered_ply, ctx.final_ply)
    print(
        "[fine-runner] copied final ply "
        f"source={filtered_ply} target={ctx.final_ply} bytes={ctx.final_ply.stat().st_size} "
        f"far_noise_metrics={_format_for_log(far_noise_metrics)}",
        flush=True,
    )

    viewer_ply = ctx.work_dir / "final_viewer.ply"
    bounds = read_ply_xyz_bounds(ctx.final_ply)
    viewer_scale_multiplier = read_float(ctx.options.get("fine_viewer_scale_multiplier"), 0.55, minimum=0.05, maximum=1.0)
    default_max_scale = max(1e-6, min(float(bounds["radius"]) * 0.009, 1.0))
    viewer_scale_max = read_float(ctx.options.get("fine_viewer_scale_max"), default_max_scale, minimum=1e-6, maximum=10.0)
    viewer_scale_metrics = write_scaled_viewer_ply(
        ctx.final_ply,
        viewer_ply,
        scale_multiplier=viewer_scale_multiplier,
        max_scale=viewer_scale_max,
    )
    print(
        "[fine-runner] viewer ply ready "
        f"source={ctx.final_ply} viewer_ply={viewer_ply} metrics={_format_for_log(viewer_scale_metrics)}",
        flush=True,
    )

    viewer_meta_payload = None
    if ctx.viewer_meta_json:
        viewer_meta_payload = write_final_viewer_meta_json(
            ctx.viewer_meta_json,
            final_ply=ctx.final_ply,
            scene_dir=scene_result.scene_dir,
            preferred_image_names=first_clear_training_images(blur.per_frame_blur),
        )
        print(
            "[fine-runner] viewer meta json written "
            f"path={ctx.viewer_meta_json} bytes={ctx.viewer_meta_json.stat().st_size} "
            f"recommended_view={viewer_meta_payload.get('recommended_view')}",
            flush=True,
        )

    ctx_progress(ctx, "final_spz_converting", 88, "converting viewer-scaled final.ply to Spark-readable final_web.spz")
    try:
        splat_count = convert_ply_to_spz(viewer_ply, ctx.final_spz)
    except PreviewFailure as exc:
        raise FineFailure(exc.code, exc.message) from exc
    print(
        "[fine-runner] final SPZ conversion complete "
        f"final_ply={ctx.final_ply} viewer_ply={viewer_ply} final_ply_bytes={ctx.final_ply.stat().st_size} "
        f"final_spz={ctx.final_spz} final_spz_bytes={ctx.final_spz.stat().st_size if ctx.final_spz.exists() else None} "
        f"splat_count={splat_count}",
        flush=True,
    )

    warnings = []
    lod_rad = build_lod_rad_if_available(ctx)
    if ctx.lod_rad and lod_rad is None:
        warnings.append("RAD LOD builder is not configured; final_lod.rad was not generated.")

    effective_algorithms = [scene_result.backend, "official_fastgs_big", "diff_gaussian_rasterization_fastgs"]
    effective_deblur_mode = str(train_options.get("fine_deblur_mode") or FASTGS_DEBLUR_MODE)
    if deblur_mlp_enabled_by_default(effective_deblur_mode, train_options):
        effective_algorithms.append("Deblurring-3DGS_GTnet_fastgs")
    metrics = {
        "pipeline": PIPELINE_NAME,
        "algorithm": f"{scene_result.backend}_official_fastgs_big",
        "requested_algorithms": [scene_result.backend, "FastGS-Big"],
        "effective_algorithms": effective_algorithms,
        "source_version": ctx.source_version,
        "source_commits": SOURCE_COMMITS_FINE,
        "artifact_converter": "Spark SPZ",
        "fine_input_type": ctx.options.get("input_type") or ("video" if ctx.input_video else "images"),
        "input_images": image_count,
        "training_images": len(image_files(train_input_dir)),
        "fine_scene_profile": fine_scene_profile,
        "fine_image_max_side": image_max_side,
        "iterations": iterations,
        "deblur_mode_source": deblur_mode_source,
        "splat_count": splat_count,
        "final_ply_bytes": ctx.final_ply.stat().st_size,
        "final_viewer_ply_bytes": viewer_ply.stat().st_size,
        "final_spz_bytes": ctx.final_spz.stat().st_size,
        "viewer_meta_json_bytes": ctx.viewer_meta_json.stat().st_size if ctx.viewer_meta_json and ctx.viewer_meta_json.exists() else None,
        "lod_rad_bytes": lod_rad.stat().st_size if lod_rad else None,
        "warnings": warnings,
        **far_noise_metrics,
        **viewer_scale_metrics,
        **blur.metrics(),
        **scene_result.metrics,
        **train_result.metrics,
    }
    ctx.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    ctx.metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[fine-runner] metrics json written "
        f"path={ctx.metrics_json} bytes={ctx.metrics_json.stat().st_size} metrics={_format_for_log(metrics)}",
        flush=True,
    )
    ctx_progress(ctx, "fine_outputs_ready", 90, "validated final.ply and final_web.spz", metrics)
    return FineResult(
        final_ply=ctx.final_ply,
        final_spz=ctx.final_spz,
        metrics_json=ctx.metrics_json,
        viewer_meta_json=ctx.viewer_meta_json if ctx.viewer_meta_json and ctx.viewer_meta_json.exists() else None,
        lod_rad=lod_rad,
        splat_count=splat_count,
        source_commits=SOURCE_COMMITS_FINE,
        metrics=metrics,
    )


def train_fastgs_with_optional_chunks(
    ctx: FineContext,
    scene_result,
    output_dir: Path,
    iterations: int,
    train_options: dict[str, Any],
) -> OfficialFastGSTrainResult:
    chunking_enabled = scene_result.backend == "colmap_cli" and read_bool(train_options.get("fastgs_target"), True)
    if not chunking_enabled:
        result = train_fastgs_with_oom_fallback(
            scene_result.scene_dir,
            output_dir,
            iterations,
            train_options,
            progress=lambda stage, progress, message: ctx_progress(ctx, stage, progress, message),
        )
        result.metrics.update({"fastgs_chunk_count": 1, "fastgs_chunk_merge_policy": "none"})
        return result

    target_images = read_int(
        scene_result.metrics.get("fastgs_policy_chunk_target_images"),
        max(1, scene_result.registered_images or scene_result.image_count),
        minimum=1,
        maximum=10_000,
    )
    overlap_ratio = read_float(
        scene_result.metrics.get("fastgs_policy_overlap_ratio"),
        0.25,
        minimum=0.0,
        maximum=0.60,
    )
    chunk_result = build_fastgs_chunks(
        scene_result.scene_dir,
        ctx.work_dir / "fine_fastgs_chunks",
        scene_type=fastgs_chunk_scene_type(train_options),
        n_images=scene_result.registered_images or scene_result.image_count,
        target_images=target_images,
        overlap_ratio=overlap_ratio,
        progress=lambda stage, progress, message: ctx_progress(ctx, stage, progress, message),
    )
    chunk_dirs = chunk_result.chunk_scene_dirs
    if len(chunk_dirs) <= 1:
        results, split_metrics = train_fastgs_chunk_or_split(
            ctx,
            chunk_dirs[0],
            output_dir,
            iterations,
            train_options,
            chunk_index=0,
            scene_type=fastgs_chunk_scene_type(train_options),
        )
        if len(results) > 1:
            merged_ply = output_dir / "final_raw.ply"
            merged_vertices = merge_gaussian_ply_chunks([Path(item.ply_path) for item in results], merged_ply)
            return OfficialFastGSTrainResult(
                ply_path=merged_ply,
                iterations=iterations,
                metrics={
                    **chunk_result.metrics,
                    **split_metrics,
                    "fastgs_chunk_count": len(results),
                    "fastgs_chunk_merge_policy": "ply_vertex_concat",
                    "fastgs_merged_vertex_count": merged_vertices,
                    "fastgs_merged_ply": str(merged_ply),
                },
            )
        result = results[0]
        result.metrics.update({**chunk_result.metrics, "fastgs_chunk_merge_policy": "none"})
        return result

    chunk_train_root = output_dir / "chunks"
    chunk_plys: list[Path] = []
    chunk_metrics: dict[str, Any] = dict(chunk_result.metrics)
    for index, chunk_scene_dir in enumerate(chunk_dirs):
        ctx_progress(
            ctx,
            "fine_fastgs_chunk_training",
            44,
            f"training FastGS chunk {index + 1}/{len(chunk_dirs)}",
        )
        results, split_metrics = train_fastgs_chunk_or_split(
            ctx,
            chunk_scene_dir,
            chunk_train_root / f"chunk_{index:03d}",
            iterations,
            train_options,
            chunk_index=index,
            scene_type=fastgs_chunk_scene_type(train_options),
        )
        chunk_metrics.update(split_metrics)
        for result_index, result in enumerate(results):
            chunk_plys.append(Path(result.ply_path))
            metric_key = f"fastgs_chunk_{index:03d}" if len(results) == 1 else f"fastgs_chunk_{index:03d}_{result_index:02d}"
            chunk_metrics[f"{metric_key}_ply"] = str(result.ply_path)
            chunk_metrics[f"{metric_key}_metrics"] = result.metrics

    merged_ply = output_dir / "final_raw.ply"
    merged_vertices = merge_gaussian_ply_chunks(chunk_plys, merged_ply)
    chunk_metrics.update(
        {
            "fastgs_chunk_count": len(chunk_plys),
            "fastgs_chunk_merge_policy": "ply_vertex_concat",
            "fastgs_merged_vertex_count": merged_vertices,
            "fastgs_merged_ply": str(merged_ply),
        }
    )
    return OfficialFastGSTrainResult(ply_path=merged_ply, iterations=iterations, metrics=chunk_metrics)


def train_fastgs_chunk_or_split(
    ctx: FineContext,
    scene_dir: Path,
    output_dir: Path,
    iterations: int,
    options: dict[str, Any],
    *,
    chunk_index: int,
    scene_type: str,
    depth: int = 0,
) -> tuple[list[OfficialFastGSTrainResult], dict[str, Any]]:
    try:
        result = train_fastgs_with_oom_fallback(
            scene_dir,
            output_dir,
            iterations,
            options,
            progress=lambda stage, progress, message: ctx_progress(ctx, stage, progress, message),
        )
        return [result], {}
    except FineFailure as exc:
        if exc.code != "FASTGS_TRAIN_FAILED" or not is_fastgs_oom_message(exc.message) or depth >= 1:
            raise
        n_images = len(image_files(scene_dir / "images"))
        if n_images < 16:
            raise
        split_root = output_dir.parent / f"{output_dir.name}_oom_split"
        split_result = build_fastgs_chunks(
            scene_dir,
            split_root,
            scene_type=scene_type,
            n_images=n_images,
            target_images=max(1, math.ceil(n_images / 2)),
            overlap_ratio=0.15 if scene_type == "outdoor" else 0.25,
            progress=lambda stage, progress, message: ctx_progress(ctx, stage, progress, message),
        )
        if len(split_result.chunk_scene_dirs) <= 1:
            raise
        results: list[OfficialFastGSTrainResult] = []
        metrics: dict[str, Any] = {
            f"fastgs_chunk_{chunk_index:03d}_oom_split_count": len(split_result.chunk_scene_dirs),
            f"fastgs_chunk_{chunk_index:03d}_oom_split_root": str(split_root),
        }
        for subindex, sub_scene_dir in enumerate(split_result.chunk_scene_dirs):
            sub_results, sub_metrics = train_fastgs_chunk_or_split(
                ctx,
                sub_scene_dir,
                output_dir.parent / f"{output_dir.name}_split_{subindex:02d}",
                iterations,
                options,
                chunk_index=chunk_index,
                scene_type=scene_type,
                depth=depth + 1,
            )
            results.extend(sub_results)
            metrics.update(sub_metrics)
        return results, metrics


def train_fastgs_with_oom_fallback(
    scene_dir: Path,
    output_dir: Path,
    iterations: int,
    options: dict[str, Any],
    *,
    progress,
) -> OfficialFastGSTrainResult:
    attempts = fastgs_oom_attempt_options(options)
    last_failure: FineFailure | None = None
    for index, attempt_options in enumerate(attempts, start=1):
        try:
            result = train_official_fastgs_big(
                scene_dir=scene_dir,
                output_dir=output_dir,
                iterations=iterations,
                options=attempt_options,
                progress=progress,
            )
            result.metrics["fastgs_oom_retry_count"] = index - 1
            return result
        except FineFailure as exc:
            if exc.code != "FASTGS_TRAIN_FAILED" or not is_fastgs_oom_message(exc.message):
                raise
            last_failure = exc
            print(
                "[fine-runner] FastGS chunk training hit OOM; retrying with lower memory options "
                f"attempt={index} scene_dir={scene_dir} output_dir={output_dir}",
                flush=True,
            )
            if output_dir.exists():
                shutil.rmtree(output_dir)
    assert last_failure is not None
    raise last_failure


def fastgs_oom_attempt_options(options: dict[str, Any]) -> list[dict[str, Any]]:
    base = dict(options)
    attempts = [base]
    cpu_options = {**base, "fine_data_device": "cpu"}
    if str(base.get("fine_data_device") or "").strip().lower() != "cpu":
        attempts.append(cpu_options)
    current_resolution = read_int(base.get("fine_train_resolution"), FASTGS_RESOLUTION, minimum=1, maximum=16_384)
    for resolution in (1440, 1280, 960):
        if current_resolution > resolution:
            attempts.append({**cpu_options, "fine_train_resolution": resolution})
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in attempts:
        key = (
            str(item.get("fine_data_device") or ""),
            read_int(item.get("fine_train_resolution"), current_resolution, minimum=1, maximum=16_384),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def is_fastgs_oom_message(message: str) -> bool:
    lowered = str(message or "").lower()
    return any(token in lowered for token in ("out of memory", "cuda", "cublas", "cudnn", "alloc"))


def fastgs_chunk_scene_type(options: dict[str, Any]) -> str:
    value = str(options.get("scene_type") or options.get("fine_scene_type") or options.get("fine_scene_profile") or "").strip().lower()
    return "outdoor" if value in {"outdoor", "outdoor_full", "outdoor_fast_clean"} else "indoor"


def build_scene(
    ctx: FineContext,
    input_dir: Path,
    scene_dir: Path,
    colmap_features: int,
    colmap_max_size: int,
    colmap_threads: int,
    matcher: str = "auto",
    sift_peak_threshold: float | None = None,
    sift_edge_threshold: float | None = None,
    estimate_affine_shape: bool = False,
    domain_size_pooling: bool = False,
    guided_matching: bool = False,
    match_max_ratio: float | None = None,
    min_sparse_points: int = COLMAP_MIN_SPARSE_POINTS,
    min_registered_ratio: float | None = None,
):
    sfm_backend = str(ctx.options.get("fine_sfm_backend") or "colmap_cli").strip().lower()
    print(
        "[fine-runner] scene build start "
        f"requested_sfm_backend={sfm_backend} input_dir={input_dir} scene_dir={scene_dir} "
        f"colmap_features={colmap_features} colmap_max_size={colmap_max_size} colmap_threads={colmap_threads} "
        f"matcher={matcher} min_sparse_points={min_sparse_points}",
        flush=True,
    )
    if sfm_backend not in {"pycolmap", "colmap", "colmap_cli"}:
        raise FineFailure("UNSUPPORTED_FINE_SFM_BACKEND", f"Unsupported fine SfM backend: {sfm_backend}")
    if sfm_backend in {"colmap", "colmap_cli"}:
        input_type = str(ctx.options.get("input_type") or ("video" if getattr(ctx, "input_video", None) else "images")).strip().lower()
        result = build_colmap_cli_scene(
            input_dir,
            scene_dir,
            scene_type=str(ctx.options.get("fine_scene_type") or ctx.options.get("scene_type") or "indoor"),
            input_type=input_type,
            quality_mode=str(ctx.options.get("quality_mode") or "auto"),
            capture_order=str(ctx.options.get("fine_capture_order") or "auto"),
            prefer_gpu=read_bool(ctx.options.get("prefer_gpu"), True),
            gpu_index=str(ctx.options.get("fine_colmap_gpu_index") or "").strip() or None,
            min_registered_ratio=min_registered_ratio,
            progress=lambda stage, progress, message: ctx_progress(ctx, stage, progress, message),
        )
        result.metrics["sfm_min_sparse_points"] = min_sparse_points
        result.metrics["sfm_target_sparse_points"] = COLMAP_TARGET_SPARSE_POINTS
        point_count = int(result.point_count or 0)
        if min_sparse_points > 0 and point_count < min_sparse_points:
            raise FineFailure(
                "SFM_SPARSE_POINTS_TOO_LOW",
                f"COLMAP CLI produced {point_count} sparse points, below quality gate {min_sparse_points}",
            )
        if sfm_backend == "colmap":
            result.metrics["sfm_backend_requested_alias"] = "colmap_maps_to_colmap_cli"
        return result

    retryable_sfm_codes = {"COLMAP_RECONSTRUCTION_FAILED", "COLMAP_RECONSTRUCTION_INCOMPLETE"}
    try:
        result = build_pycolmap_scene(
            input_dir,
            scene_dir,
            max_num_features=colmap_features,
            max_image_size=colmap_max_size,
            min_model_size=max(3, min(10, len(image_files(input_dir)))),
            num_threads=colmap_threads,
            matcher=matcher,
            sift_peak_threshold=sift_peak_threshold,
            sift_edge_threshold=sift_edge_threshold,
            estimate_affine_shape=estimate_affine_shape,
            domain_size_pooling=domain_size_pooling,
            guided_matching=guided_matching,
            match_max_ratio=match_max_ratio,
            profile_name="primary",
            min_registered_ratio=min_registered_ratio,
            progress=lambda stage, progress, message: ctx_progress(ctx, stage, progress, message),
        )
        point_count = int(result.point_count or 0)
        if min_sparse_points > 0 and point_count < min_sparse_points:
            print(
                "[fine-runner] sparse SfM below gate; retrying high-recall COLMAP "
                f"points={point_count} min_sparse_points={min_sparse_points}",
                flush=True,
            )
            result = build_pycolmap_scene(
                input_dir,
                scene_dir,
                max_num_features=max(colmap_features, COLMAP_SIFT_MAX_NUM_FEATURES),
                max_image_size=max(colmap_max_size, COLMAP_MAX_IMAGE_SIZE),
                min_model_size=max(3, min(10, len(image_files(input_dir)))),
                num_threads=colmap_threads,
                matcher="exhaustive",
                sift_peak_threshold=COLMAP_SIFT_PEAK_THRESHOLD,
                sift_edge_threshold=COLMAP_SIFT_EDGE_THRESHOLD,
                estimate_affine_shape=COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE,
                domain_size_pooling=COLMAP_SIFT_DOMAIN_SIZE_POOLING,
                guided_matching=COLMAP_GUIDED_MATCHING,
                match_max_ratio=COLMAP_SIFT_MATCH_MAX_RATIO,
                profile_name="high_recall",
                min_registered_ratio=min_registered_ratio,
                progress=lambda stage, progress, message: ctx_progress(ctx, stage, progress, message),
            )
            result.metrics["sfm_high_recall_retry"] = True
            point_count = int(result.point_count or 0)
        else:
            result.metrics["sfm_high_recall_retry"] = False
    except FineFailure as exc:
        if exc.code not in retryable_sfm_codes:
            raise
        print(
            "[fine-runner] initial COLMAP pass failed; retrying high-recall COLMAP "
            f"code={exc.code}",
            flush=True,
        )
        result = build_pycolmap_scene(
            input_dir,
            scene_dir,
            max_num_features=max(colmap_features, COLMAP_SIFT_MAX_NUM_FEATURES),
            max_image_size=max(colmap_max_size, COLMAP_MAX_IMAGE_SIZE),
            min_model_size=max(3, min(10, len(image_files(input_dir)))),
            num_threads=colmap_threads,
            matcher="exhaustive",
            sift_peak_threshold=COLMAP_SIFT_PEAK_THRESHOLD,
            sift_edge_threshold=COLMAP_SIFT_EDGE_THRESHOLD,
            estimate_affine_shape=COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE,
            domain_size_pooling=COLMAP_SIFT_DOMAIN_SIZE_POOLING,
            guided_matching=COLMAP_GUIDED_MATCHING,
            match_max_ratio=COLMAP_SIFT_MATCH_MAX_RATIO,
            profile_name="high_recall",
            min_registered_ratio=min_registered_ratio,
            progress=lambda stage, progress, message: ctx_progress(ctx, stage, progress, message),
        )
        result.metrics["sfm_high_recall_retry"] = True
        point_count = int(result.point_count or 0)
    result.metrics["sfm_min_sparse_points"] = min_sparse_points
    result.metrics["sfm_target_sparse_points"] = COLMAP_TARGET_SPARSE_POINTS
    if min_sparse_points > 0 and point_count < min_sparse_points:
        raise FineFailure(
            "SFM_SPARSE_POINTS_TOO_LOW",
            f"COLMAP produced {point_count} sparse points, below quality gate {min_sparse_points}",
        )
    if sfm_backend == "colmap":
        result.metrics["sfm_backend_requested_alias"] = "colmap_maps_to_pycolmap"
    return result


def normalize_fine_pipeline(value: str | None) -> str:
    normalized = (value or PIPELINE_NAME).strip().lower()
    aliases = {
        PIPELINE_NAME: PIPELINE_NAME,
    }
    return aliases.get(normalized, normalized)


def resolve_fine_scene_profile(options: dict[str, Any]) -> str:
    profile = str(
        options.get("fine_scene_profile") or options.get("preview_scene_profile") or DEFAULT_FINE_SCENE_PROFILE
    ).strip().lower()
    if profile not in FINE_SCENE_PROFILE_MAX_SIDES:
        return DEFAULT_FINE_SCENE_PROFILE
    return profile


def assert_runtime_ready() -> None:
    print("[fine-runner] checking CUDA and official FastGS-Big modules", flush=True)
    try:
        import torch
    except Exception as exc:
        raise FineFailure("TORCH_UNAVAILABLE", f"PyTorch import failed: {exc}") from exc
    if not torch.cuda.is_available():
        raise FineFailure("GPU_RESOURCE_UNAVAILABLE", "CUDA GPU is required for fine reconstruction")
    print(
        "[fine-runner] torch CUDA ready "
        f"torch_version={getattr(torch, '__version__', 'unknown')} cuda_version={getattr(torch.version, 'cuda', None)} "
        f"device_count={torch.cuda.device_count()} current_device={torch.cuda.current_device() if torch.cuda.device_count() else None} "
        f"device_name={torch.cuda.get_device_name(0) if torch.cuda.device_count() else None}",
        flush=True,
    )
    missing = []
    for module_name in ("diff_gaussian_rasterization_fastgs", "simple_knn", "fused_ssim"):
        try:
            __import__(module_name)
            print(f"[fine-runner] module import ok module={module_name}", flush=True)
        except Exception as exc:
            missing.append(f"{module_name}: {exc}")
    if missing:
        raise FineFailure("FINE_RUNTIME_UNAVAILABLE", f"Fine reconstruction runtime dependency missing: {'; '.join(missing)}")
    fastgs_rasterizer = __import__("diff_gaussian_rasterization_fastgs")
    if not hasattr(fastgs_rasterizer, "GaussianRasterizer"):
        raise FineFailure("FINE_RUNTIME_UNAVAILABLE", "diff_gaussian_rasterization_fastgs is missing GaussianRasterizer")


def deblur_mlp_enabled_by_default(blur_mode: str, options: dict[str, Any]) -> bool:
    value = str(options.get("fine_deblur_enabled", FASTGS_DEBLUR_ENABLED)).lower()
    if value in {"0", "false", "no", "off"}:
        return False
    return str(blur_mode or FASTGS_DEBLUR_MODE).strip().lower() != "sharp"


def resolve_fine_deblur_mode(
    options: dict[str, Any],
    _detected_mode: str,
    _per_frame_blur: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    explicit = str(options.get("fine_deblur_mode") or "").strip().lower()
    if explicit in {"sharp", "defocus", "motion", "mixed"}:
        return explicit, "override"
    return FASTGS_DEBLUR_MODE, "default_mixed"


def first_clear_training_images(per_frame_blur: dict[str, dict[str, Any]]) -> list[str]:
    clear: list[str] = []
    for item in per_frame_blur.values():
        if item.get("rejected") or item.get("blurred"):
            continue
        training_image = item.get("training_image")
        if isinstance(training_image, str) and training_image:
            clear.append(training_image)
    return clear


def read_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return fallback


def _optional_float(options: dict[str, Any], key: str, *, fallback: float | None = None, minimum: float, maximum: float) -> float | None:
    if key not in options or options.get(key) in {None, ""}:
        return fallback
    return read_float(options.get(key), fallback if fallback is not None else minimum, minimum=minimum, maximum=maximum)


def build_lod_rad_if_available(ctx: FineContext) -> Path | None:
    if ctx.lod_rad is None:
        return None
    builder = os.getenv("SPARK_RAD_CLI")
    if not builder:
        return None
    command = ["node", builder, "convert", str(ctx.final_ply), str(ctx.lod_rad)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if completed.returncode != 0:
        raise FineFailure("RAD_CONVERSION_FAILED", (completed.stderr or completed.stdout or "").strip() or "RAD converter failed")
    if not ctx.lod_rad.exists() or ctx.lod_rad.stat().st_size <= 0:
        raise FineFailure("RAD_NOT_FOUND", f"RAD converter did not create non-empty output: {ctx.lod_rad}")
    return ctx.lod_rad


def ctx_progress(ctx: FineContext, stage: str, progress: int, message: str | None = None, metrics: dict[str, Any] | None = None) -> None:
    if ctx.progress:
        ctx.progress(stage, progress, message, metrics)


def _first_image_names(input_dir: Path, limit: int = 8) -> str:
    files = image_files(input_dir)
    names = [path.name for path in files[:limit]]
    suffix = "" if len(files) <= limit else f", ... +{len(files) - limit}"
    return "[" + ", ".join(names) + suffix + "]"


def _format_for_log(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)
