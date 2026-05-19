from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_settings, resolve_local_path
from app.fine.colmap_cli import build_colmap_cli_scene
from app.fine.colmap_defaults import (
    COLMAP_GUIDED_MATCHING,
    COLMAP_MATCHER,
    COLMAP_MAX_IMAGE_SIZE,
    COLMAP_MAX_NUM_MATCHES,
    COLMAP_MIN_REGISTERED_RATIO,
    COLMAP_MIN_SPARSE_POINTS,
    COLMAP_SEQUENTIAL_OVERLAP,
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
    FINE_PIPELINE_NAME,
    FINE_SCENE_PROFILE_MAX_SIDES,
    LEGACY_FINE_PIPELINE_ALIASES,
)
from app.fine.dash_deblur_group import resolve_runtime_paths, run_dash_deblur_group_training
from app.fine.option_utils import read_float, read_int
from app.fine.preprocess import build_pycolmap_scene, prepare_fine_images
from app.fine.types import FineContext, FineFailure, FineResult
from app.preview.utils import image_files


PIPELINE_NAME = FINE_PIPELINE_NAME

SOURCE_COMMITS_FINE = {
    "COLMAP": "system",
    "pycolmap": "3.12.6",
    "DashDeblurGroupGS": "runtime",
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
    reject_removed_options(ctx.options)
    assert_runtime_ready()
    assert_training_runtime_ready(ctx.options, resolve_local_path(settings.repo_cache_dir))

    image_count = len(image_files(ctx.input_dir))
    if image_count < 3:
        raise FineFailure("INSUFFICIENT_IMAGES", "COLMAP fine reconstruction requires at least 3 images")

    fine_scene_profile = resolve_fine_scene_profile(ctx.options)
    profile_image_max_side = FINE_SCENE_PROFILE_MAX_SIDES[fine_scene_profile]
    default_image_max_side = min(profile_image_max_side, settings.fine_image_max_side, FINE_IMAGE_MAX_SIDE)
    image_max_side = read_int(ctx.options.get("fine_image_max_side"), default_image_max_side, minimum=512, maximum=FINE_IMAGE_MAX_SIDE)
    reject_ratio = read_float(ctx.options.get("fine_blur_reject_ratio"), 0.0, minimum=0.0, maximum=0.45)
    colmap_max_size = read_int(ctx.options.get("fine_colmap_max_image_size"), COLMAP_MAX_IMAGE_SIZE, minimum=512, maximum=4_096)
    colmap_max_matches = read_int(ctx.options.get("fine_colmap_max_num_matches"), COLMAP_MAX_NUM_MATCHES, minimum=1024, maximum=65_536)
    colmap_sequential_overlap = read_int(ctx.options.get("fine_colmap_sequential_overlap"), COLMAP_SEQUENTIAL_OVERLAP, minimum=4, maximum=200)
    colmap_threads = read_int(ctx.options.get("fine_colmap_threads"), COLMAP_THREADS, minimum=1, maximum=32)
    colmap_matcher = str(ctx.options.get("fine_colmap_matcher") or COLMAP_MATCHER).strip().lower()
    min_sparse_points = read_int(ctx.options.get("fine_sfm_min_sparse_points"), COLMAP_MIN_SPARSE_POINTS, minimum=0, maximum=1_000_000)
    min_registered_ratio = _optional_float(
        ctx.options,
        "fine_min_registered_ratio",
        fallback=COLMAP_MIN_REGISTERED_RATIO,
        minimum=0.30,
        maximum=0.95,
    )

    ctx_progress(ctx, "fine_image_preparing", 20, "normalizing images and filtering low quality frames")
    train_input_dir, quality = prepare_fine_images(
        ctx.input_dir,
        ctx.work_dir / "fine_input",
        reject_ratio=reject_ratio,
        min_images=3,
    )
    colmap_features, colmap_feature_metrics = resolve_colmap_feature_budget(
        ctx.options,
        quality,
        fine_scene_profile=fine_scene_profile,
        image_count=len(image_files(train_input_dir)),
        matcher=colmap_matcher,
        max_image_size=colmap_max_size,
    )
    print(
        "[fine-runner] colmap feature budget "
        f"max_num_features={colmap_features} "
        f"requested={colmap_feature_metrics['colmap_sift_feature_budget_requested']} "
        f"reason={colmap_feature_metrics['colmap_sift_feature_budget_reason']} "
        f"blur_ratio={colmap_feature_metrics['colmap_sift_feature_budget_blur_ratio']} "
        f"sharp_score={colmap_feature_metrics['colmap_sift_feature_budget_sharp_score']} "
        f"max_image_size={colmap_max_size}",
        flush=True,
    )

    scene_dir = ctx.work_dir / "fine_scene"
    scene_result = build_scene(
        ctx,
        train_input_dir,
        scene_dir,
        colmap_features,
        colmap_max_size,
        colmap_threads,
        colmap_max_matches=colmap_max_matches,
        colmap_sequential_overlap=colmap_sequential_overlap,
        matcher=colmap_matcher,
        sift_peak_threshold=read_float(ctx.options.get("fine_colmap_sift_peak_threshold"), COLMAP_SIFT_PEAK_THRESHOLD, minimum=0.0001, maximum=0.1),
        sift_edge_threshold=read_float(ctx.options.get("fine_colmap_sift_edge_threshold"), COLMAP_SIFT_EDGE_THRESHOLD, minimum=1.0, maximum=100.0),
        estimate_affine_shape=read_bool(ctx.options.get("fine_colmap_estimate_affine_shape"), COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE),
        domain_size_pooling=read_bool(ctx.options.get("fine_colmap_domain_size_pooling"), COLMAP_SIFT_DOMAIN_SIZE_POOLING),
        guided_matching=read_bool(ctx.options.get("fine_colmap_guided_matching"), COLMAP_GUIDED_MATCHING),
        match_max_ratio=read_float(ctx.options.get("fine_colmap_sift_match_max_ratio"), COLMAP_SIFT_MATCH_MAX_RATIO, minimum=0.1, maximum=1.0),
        min_sparse_points=min_sparse_points,
        min_registered_ratio=min_registered_ratio,
    )
    scene_result.metrics.update(colmap_feature_metrics)
    remap_blur_registry_to_scene_images(quality, train_input_dir, scene_result.scene_dir)
    print(
        "[fine-runner] scene build complete "
        f"backend={scene_result.backend} scene_dir={scene_result.scene_dir} "
        f"registered_images={scene_result.registered_images} point_count={scene_result.point_count}",
        flush=True,
    )

    ctx_progress(ctx, "fine_training_start", 42, "starting DashDeblurGroupGS from COLMAP scene")
    training_result = run_dash_deblur_group_training(
        scene_dir=scene_result.scene_dir,
        work_dir=ctx.work_dir,
        final_ply=ctx.final_ply,
        final_spz=ctx.final_spz,
        options=ctx.options,
        repo_cache_dir=resolve_local_path(settings.repo_cache_dir),
        blur_analysis=quality,
        progress=lambda stage, progress, message: ctx_progress(ctx, stage, progress, message),
    )

    ctx_progress(ctx, "fine_outputs_validating", 82, "validating fine Gaussian outputs")
    from app.fine.viewer_meta import read_ply_xyz_bounds, write_final_viewer_meta_json

    bounds = read_ply_xyz_bounds(ctx.final_ply)

    viewer_meta_payload = None
    if ctx.viewer_meta_json:
        viewer_meta_payload = write_final_viewer_meta_json(
            ctx.viewer_meta_json,
            final_ply=ctx.final_ply,
            scene_dir=scene_result.scene_dir,
            asset_type="fine_dash_deblur_group_gaussians",
            point_source="dash_deblur_group_gs",
            default_view_mode="splats",
        )

    source_commits = {
        **SOURCE_COMMITS_FINE,
        "DashDeblurGroupGS": training_result.source_commit,
    }
    metrics = {
        "pipeline": PIPELINE_NAME,
        "algorithm": "dash_deblur_group_gs",
        "requested_algorithms": [scene_result.backend, "dash_deblur_group_gs"],
        "effective_algorithms": [scene_result.backend, "dash_deblur_group_gs"],
        "source_version": ctx.source_version,
        "source_commits": source_commits,
        "fine_input_type": ctx.options.get("input_type") or ("video" if ctx.input_video else "images"),
        "input_images": image_count,
        "training_images": len(image_files(train_input_dir)),
        "fine_scene_profile": fine_scene_profile,
        "fine_image_max_side": image_max_side,
        "final_ply_bytes": ctx.final_ply.stat().st_size,
        "final_spz_bytes": training_result.final_spz.stat().st_size if training_result.final_spz else None,
        "viewer_meta_json_bytes": ctx.viewer_meta_json.stat().st_size if ctx.viewer_meta_json and ctx.viewer_meta_json.exists() else None,
        "bbox_min": bounds["bbox_min"],
        "bbox_max": bounds["bbox_max"],
        "bbox_radius": bounds["radius"],
        **quality.metrics(),
        **scene_result.metrics,
        **training_result.metrics,
    }
    if viewer_meta_payload is not None:
        metrics["viewer_meta_asset_type"] = viewer_meta_payload.get("asset_type")
    ctx.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    ctx.metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    ctx_progress(ctx, "fine_outputs_ready", 90, "validated DashDeblurGroupGS outputs", metrics)
    return FineResult(
        final_ply=ctx.final_ply,
        final_spz=training_result.final_spz,
        metrics_json=ctx.metrics_json,
        viewer_meta_json=ctx.viewer_meta_json if ctx.viewer_meta_json and ctx.viewer_meta_json.exists() else None,
        lod_rad=None,
        splat_count=training_result.splat_count,
        source_commits=source_commits,
        metrics=metrics,
    )


def build_scene(
    ctx: FineContext,
    input_dir: Path,
    scene_dir: Path,
    colmap_features: int,
    colmap_max_size: int,
    colmap_threads: int,
    matcher: str = "auto",
    colmap_max_matches: int = COLMAP_MAX_NUM_MATCHES,
    colmap_sequential_overlap: int = COLMAP_SEQUENTIAL_OVERLAP,
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
    if sfm_backend not in {"pycolmap", "colmap", "colmap_cli"}:
        raise FineFailure("UNSUPPORTED_FINE_SFM_BACKEND", f"Unsupported fine SfM backend: {sfm_backend}")
    if sfm_backend in {"colmap", "colmap_cli"}:
        result = build_colmap_cli_scene(
            input_dir,
            scene_dir,
            scene_type=str(ctx.options.get("fine_scene_type") or ctx.options.get("scene_type") or "indoor"),
            input_type=str(ctx.options.get("input_type") or ("video" if getattr(ctx, "input_video", None) else "images")),
            quality_mode=str(ctx.options.get("quality_mode") or "auto"),
            capture_order=str(ctx.options.get("fine_capture_order") or "auto"),
            matcher_policy=matcher,
            prefer_gpu=read_bool(ctx.options.get("prefer_gpu"), True),
            gpu_index=str(ctx.options.get("fine_colmap_gpu_index") or "").strip() or None,
            min_registered_ratio=min_registered_ratio,
            max_num_features=colmap_features,
            max_image_size=colmap_max_size,
            max_num_matches=colmap_max_matches,
            sequential_overlap=colmap_sequential_overlap,
            progress=lambda stage, progress, message: ctx_progress(ctx, stage, progress, message),
        )
    else:
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

    result.metrics["sfm_min_sparse_points"] = min_sparse_points
    result.metrics["sfm_target_sparse_points"] = COLMAP_TARGET_SPARSE_POINTS
    point_count = int(result.point_count or 0)
    if point_count < COLMAP_TARGET_SPARSE_POINTS:
        result.metrics["sfm_sparse_points_below_target"] = True
    return result


def normalize_fine_pipeline(value: str | None) -> str:
    normalized = (value or PIPELINE_NAME).strip().lower()
    if normalized in LEGACY_FINE_PIPELINE_ALIASES:
        return PIPELINE_NAME
    return normalized


def remap_blur_registry_to_scene_images(quality: Any, train_input_dir: Path, scene_dir: Path) -> None:
    registry = getattr(quality, "per_frame_blur", None)
    if not isinstance(registry, dict) or not registry:
        return
    source_images = image_files(train_input_dir)
    scene_images_dir = scene_dir / "images"
    if not scene_images_dir.exists():
        return
    scene_images = image_files(scene_images_dir)
    if len(source_images) != len(scene_images):
        print(
            "[fine-runner] blur label remap skipped "
            f"source_images={len(source_images)} scene_images={len(scene_images)}",
            flush=True,
        )
        return

    source_by_name = {path.name: path for path in source_images}
    source_by_stem = {path.stem: path for path in source_images}
    source_to_scene: dict[str, str] = {}
    unused_scene = set(scene_images)
    for source in source_images:
        matched = None
        for scene_image in list(unused_scene):
            if scene_image.name == source.name or scene_image.stem == source.stem or scene_image.stem.endswith("_" + source.stem):
                matched = scene_image
                break
        if matched is not None:
            unused_scene.remove(matched)
            source_to_scene[source.name] = matched.name

    if len(source_to_scene) != len(source_images):
        source_to_scene = {source.name: scene.name for source, scene in zip(source_images, scene_images)}

    remapped: dict[str, dict[str, Any]] = {}
    for key, item in registry.items():
        if not isinstance(item, dict) or item.get("rejected"):
            remapped[key] = item
            continue
        training_image = item.get("training_image") or key
        source_path = source_by_name.get(str(training_image)) or source_by_stem.get(Path(str(training_image)).stem)
        scene_name = source_to_scene.get(source_path.name) if source_path is not None else None
        if scene_name is None:
            remapped[key] = item
            continue
        updated = dict(item)
        updated["training_image"] = scene_name
        updated["training_stem"] = Path(scene_name).stem
        remapped[scene_name] = updated
    quality.per_frame_blur = remapped


def resolve_fine_scene_profile(options: dict[str, Any]) -> str:
    profile = str(
        options.get("fine_scene_profile") or options.get("preview_scene_profile") or DEFAULT_FINE_SCENE_PROFILE
    ).strip().lower()
    if profile not in FINE_SCENE_PROFILE_MAX_SIDES:
        return DEFAULT_FINE_SCENE_PROFILE
    return profile


def resolve_colmap_feature_budget(
    options: dict[str, Any],
    quality: Any,
    *,
    fine_scene_profile: str,
    image_count: int,
    matcher: str,
    max_image_size: int,
) -> tuple[int, dict[str, Any]]:
    profile_default = 32_768 if fine_scene_profile == "indoor_full" else COLMAP_SIFT_MAX_NUM_FEATURES
    requested = read_int(options.get("fine_sift_max_num_features"), profile_default, minimum=1024, maximum=65_536)
    auto_enabled = read_bool(options.get("fine_sift_max_num_features_auto"), True)
    legacy_default_values = {32_768, 65_536, COLMAP_SIFT_MAX_NUM_FEATURES, profile_default}
    if not auto_enabled:
        return requested, _colmap_feature_budget_metrics(False, requested, requested, "manual_fixed", quality, image_count, matcher, max_image_size)
    if options.get("fine_sift_max_num_features") not in {None, ""} and requested not in legacy_default_values:
        return requested, _colmap_feature_budget_metrics(False, requested, requested, "manual_override", quality, image_count, matcher, max_image_size)

    blur_ratio = _training_blur_ratio(quality)
    sharp_score = float(getattr(quality, "mean_sharp_score", 0.0) or 0.0)
    if blur_ratio >= 0.55 or sharp_score <= -0.75:
        budget = 20_480
        reason = "very_low_quality_more_features"
    elif blur_ratio >= 0.30 or sharp_score <= 0.10:
        budget = 16_384
        reason = "low_quality_more_features"
    elif blur_ratio <= 0.15 and sharp_score >= 1.00:
        budget = 8_192
        reason = "sharp_high_quality_less_features"
    else:
        budget = 12_288
        reason = "balanced_quality"

    effective_matcher = matcher.strip().lower()
    if effective_matcher == "auto":
        effective_matcher = "exhaustive" if image_count <= 250 else "sequential"
    small_exhaustive_set = effective_matcher == "exhaustive" and 0 < image_count <= 30
    memory_cap = 24_576
    if effective_matcher == "exhaustive":
        if image_count >= 80:
            memory_cap = 12_288
        elif image_count >= 40:
            memory_cap = 16_384
        else:
            memory_cap = 20_480
    if small_exhaustive_set:
        small_set_budget = min(requested, 32_768)
        budget = max(budget, small_set_budget)
        memory_cap = max(memory_cap, small_set_budget)
        reason = "small_image_set_more_features"
    if max_image_size > COLMAP_MAX_IMAGE_SIZE and not small_exhaustive_set:
        memory_cap = min(memory_cap, 12_288)

    resolved = min(budget, memory_cap)
    return resolved, _colmap_feature_budget_metrics(True, requested, resolved, reason, quality, image_count, effective_matcher, max_image_size)


def _training_blur_ratio(quality: Any) -> float:
    kept_images = int(getattr(quality, "kept_images", 0) or 0)
    if kept_images > 0:
        return max(0.0, min(1.0, float(getattr(quality, "training_blur_frames", 0) or 0) / kept_images))
    image_count = int(getattr(quality, "blurred_images", 0) or 0)
    return 1.0 if image_count > 0 else 0.0


def _colmap_feature_budget_metrics(
    auto: bool,
    requested: int,
    resolved: int,
    reason: str,
    quality: Any,
    image_count: int,
    matcher: str,
    max_image_size: int,
) -> dict[str, Any]:
    return {
        "colmap_sift_feature_budget_auto": auto,
        "colmap_sift_feature_budget_requested": requested,
        "colmap_sift_feature_budget_resolved": resolved,
        "colmap_sift_feature_budget_reason": reason,
        "colmap_sift_feature_budget_blur_ratio": round(_training_blur_ratio(quality), 6),
        "colmap_sift_feature_budget_sharp_score": round(float(getattr(quality, "mean_sharp_score", 0.0) or 0.0), 6),
        "colmap_sift_feature_budget_images": image_count,
        "colmap_sift_feature_budget_matcher": matcher,
        "colmap_sift_feature_budget_max_image_size": max_image_size,
    }


def assert_runtime_ready() -> None:
    return None


def assert_training_runtime_ready(options: dict[str, Any], repo_cache_dir: Path) -> None:
    resolve_runtime_paths(options, repo_cache_dir)


def reject_removed_options(options: dict[str, Any]) -> None:
    if read_bool(options.get("fine_edgs_enabled"), False):
        raise FineFailure("UNSUPPORTED_FINE_OPTION", "EDGS/RoMA dense initialization has been removed from this worker image")


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


def ctx_progress(ctx: FineContext, stage: str, progress: int, message: str | None = None, metrics: dict[str, Any] | None = None) -> None:
    if ctx.progress:
        ctx.progress(stage, progress, message, metrics)


def _format_for_log(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)
