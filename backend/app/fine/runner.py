from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.fine.colmap_cli import build_colmap_cli_scene
from app.fine.colmap_defaults import (
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
    FINE_PIPELINE_NAME,
    FINE_SCENE_PROFILE_MAX_SIDES,
    LEGACY_FINE_PIPELINE_ALIASES,
)
from app.fine.option_utils import read_float, read_int
from app.fine.preprocess import build_pycolmap_scene, prepare_fine_images
from app.fine.types import FineContext, FineFailure, FineResult
from app.preview.utils import image_files


PIPELINE_NAME = FINE_PIPELINE_NAME

SOURCE_COMMITS_FINE = {
    "COLMAP": "system",
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
    reject_removed_options(ctx.options)
    assert_runtime_ready()

    image_count = len(image_files(ctx.input_dir))
    if image_count < 3:
        raise FineFailure("INSUFFICIENT_IMAGES", "COLMAP fine reconstruction requires at least 3 images")

    fine_scene_profile = resolve_fine_scene_profile(ctx.options)
    profile_image_max_side = FINE_SCENE_PROFILE_MAX_SIDES[fine_scene_profile]
    default_image_max_side = min(profile_image_max_side, settings.fine_image_max_side, FINE_IMAGE_MAX_SIDE)
    image_max_side = read_int(ctx.options.get("fine_image_max_side"), default_image_max_side, minimum=512, maximum=FINE_IMAGE_MAX_SIDE)
    reject_ratio = read_float(ctx.options.get("fine_blur_reject_ratio"), 0.10, minimum=0.0, maximum=0.45)
    colmap_features = read_int(ctx.options.get("fine_sift_max_num_features"), COLMAP_SIFT_MAX_NUM_FEATURES, minimum=1024, maximum=65_536)
    colmap_max_size = read_int(ctx.options.get("fine_colmap_max_image_size"), COLMAP_MAX_IMAGE_SIZE, minimum=512, maximum=4_096)
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

    scene_dir = ctx.work_dir / "fine_scene"
    scene_result = build_scene(
        ctx,
        train_input_dir,
        scene_dir,
        colmap_features,
        colmap_max_size,
        colmap_threads,
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
    print(
        "[fine-runner] scene build complete "
        f"backend={scene_result.backend} scene_dir={scene_result.scene_dir} "
        f"registered_images={scene_result.registered_images} point_count={scene_result.point_count}",
        flush=True,
    )

    ctx_progress(ctx, "fine_colmap_export", 82, "exporting COLMAP sparse point cloud")
    sparse_ply = ctx.work_dir / "colmap_sparse.ply"
    sparse_points = export_colmap_sparse_ply(scene_result.scene_dir / "sparse" / "0", sparse_ply)
    if sparse_points <= 0:
        raise FineFailure("COLMAP_SPARSE_POINTS_EMPTY", "COLMAP sparse reconstruction contains no exportable 3D points")

    ctx.final_ply.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sparse_ply, ctx.final_ply)
    from app.fine.viewer_meta import read_ply_xyz_bounds, write_final_viewer_meta_json

    bounds = read_ply_xyz_bounds(ctx.final_ply)

    viewer_meta_payload = None
    if ctx.viewer_meta_json:
        viewer_meta_payload = write_final_viewer_meta_json(
            ctx.viewer_meta_json,
            final_ply=ctx.final_ply,
            scene_dir=scene_result.scene_dir,
            preferred_image_names=first_clear_training_images(quality.per_frame_blur),
        )

    metrics = {
        "pipeline": PIPELINE_NAME,
        "algorithm": scene_result.backend,
        "requested_algorithms": [scene_result.backend],
        "effective_algorithms": [scene_result.backend],
        "source_version": ctx.source_version,
        "source_commits": SOURCE_COMMITS_FINE,
        "fine_input_type": ctx.options.get("input_type") or ("video" if ctx.input_video else "images"),
        "input_images": image_count,
        "training_images": len(image_files(train_input_dir)),
        "fine_scene_profile": fine_scene_profile,
        "fine_image_max_side": image_max_side,
        "colmap_sparse_points_exported": sparse_points,
        "final_ply_bytes": ctx.final_ply.stat().st_size,
        "viewer_meta_json_bytes": ctx.viewer_meta_json.stat().st_size if ctx.viewer_meta_json and ctx.viewer_meta_json.exists() else None,
        "bbox_min": bounds["bbox_min"],
        "bbox_max": bounds["bbox_max"],
        "bbox_radius": bounds["radius"],
        **quality.metrics(),
        **scene_result.metrics,
    }
    if viewer_meta_payload is not None:
        metrics["viewer_meta_asset_type"] = viewer_meta_payload.get("asset_type")
    ctx.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    ctx.metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    ctx_progress(ctx, "fine_outputs_ready", 90, "validated COLMAP sparse point cloud", metrics)
    return FineResult(
        final_ply=ctx.final_ply,
        final_spz=None,
        metrics_json=ctx.metrics_json,
        viewer_meta_json=ctx.viewer_meta_json if ctx.viewer_meta_json and ctx.viewer_meta_json.exists() else None,
        lod_rad=None,
        splat_count=None,
        source_commits=SOURCE_COMMITS_FINE,
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
            prefer_gpu=read_bool(ctx.options.get("prefer_gpu"), True),
            gpu_index=str(ctx.options.get("fine_colmap_gpu_index") or "").strip() or None,
            min_registered_ratio=min_registered_ratio,
            progress=lambda stage, progress, message: ctx_progress(ctx, stage, progress, message),
        )
        if sfm_backend == "colmap":
            result.metrics["sfm_backend_requested_alias"] = "colmap_maps_to_colmap_cli"
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
    if min_sparse_points > 0 and point_count < min_sparse_points:
        raise FineFailure(
            "SFM_SPARSE_POINTS_TOO_LOW",
            f"COLMAP produced {point_count} sparse points, below quality gate {min_sparse_points}",
        )
    return result


def export_colmap_sparse_ply(model_dir: Path, output_ply: Path) -> int:
    try:
        import pycolmap
    except Exception as exc:
        raise FineFailure("PYCOLMAP_UNAVAILABLE", f"pycolmap import failed: {exc}") from exc

    reconstruction = pycolmap.Reconstruction(model_dir)
    points = list(getattr(reconstruction, "points3D", {}).values())
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with output_ply.open("wb") as handle:
        handle.write(header)
        for point in points:
            xyz = [float(value) for value in getattr(point, "xyz", (0.0, 0.0, 0.0))]
            color = [int(max(0, min(255, value))) for value in getattr(point, "color", (255, 255, 255))]
            handle.write(struct.pack("<fffBBB", xyz[0], xyz[1], xyz[2], color[0], color[1], color[2]))
    return len(points)


def normalize_fine_pipeline(value: str | None) -> str:
    normalized = (value or PIPELINE_NAME).strip().lower()
    if normalized in LEGACY_FINE_PIPELINE_ALIASES:
        return PIPELINE_NAME
    return normalized


def resolve_fine_scene_profile(options: dict[str, Any]) -> str:
    profile = str(
        options.get("fine_scene_profile") or options.get("preview_scene_profile") or DEFAULT_FINE_SCENE_PROFILE
    ).strip().lower()
    if profile not in FINE_SCENE_PROFILE_MAX_SIDES:
        return DEFAULT_FINE_SCENE_PROFILE
    return profile


def assert_runtime_ready() -> None:
    try:
        import pycolmap  # noqa: F401
    except Exception as exc:
        raise FineFailure("PYCOLMAP_UNAVAILABLE", f"pycolmap import failed: {exc}") from exc


def reject_removed_options(options: dict[str, Any]) -> None:
    if read_bool(options.get("fine_edgs_enabled"), False):
        raise FineFailure("UNSUPPORTED_FINE_OPTION", "EDGS/RoMA dense initialization has been removed from this worker image")


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


def ctx_progress(ctx: FineContext, stage: str, progress: int, message: str | None = None, metrics: dict[str, Any] | None = None) -> None:
    if ctx.progress:
        ctx.progress(stage, progress, message, metrics)


def _format_for_log(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)
