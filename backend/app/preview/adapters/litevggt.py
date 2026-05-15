from __future__ import annotations

from typing import Any

from app.preview.io.ply import convert_pointcloud_ply_to_fixed_splat_ply
from app.preview.io.spz import convert_ply_to_spz
from app.preview.types import PreviewContext, PreviewResult, SOURCE_COMMITS
from app.preview.utils import StageTimer, image_files, require_file
from app.preview.vendor.litevggt_runtime import run_litevggt_pointcloud


DEFAULT_PREVIEW_SCENE_PROFILE = "mixed_balanced"
LITEVGGT_SCENE_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "mixed_balanced": {
        "litevggt_keep_ratio": 0.85,
        "preview_max_points": 15_000_000,
        "litevggt_max_input_frames": None,
        "litevggt_target_size": 476,
        "litevggt_frame_stride": None,
        "litevggt_depth_conf_thresh": None,
        "litevggt_preprocess_mode": "pad",
        "litevggt_inference_mode": "single",
        "litevggt_chunk_size": 64,
        "litevggt_overlap": 16,
        "litevggt_loop_closure": True,
        "litevggt_keyframe_target": 240,
        "litevggt_min_frame_gap": 1,
        "litevggt_min_scene_change": 6.0,
        "litevggt_window_voxel_diag_ratio": 1 / 700,
        "litevggt_final_voxel_diag_ratio": 1 / 800,
        "litevggt_point_selection_strategy": "scene_coverage",
        "litevggt_axis_trim_low_quantile": 0.001,
        "litevggt_axis_trim_high_quantile": 0.999,
        "litevggt_spatial_keep_quantile": 0.9975,
    },
    "indoor_full": {
        "litevggt_keep_ratio": 0.95,
        "preview_max_points": 20_000_000,
        "litevggt_max_input_frames": None,
        "litevggt_target_size": 476,
        "litevggt_frame_stride": None,
        "litevggt_depth_conf_thresh": None,
        "litevggt_preprocess_mode": "pad",
        "litevggt_inference_mode": "single",
        "litevggt_chunk_size": 48,
        "litevggt_overlap": 24,
        "litevggt_loop_closure": True,
        "litevggt_keyframe_target": None,
        "litevggt_min_frame_gap": 1,
        "litevggt_min_scene_change": 0.0,
        "litevggt_window_voxel_diag_ratio": 0.0,
        "litevggt_final_voxel_diag_ratio": 1 / 1000,
        "litevggt_point_selection_strategy": "scene_coverage",
        "litevggt_axis_trim_low_quantile": 0.0005,
        "litevggt_axis_trim_high_quantile": 0.9995,
        "litevggt_spatial_keep_quantile": 0.999,
    },
    "outdoor_fast_clean": {
        "litevggt_keep_ratio": 0.25,
        "preview_max_points": 5_000_000,
        "litevggt_max_input_frames": None,
        "litevggt_target_size": 476,
        "litevggt_frame_stride": None,
        "litevggt_depth_conf_thresh": None,
        "litevggt_preprocess_mode": "pad",
        "litevggt_inference_mode": "single",
        "litevggt_chunk_size": 48,
        "litevggt_overlap": 8,
        "litevggt_loop_closure": True,
        "litevggt_keyframe_target": 160,
        "litevggt_min_frame_gap": 2,
        "litevggt_min_scene_change": 8.0,
        "litevggt_window_voxel_diag_ratio": 1 / 350,
        "litevggt_final_voxel_diag_ratio": 1 / 450,
        "litevggt_point_selection_strategy": "global_confidence",
        "litevggt_axis_trim_low_quantile": 0.01,
        "litevggt_axis_trim_high_quantile": 0.99,
        "litevggt_spatial_keep_quantile": 0.985,
    },
}


def run(ctx: PreviewContext) -> PreviewResult:
    timer = StageTimer()
    weight = require_file(
        ctx.model_path("litevggt", "te_dict.pt"),
        "LITEVGGT_WEIGHT_MISSING",
        "LiteVGGT weight",
    )
    points_ply_path = ctx.work_dir / "litevggt" / "preview_points.ply"
    splats_ply_path = ctx.work_dir / "litevggt" / "preview_splats.ply"
    meta_path = ctx.work_dir / "litevggt" / "preview_meta.json"
    files = image_files(ctx.input_dir)

    scene_profile, profile_defaults = _resolve_scene_profile(ctx.options)
    keep_ratio = _read_float(ctx.options.get("litevggt_keep_ratio"), profile_defaults["litevggt_keep_ratio"])
    max_points = _read_int(ctx.options.get("preview_max_points"), profile_defaults["preview_max_points"])
    max_input_frames_value = ctx.options.get("litevggt_max_input_frames")
    if max_input_frames_value is not None:
        max_input_frames = _read_optional_int(max_input_frames_value)
    else:
        max_input_frames = profile_defaults["litevggt_max_input_frames"]
    target_size = _read_int(ctx.options.get("litevggt_target_size"), profile_defaults["litevggt_target_size"])
    frame_stride = _read_optional_int(ctx.options.get("litevggt_frame_stride", profile_defaults["litevggt_frame_stride"]))
    depth_conf_thresh = _read_optional_float(
        ctx.options.get("litevggt_depth_conf_thresh"),
        profile_defaults["litevggt_depth_conf_thresh"],
    )
    preprocess_mode = str(ctx.options.get("litevggt_preprocess_mode") or profile_defaults["litevggt_preprocess_mode"])
    point_selection_strategy = str(
        ctx.options.get("litevggt_point_selection_strategy") or profile_defaults["litevggt_point_selection_strategy"]
    )
    inference_mode = str(ctx.options.get("litevggt_inference_mode") or profile_defaults["litevggt_inference_mode"])
    chunk_size = _read_int(ctx.options.get("litevggt_chunk_size"), profile_defaults["litevggt_chunk_size"])
    overlap = _read_int(ctx.options.get("litevggt_overlap"), profile_defaults["litevggt_overlap"])
    loop_closure = _read_bool(ctx.options.get("litevggt_loop_closure"), profile_defaults["litevggt_loop_closure"])
    keyframe_target = _read_optional_int(
        ctx.options.get("litevggt_keyframe_target", profile_defaults["litevggt_keyframe_target"])
    )
    min_frame_gap = _read_int(ctx.options.get("litevggt_min_frame_gap"), profile_defaults["litevggt_min_frame_gap"])
    min_scene_change = _read_float(
        ctx.options.get("litevggt_min_scene_change"),
        profile_defaults["litevggt_min_scene_change"],
    )
    window_voxel_diag_ratio = _read_float(
        ctx.options.get("litevggt_window_voxel_diag_ratio"),
        profile_defaults["litevggt_window_voxel_diag_ratio"],
    )
    final_voxel_diag_ratio = _read_float(
        ctx.options.get("litevggt_final_voxel_diag_ratio"),
        profile_defaults["litevggt_final_voxel_diag_ratio"],
    )
    axis_trim_low_quantile = _read_float(
        ctx.options.get("litevggt_axis_trim_low_quantile"),
        profile_defaults["litevggt_axis_trim_low_quantile"],
    )
    axis_trim_high_quantile = _read_float(
        ctx.options.get("litevggt_axis_trim_high_quantile"),
        profile_defaults["litevggt_axis_trim_high_quantile"],
    )
    spatial_keep_quantile = _read_float(
        ctx.options.get("litevggt_spatial_keep_quantile"),
        profile_defaults["litevggt_spatial_keep_quantile"],
    )
    params: dict[str, Any] = {
        "preview_scene_profile": scene_profile,
        "keep_ratio": keep_ratio,
        "max_points": max_points,
        "max_input_frames": max_input_frames,
        "target_size": target_size,
        "frame_stride": frame_stride if frame_stride is not None else "auto",
        "depth_conf_thresh": depth_conf_thresh,
        "preprocess_mode": preprocess_mode,
        "inference_mode": inference_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "loop_closure": loop_closure,
        "keyframe_target": keyframe_target,
        "min_frame_gap": min_frame_gap,
        "min_scene_change": min_scene_change,
        "window_voxel_diag_ratio": window_voxel_diag_ratio,
        "final_voxel_diag_ratio": final_voxel_diag_ratio,
        "point_selection_strategy": point_selection_strategy,
        "axis_trim_low_quantile": axis_trim_low_quantile,
        "axis_trim_high_quantile": axis_trim_high_quantile,
        "spatial_keep_quantile": spatial_keep_quantile,
    }
    print(
        "[litevggt-preview] adapter params "
        f"task_id={ctx.task_id} project_id={ctx.project_id} input_dir={ctx.input_dir} "
        f"image_count={len(files)} first_images={_first_names(files)} weight={weight} "
        f"weight_bytes={weight.stat().st_size} output_points_ply={points_ply_path} "
        f"output_splats_ply={splats_ply_path} output_spz={ctx.output_spz} "
        + " ".join(f"{key}={value}" for key, value in params.items()),
        flush=True,
    )
    ctx.report(
        "litevggt_preflight",
        22,
        f"LiteVGGT preview path: images={len(files)} mode={inference_mode} chunk={chunk_size} overlap={overlap} keep_ratio={params['keep_ratio']} max_points={max_points}",
    )

    def report(stage: str, progress: int, message: str) -> None:
        ctx.report(stage, progress, message)

    metrics = run_litevggt_pointcloud(
        input_dir=ctx.input_dir,
        checkpoint_path=weight,
        output_ply=points_ply_path,
        output_meta_json=meta_path,
        keep_ratio=keep_ratio,
        max_points=max_points,
        max_input_frames=max_input_frames,
        target_size=target_size,
        frame_stride=frame_stride,
        depth_conf_thresh=depth_conf_thresh,
        preprocess_mode=preprocess_mode,
        inference_mode=inference_mode,
        chunk_size=chunk_size,
        overlap=overlap,
        loop_closure=loop_closure,
        scene_profile=scene_profile,
        keyframe_target=keyframe_target,
        min_frame_gap=min_frame_gap,
        min_scene_change=min_scene_change,
        window_voxel_diag_ratio=window_voxel_diag_ratio,
        final_voxel_diag_ratio=final_voxel_diag_ratio,
        selection_strategy=point_selection_strategy,
        axis_trim_low_quantile=axis_trim_low_quantile,
        axis_trim_high_quantile=axis_trim_high_quantile,
        spatial_keep_quantile=spatial_keep_quantile,
        progress=report,
    )
    timer.mark("litevggt_inference")
    print(
        "[litevggt-preview] inference metrics "
        f"task_id={ctx.task_id} output_points_ply={points_ply_path} "
        f"points_ply_exists={points_ply_path.exists()} "
        f"points_ply_bytes={points_ply_path.stat().st_size if points_ply_path.exists() else None} "
        f"metrics={_format_metrics(metrics)}",
        flush=True,
    )

    base_point_radius = _read_bounded_float(metrics.get("litevggt_preview_point_radius"), 0.002, 1e-8, 1.0)
    point_radius_scale = _read_bounded_float(
        ctx.options.get("preview_fixed_splat_radius_scale", ctx.options.get("litevggt_point_radius_scale")),
        0.22,
        0.05,
        20.0,
    )
    fixed_splat_opacity = _read_bounded_float(ctx.options.get("preview_fixed_splat_opacity"), 0.55, 0.05, 0.99)
    point_radius = base_point_radius * point_radius_scale
    ctx.report("splat_conversion", 82, "converting LiteVGGT point cloud PLY to fixed Gaussian PLY")
    converted_splat_count = convert_pointcloud_ply_to_fixed_splat_ply(
        points_ply_path,
        splats_ply_path,
        point_radius=point_radius,
        opacity=fixed_splat_opacity,
    )
    timer.mark("fixed_splat_conversion")

    ctx.report("spz_conversion", 86, "converting LiteVGGT fixed Gaussian PLY to Spark SPZ")
    splat_count = convert_ply_to_spz(splats_ply_path, ctx.output_spz)
    timer.mark("spz_conversion")
    print(
        "[litevggt-preview] conversion complete "
        f"task_id={ctx.task_id} output_splats_ply={splats_ply_path} output_spz={ctx.output_spz} "
        f"splats_ply_exists={splats_ply_path.exists()} "
        f"splats_ply_bytes={splats_ply_path.stat().st_size if splats_ply_path.exists() else None} "
        f"spz_exists={ctx.output_spz.exists()} spz_bytes={ctx.output_spz.stat().st_size if ctx.output_spz.exists() else None} "
        f"splat_count={splat_count} stage_durations={timer.metrics().get('stage_durations')}",
        flush=True,
    )

    return PreviewResult(
        output_spz=ctx.output_spz,
        intermediate_ply=points_ply_path,
        splat_count=splat_count,
        metrics={
            **metrics,
            **timer.metrics(),
            "adapter": "litevggt_spz",
            "preview_scene_profile": scene_profile,
            "intermediate_ply_size": points_ply_path.stat().st_size,
            "intermediate_points_ply": str(points_ply_path),
            "intermediate_splats_ply": str(splats_ply_path),
            "preview_meta_json": str(meta_path),
            "intermediate_points_ply_size": points_ply_path.stat().st_size,
            "intermediate_splats_ply_size": splats_ply_path.stat().st_size,
            "preview_meta_json_size": meta_path.stat().st_size,
            "fixed_splat_base_point_radius": base_point_radius,
            "fixed_splat_point_radius_scale": point_radius_scale,
            "fixed_splat_point_radius": point_radius,
            "fixed_splat_opacity": fixed_splat_opacity,
            "fixed_splat_count": converted_splat_count,
        },
        source_commits={"LiteVGGT": SOURCE_COMMITS["LiteVGGT"], "Spark": SOURCE_COMMITS["Spark"]},
    )


def _resolve_scene_profile(options: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    profile = str(options.get("preview_scene_profile") or DEFAULT_PREVIEW_SCENE_PROFILE).strip().lower()
    if profile not in LITEVGGT_SCENE_PROFILE_DEFAULTS:
        profile = DEFAULT_PREVIEW_SCENE_PROFILE
    return profile, LITEVGGT_SCENE_PROFILE_DEFAULTS[profile]


def _first_names(paths, limit: int = 8) -> str:
    names = [path.name for path in paths[:limit]]
    suffix = "" if len(paths) <= limit else f", ... +{len(paths) - limit}"
    return "[" + ", ".join(names) + suffix + "]"


def _format_metrics(metrics: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in sorted(metrics.items()))


def _read_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "auto", "none"}:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _read_int(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _read_float(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _read_bool(value: Any, fallback: bool) -> bool:
    if value is None:
        return bool(fallback)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(fallback)


def _read_bounded_float(value: Any, fallback: float, minimum: float, maximum: float) -> float:
    parsed = _read_float(value, fallback)
    return max(minimum, min(maximum, parsed))


def _read_optional_float(value: Any, fallback: float | None) -> float | None:
    if value is None:
        return fallback
    normalized = str(value).strip().lower()
    if normalized in {"", "auto"}:
        return fallback
    if normalized in {"none", "off", "false"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback
