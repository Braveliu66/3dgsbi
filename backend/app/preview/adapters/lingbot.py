from __future__ import annotations

from pathlib import Path

from app.preview.io.ply import convert_pointcloud_ply_to_fixed_splat_ply
from app.preview.io.spz import convert_ply_to_spz
from app.preview.types import PreviewContext, PreviewFailure, PreviewResult, SOURCE_COMMITS
from app.preview.utils import StageTimer, require_file
from app.preview.vendor.lingbot_runtime import run_lingbot_video_preview


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def run(ctx: PreviewContext) -> PreviewResult:
    timer = StageTimer()
    video_path = single_video_file(ctx.input_dir)
    weight = require_file(
        ctx.model_path("lingbot", "lingbot-map-long.pt"),
        "LINGBOT_WEIGHT_MISSING",
        "LingBot-Map weight",
    )
    points_ply_path = ctx.work_dir / "lingbot" / "preview_points.ply"
    splats_ply_path = ctx.work_dir / "lingbot" / "preview_splats.ply"
    meta_path = ctx.work_dir / "lingbot" / "preview_meta.json"
    official_predictions_path = ctx.work_dir / "lingbot" / "official_predictions.npz"

    params = {
        "fps": read_int(ctx.options.get("preview_lingbot_fps"), 3, minimum=0, maximum=60),
        "max_frames": read_int(ctx.options.get("preview_lingbot_max_frames"), 0, minimum=0, maximum=100_000),
        "image_size": read_int(ctx.options.get("preview_lingbot_image_size"), 518, minimum=224, maximum=1024),
        "target_width": read_int(ctx.options.get("preview_lingbot_target_width"), 518, minimum=14, maximum=2048),
        "target_height": read_int(ctx.options.get("preview_lingbot_target_height"), 378, minimum=14, maximum=2048),
        "mode": str(ctx.options.get("preview_lingbot_mode") or "windowed"),
        "keyframe_interval": read_int(ctx.options.get("preview_lingbot_keyframe_interval"), 4, minimum=1, maximum=100_000),
        "camera_iterations": read_int(ctx.options.get("preview_lingbot_camera_iterations"), 1, minimum=1, maximum=8),
        "num_scale_frames": read_int(ctx.options.get("preview_lingbot_num_scale_frames"), 4, minimum=1, maximum=64),
        "preprocess_mode": str(ctx.options.get("preview_lingbot_preprocess_mode") or "crop"),
        "window_size": read_int(ctx.options.get("preview_lingbot_window_size"), 32, minimum=8, maximum=512),
        "overlap_keyframes": read_int(ctx.options.get("preview_lingbot_overlap_keyframes"), 8, minimum=1, maximum=128),
        "max_points": read_int(ctx.options.get("preview_lingbot_max_points"), 1_000_000_000, minimum=0, maximum=1_000_000_000),
        "frame_stride": read_int(ctx.options.get("preview_lingbot_frame_stride"), 1, minimum=1, maximum=10_000),
        "pixel_stride": read_int(ctx.options.get("preview_lingbot_pixel_stride"), 2, minimum=1, maximum=512),
        "conf_percentile": read_float(ctx.options.get("preview_lingbot_conf_percentile"), 10.0, minimum=0.0, maximum=100.0),
        "min_conf": read_float(ctx.options.get("preview_lingbot_min_conf"), 0.0, minimum=-100.0, maximum=100.0),
        "save_predictions": read_bool(ctx.options.get("preview_lingbot_save_predictions"), True),
        "compile_model": read_bool(ctx.options.get("preview_lingbot_compile"), False),
        "keyframes_only_points": read_bool(ctx.options.get("preview_lingbot_keyframes_only_points"), True),
        "allow_sdpa_fallback": read_bool(ctx.options.get("preview_lingbot_allow_sdpa_fallback"), False),
        "min_inference_fps": read_float(ctx.options.get("preview_lingbot_min_inference_fps"), 3.0, minimum=0.0, maximum=1000.0),
    }
    print(
        "[lingbot-preview] adapter params "
        f"task_id={ctx.task_id} project_id={ctx.project_id} input_dir={ctx.input_dir} work_dir={ctx.work_dir} "
        f"video={video_path} video_bytes={video_path.stat().st_size} weight={weight} weight_bytes={weight.stat().st_size} "
        f"output_points_ply={points_ply_path} output_splats_ply={splats_ply_path} "
        f"output_meta_json={meta_path} output_official_npz={official_predictions_path} output_spz={ctx.output_spz} "
        + " ".join(f"{key}={value}" for key, value in params.items()),
        flush=True,
    )
    ctx.report(
        "lingbot_preflight",
        22,
        (
            "LingBot params: "
            f"fps={params['fps']} max_frames={params['max_frames']} mode={params['mode']} "
            f"target={params['target_width']}x{params['target_height']} "
            f"camera_iters={params['camera_iterations']} pixel_stride={params['pixel_stride']} "
            f"conf_p={params['conf_percentile']}"
        ),
    )

    metrics = run_lingbot_video_preview(
        video_path=video_path,
        model_path=weight,
        work_dir=ctx.work_dir / "lingbot",
        output_points_ply=points_ply_path,
        output_splats_ply=splats_ply_path,
        output_meta_json=meta_path,
        output_official_predictions_npz=official_predictions_path,
        **params,
        progress=lambda stage, progress, message: ctx.report(stage, progress, message),
    )
    timer.mark("lingbot_inference")
    print(
        "[lingbot-preview] runtime metrics "
        f"task_id={ctx.task_id} metrics="
        + " ".join(f"{key}={value}" for key, value in sorted(metrics.items())),
        flush=True,
    )

    base_point_radius = read_float(metrics.get("lingbot_preview_point_radius"), 0.002, minimum=1e-8, maximum=1.0)
    point_radius_scale = read_float(ctx.options.get("preview_lingbot_point_radius_scale"), 1.0, minimum=0.1, maximum=20.0)
    point_radius = base_point_radius * point_radius_scale
    print(
        "[lingbot-preview] pointcloud summary "
        f"source={metrics.get('lingbot_point_source')} depth_fallback={metrics.get('lingbot_depth_reprojection_fallback')} "
        f"frames={metrics.get('lingbot_point_frame_count')}/{metrics.get('lingbot_point_source_frames')} "
        f"raw={metrics.get('point_count_raw')} conf_filtered={metrics.get('lingbot_points_filtered_by_confidence')} "
        f"after_conf={metrics.get('lingbot_points_after_confidence_filter')} "
        f"exported={metrics.get('point_count_exported')} bbox_radius={metrics.get('bbox_radius')} "
        f"base_radius={base_point_radius} radius_scale={point_radius_scale} final_radius={point_radius}",
        flush=True,
    )
    ctx.report(
        "lingbot_pointcloud_ready",
        78,
        (
            "LingBot points: "
            f"raw={metrics.get('point_count_raw')} "
            f"filtered={metrics.get('lingbot_points_filtered_by_confidence')} "
            f"exported={metrics.get('point_count_exported')} "
            f"bbox_radius={metrics.get('bbox_radius')}"
        ),
    )
    ctx.report("splat_ply_conversion", 82, f"fixed Gaussian radius={point_radius:.6g} scale={point_radius_scale:.2f}")
    print(
        "[lingbot-preview] converting point cloud to fixed Gaussian PLY "
        f"input={points_ply_path} input_bytes={points_ply_path.stat().st_size if points_ply_path.exists() else None} "
        f"output={splats_ply_path} point_radius={point_radius} opacity=0.75",
        flush=True,
    )
    splat_ply_count = convert_pointcloud_ply_to_fixed_splat_ply(
        points_ply_path,
        splats_ply_path,
        point_radius=point_radius,
        opacity=0.75,
    )
    timer.mark("splat_ply_conversion")
    print(
        "[lingbot-preview] fixed Gaussian PLY ready "
        f"output={splats_ply_path} output_bytes={splats_ply_path.stat().st_size if splats_ply_path.exists() else None} "
        f"splat_ply_count={splat_ply_count}",
        flush=True,
    )

    ctx.report("spz_conversion", 86, "converting fixed LingBot-Map Gaussian PLY to Spark SPZ")
    splat_count = convert_ply_to_spz(splats_ply_path, ctx.output_spz)
    timer.mark("spz_conversion")
    print(
        "[lingbot-preview] conversion complete "
        f"task_id={ctx.task_id} output_spz={ctx.output_spz} "
        f"spz_exists={ctx.output_spz.exists()} spz_bytes={ctx.output_spz.stat().st_size if ctx.output_spz.exists() else None} "
        f"splat_count={splat_count} meta_bytes={meta_path.stat().st_size if meta_path.exists() else None} "
        f"stage_durations={timer.metrics().get('stage_durations')}",
        flush=True,
    )

    return PreviewResult(
        output_spz=ctx.output_spz,
        intermediate_ply=points_ply_path,
        splat_count=splat_count,
        metrics={
            **metrics,
            **timer.metrics(),
            "adapter": "lingbot_map_spz",
            "point_source": metrics.get("lingbot_point_source"),
            "fixed_splat_ply_count": splat_ply_count,
            "fixed_splat_point_radius": point_radius,
            "fixed_splat_base_point_radius": base_point_radius,
            "fixed_splat_point_radius_scale": point_radius_scale,
            "intermediate_points_ply": str(points_ply_path),
            "intermediate_splats_ply": str(splats_ply_path),
            "preview_meta_json": str(meta_path),
            "lingbot_official_predictions_npz": str(official_predictions_path),
            "spark_asset": str(ctx.output_spz),
            "intermediate_ply_size": points_ply_path.stat().st_size,
            "intermediate_points_ply_size": points_ply_path.stat().st_size,
            "intermediate_splats_ply_size": splats_ply_path.stat().st_size,
            "preview_meta_json_size": meta_path.stat().st_size,
            "lingbot_official_predictions_npz_size": official_predictions_path.stat().st_size,
        },
        source_commits={"LingBot-Map": SOURCE_COMMITS["LingBot-Map"], "Spark": SOURCE_COMMITS["Spark"]},
    )


def single_video_file(input_dir: Path) -> Path:
    videos = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)
    if len(videos) != 1:
        raise PreviewFailure("INVALID_VIDEO_INPUT", "LingBot-Map video preview requires exactly one video file")
    return videos[0]


def read_int(value, fallback: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, min(maximum, parsed))


def read_float(value, fallback: float, *, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, min(maximum, parsed))


def read_bool(value, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return fallback
