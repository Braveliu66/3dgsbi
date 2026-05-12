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

    metrics = run_lingbot_video_preview(
        video_path=video_path,
        model_path=weight,
        work_dir=ctx.work_dir / "lingbot",
        output_points_ply=points_ply_path,
        output_splats_ply=splats_ply_path,
        output_meta_json=meta_path,
        fps=read_int(ctx.options.get("preview_lingbot_fps"), 5, minimum=1, maximum=60),
        max_frames=read_int(ctx.options.get("preview_lingbot_max_frames"), 160, minimum=0, maximum=100_000),
        image_size=read_int(ctx.options.get("preview_lingbot_image_size"), 518, minimum=224, maximum=1024),
        mode=str(ctx.options.get("preview_lingbot_mode") or "windowed"),
        keyframe_interval=read_optional_int(ctx.options.get("preview_lingbot_keyframe_interval") or 2),
        camera_iterations=read_int(ctx.options.get("preview_lingbot_camera_iterations"), 1, minimum=1, maximum=8),
        num_scale_frames=read_int(ctx.options.get("preview_lingbot_num_scale_frames"), 4, minimum=1, maximum=64),
        window_size=read_int(ctx.options.get("preview_lingbot_window_size"), 64, minimum=8, maximum=512),
        overlap_keyframes=read_int(ctx.options.get("preview_lingbot_overlap_keyframes"), 12, minimum=1, maximum=128),
        max_points=read_int(ctx.options.get("preview_lingbot_max_points"), 600_000, minimum=0, maximum=200_000_000),
        frame_stride=read_int(ctx.options.get("preview_lingbot_frame_stride"), 1, minimum=1, maximum=10_000),
        pixel_stride=read_int(ctx.options.get("preview_lingbot_pixel_stride"), 8, minimum=1, maximum=512),
        conf_percentile=read_float(ctx.options.get("preview_lingbot_conf_percentile"), 45.0, minimum=0.0, maximum=100.0),
        min_conf=read_float(ctx.options.get("preview_lingbot_min_conf"), 1e-5, minimum=-100.0, maximum=100.0),
        save_predictions=read_bool(ctx.options.get("preview_lingbot_save_predictions"), False),
        compile_model=read_bool(ctx.options.get("preview_lingbot_compile"), True),
        keyframes_only_points=read_bool(ctx.options.get("preview_lingbot_keyframes_only_points"), True),
        allow_sdpa_fallback=read_bool(ctx.options.get("preview_lingbot_allow_sdpa_fallback"), False),
        min_inference_fps=read_float(ctx.options.get("preview_lingbot_min_inference_fps"), 3.0, minimum=0.0, maximum=1000.0),
        progress=lambda stage, progress, message: ctx.report(stage, progress, message),
    )
    timer.mark("lingbot_inference")

    point_radius = read_float(metrics.get("lingbot_preview_point_radius"), 0.002, minimum=1e-8, maximum=1.0)
    ctx.report("splat_ply_conversion", 82, "converting LingBot-Map point-cloud PLY to fixed Gaussian PLY")
    splat_ply_count = convert_pointcloud_ply_to_fixed_splat_ply(
        points_ply_path,
        splats_ply_path,
        point_radius=point_radius,
        opacity=0.75,
    )
    timer.mark("splat_ply_conversion")

    ctx.report("spz_conversion", 86, "converting fixed LingBot-Map Gaussian PLY to Spark SPZ")
    splat_count = convert_ply_to_spz(splats_ply_path, ctx.output_spz)
    timer.mark("spz_conversion")

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
            "intermediate_points_ply": str(points_ply_path),
            "intermediate_splats_ply": str(splats_ply_path),
            "preview_meta_json": str(meta_path),
            "spark_asset": str(ctx.output_spz),
            "intermediate_ply_size": points_ply_path.stat().st_size,
            "intermediate_points_ply_size": points_ply_path.stat().st_size,
            "intermediate_splats_ply_size": splats_ply_path.stat().st_size,
            "preview_meta_json_size": meta_path.stat().st_size,
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


def read_optional_int(value) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


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
