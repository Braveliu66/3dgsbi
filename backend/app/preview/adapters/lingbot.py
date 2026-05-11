from __future__ import annotations

from pathlib import Path

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
    ply_path = ctx.work_dir / "lingbot" / "preview.ply"

    metrics = run_lingbot_video_preview(
        video_path=video_path,
        model_path=weight,
        output_ply=ply_path,
        work_dir=ctx.work_dir / "lingbot",
        fps=read_int(ctx.options.get("preview_lingbot_fps"), 2, minimum=1, maximum=60),
        max_frames=read_int(ctx.options.get("preview_lingbot_max_frames"), 128, minimum=0, maximum=100_000),
        image_size=read_int(ctx.options.get("preview_lingbot_image_size"), 518, minimum=224, maximum=1024),
        mode=str(ctx.options.get("preview_lingbot_mode") or "auto"),
        keyframe_interval=read_optional_int(ctx.options.get("preview_lingbot_keyframe_interval")),
        camera_iterations=read_int(ctx.options.get("preview_lingbot_camera_iterations"), 1, minimum=1, maximum=8),
        num_scale_frames=read_int(ctx.options.get("preview_lingbot_num_scale_frames"), 2, minimum=1, maximum=64),
        window_size=read_int(ctx.options.get("preview_lingbot_window_size"), 64, minimum=8, maximum=512),
        overlap_keyframes=read_int(ctx.options.get("preview_lingbot_overlap_keyframes"), 4, minimum=1, maximum=128),
        max_points=read_int(ctx.options.get("preview_lingbot_max_points"), 2_000_000, minimum=0, maximum=200_000_000),
        frame_stride=read_int(ctx.options.get("preview_lingbot_frame_stride"), 2, minimum=1, maximum=10_000),
        pixel_stride=read_int(ctx.options.get("preview_lingbot_pixel_stride"), 6, minimum=1, maximum=512),
        conf_percentile=read_float(ctx.options.get("preview_lingbot_conf_percentile"), 10.0, minimum=0.0, maximum=100.0),
        min_conf=read_float(ctx.options.get("preview_lingbot_min_conf"), 1e-5, minimum=-100.0, maximum=100.0),
        save_predictions=read_bool(ctx.options.get("preview_lingbot_save_predictions"), False),
        compile_model=read_bool(ctx.options.get("preview_lingbot_compile"), False),
        keyframes_only_points=read_bool(ctx.options.get("preview_lingbot_keyframes_only_points"), True),
        allow_sdpa_fallback=read_bool(ctx.options.get("preview_lingbot_allow_sdpa_fallback"), False),
        min_inference_fps=read_float(ctx.options.get("preview_lingbot_min_inference_fps"), 3.0, minimum=0.0, maximum=1000.0),
        progress=lambda stage, progress, message: ctx.report(stage, progress, message),
    )
    timer.mark("lingbot_inference")

    ctx.report("spz_conversion", 86, "converting LingBot-Map plain PLY to Spark SPZ")
    splat_count = convert_ply_to_spz(ply_path, ctx.output_spz)
    timer.mark("spz_conversion")

    return PreviewResult(
        output_spz=ctx.output_spz,
        intermediate_ply=ply_path,
        splat_count=splat_count,
        metrics={
            **metrics,
            **timer.metrics(),
            "adapter": "lingbot_map_spz",
            "intermediate_ply_size": ply_path.stat().st_size,
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
