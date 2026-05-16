from __future__ import annotations

from pathlib import Path

from app.preview.types import PreviewArtifactResult, PreviewContext, PreviewFailure, PreviewResult, SOURCE_COMMITS
from app.preview.utils import StageTimer, require_file
from app.preview.vendor.lingbot_runtime import PointCloudVideoConfig, run_lingbot_video_pointcloud_fast


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def run(ctx: PreviewContext) -> PreviewResult:
    timer = StageTimer()
    video_path = single_video_file(ctx.input_dir)
    weight = require_file(
        ctx.model_path("lingbot", "lingbot-map-long.pt"),
        "LINGBOT_WEIGHT_MISSING",
        "LingBot-Map weight",
    )
    output_dir = ctx.work_dir / "lingbot_pointcloud"
    fast_ply_path = output_dir / "preview_fast.ply"
    full_ply_path = output_dir / "preview_full.ply"
    camera_path_json = output_dir / "camera_path.json"
    metrics_json = output_dir / "metrics.json"
    meta_path = output_dir / "preview_meta.json"

    config = PointCloudVideoConfig(
        fps=read_float(ctx.options.get("preview_lingbot_fps"), 10.0, minimum=0.1, maximum=60.0),
        image_size=read_int(ctx.options.get("preview_lingbot_image_size"), 518, minimum=224, maximum=1024),
        target_width=read_int(ctx.options.get("preview_lingbot_target_width"), 518, minimum=14, maximum=2048),
        target_height=read_int(ctx.options.get("preview_lingbot_target_height"), 378, minimum=14, maximum=2048),
        window_size=read_int(ctx.options.get("preview_lingbot_window_size"), 128, minimum=8, maximum=512),
        keyframe_interval=read_int(ctx.options.get("preview_lingbot_keyframe_interval"), 13, minimum=1, maximum=100_000),
        overlap_keyframes=read_int(ctx.options.get("preview_lingbot_overlap_keyframes"), 16, minimum=1, maximum=128),
        num_scale_frames=read_int(ctx.options.get("preview_lingbot_num_scale_frames"), 8, minimum=1, maximum=64),
        camera_iterations_fast=read_int(ctx.options.get("preview_lingbot_camera_iterations"), 2, minimum=1, maximum=8),
        camera_iterations_retry=read_int(ctx.options.get("preview_lingbot_camera_iterations_retry"), 2, minimum=1, maximum=8),
        pixel_stride_fast=read_int(ctx.options.get("preview_lingbot_pixel_stride_fast"), 5, minimum=1, maximum=512),
        pixel_stride_full=read_int(ctx.options.get("preview_lingbot_pixel_stride_full"), 3, minimum=1, maximum=512),
        conf_percentile_fast=read_float(ctx.options.get("preview_lingbot_conf_percentile_fast"), 45.0, minimum=0.0, maximum=100.0),
        conf_percentile_full=read_float(ctx.options.get("preview_lingbot_conf_percentile_full"), 35.0, minimum=0.0, maximum=100.0),
        min_conf=read_float(ctx.options.get("preview_lingbot_min_conf"), 0.8, minimum=-100.0, maximum=100.0),
        allow_sdpa_fallback=read_bool(ctx.options.get("preview_lingbot_allow_sdpa_fallback"), False),
        compile_model=read_bool(ctx.options.get("preview_lingbot_compile"), False),
        write_progressive_preview=read_bool(ctx.options.get("preview_lingbot_write_progressive_preview"), True),
        voxel_target_fast=read_int(ctx.options.get("preview_lingbot_voxel_target_fast"), 3000, minimum=1, maximum=100_000),
        voxel_target_full=read_int(ctx.options.get("preview_lingbot_voxel_target_full"), 5200, minimum=1, maximum=100_000),
        coverage_keyframes=read_bool(ctx.options.get("preview_lingbot_coverage_keyframes"), True),
        save_debug_predictions=read_bool(ctx.options.get("preview_lingbot_save_official_predictions"), False),
    )
    print(
        "[lingbot-pointcloud] adapter params "
        f"task_id={ctx.task_id} project_id={ctx.project_id} input_dir={ctx.input_dir} work_dir={ctx.work_dir} "
        f"video={video_path} video_bytes={video_path.stat().st_size} weight={weight} weight_bytes={weight.stat().st_size} "
        f"output_fast_ply={fast_ply_path} output_full_ply={full_ply_path} "
        f"camera_path_json={camera_path_json} metrics_json={metrics_json} preview_meta_json={meta_path} "
        f"config={config}",
        flush=True,
    )
    ctx.report(
        "lingbot_pointcloud_preflight",
        22,
        (
            "LingBot point cloud params: "
            f"fps={config.fps:g} max_frames=0 mode=windowed "
            f"window={config.window_size} keyframe_interval={config.keyframe_interval} "
            f"camera_iters={config.camera_iterations_fast}"
        ),
    )

    metrics = run_lingbot_video_pointcloud_fast(
        video_path=video_path,
        model_path=weight,
        work_dir=output_dir,
        output_fast_ply=fast_ply_path,
        output_full_ply=full_ply_path,
        output_camera_path_json=camera_path_json,
        output_metrics_json=metrics_json,
        output_meta_json=meta_path,
        config=config,
        progress=lambda stage, progress, message: ctx.report(stage, progress, message),
    )
    timer.mark("lingbot_pointcloud")
    print(
        "[lingbot-pointcloud] runtime metrics "
        f"task_id={ctx.task_id} metrics="
        + " ".join(f"{key}={value}" for key, value in sorted(metrics.items()) if key != "lingbot_window_metrics"),
        flush=True,
    )
    return PreviewResult(
        output_spz=ctx.output_spz,
        intermediate_ply=fast_ply_path,
        splat_count=None,
        metrics={
            **metrics,
            **timer.metrics(),
            "adapter": "lingbot_video_pointcloud_fast",
            "intermediate_ply": str(fast_ply_path),
            "intermediate_ply_size": fast_ply_path.stat().st_size,
            "point_source": "world_points_from_depth",
        },
        source_commits={"LingBot-Map": SOURCE_COMMITS["LingBot-Map"]},
        primary_artifact=fast_ply_path,
        primary_artifact_kind="preview_pointcloud_ply",
        primary_artifact_file_name="preview_fast.ply",
        primary_artifact_format="ply",
        extra_artifacts=(
            PreviewArtifactResult(full_ply_path, "preview_full_ply", "preview_full.ply"),
            PreviewArtifactResult(camera_path_json, "camera_path_json", "camera_path.json"),
            PreviewArtifactResult(metrics_json, "preview_metrics_json", "metrics.json"),
        ),
    )


def single_video_file(input_dir: Path) -> Path:
    videos = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)
    if len(videos) != 1:
        raise PreviewFailure("INVALID_VIDEO_INPUT", "LingBot point-cloud preview requires exactly one video file")
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
