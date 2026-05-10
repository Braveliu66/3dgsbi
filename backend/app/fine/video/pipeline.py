from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from app.config import Settings
from app.fine.option_utils import read_int
from app.fine.types import FineContext, FineFailure, FineResult
from app.fine.video.artdeco_trainer import ARTDECO_COMMIT, SPEED3R_COMMIT, run_artdeco_speed3r_training
from app.fine.video.calibration import resolve_video_intrinsics, write_artdeco_intrinsics
from app.fine.video.frames import extract_video_frames
from app.preview.io.spz import convert_ply_to_spz
from app.preview.types import PreviewFailure, SOURCE_COMMITS


VIDEO_PIPELINE_NAME = "video_artdeco_speed3r"


SOURCE_COMMITS_VIDEO_FINE = {
    "ARTDECO": ARTDECO_COMMIT,
    "Speed3R": SPEED3R_COMMIT,
    "MASt3R": "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    "Spark": SOURCE_COMMITS["Spark"],
}


def run_video_artdeco_speed3r_pipeline(
    ctx: FineContext,
    *,
    settings: Settings,
    lod_builder: Callable[[FineContext], Path | None],
) -> FineResult:
    if ctx.input_video is None:
        raise FineFailure("VIDEO_INPUT_MISSING", "Video ARTDECO fine reconstruction requires FineContext.input_video")
    if not ctx.input_video.exists() or ctx.input_video.stat().st_size <= 0:
        raise FineFailure("VIDEO_INPUT_MISSING", f"Missing non-empty input video: {ctx.input_video}")

    ctx_progress(ctx, "video_frame_extracting", 20, "extracting video frames for ARTDECO selfCaptured dataset")
    max_frames = read_int(ctx.options.get("fine_video_max_frames"), 240, minimum=2, maximum=2_000)
    max_side = read_int(ctx.options.get("fine_video_frame_max_side"), 1600, minimum=256, maximum=4096)
    frames = extract_video_frames(
        ctx.input_video,
        ctx.work_dir / "artdeco_dataset",
        max_frames=max_frames,
        max_side=max_side,
    )
    intrinsics = resolve_video_intrinsics(frames.width, frames.height, ctx.options)
    intrinsics_path = write_artdeco_intrinsics(frames.dataset_root / "intr.yaml", intrinsics)
    ctx_progress(ctx, "artdeco_dataset_ready", 28, f"prepared {frames.count} video frames and pinhole calibration", intrinsics.metrics())

    training = run_artdeco_speed3r_training(
        frames=frames,
        intrinsics=intrinsics,
        intrinsics_path=intrinsics_path,
        output_dir=ctx.work_dir / "artdeco_output",
        model_cache_dir=ctx.model_cache_dir,
        settings=settings,
        options=ctx.options,
        progress=lambda stage, progress, message=None, metrics=None: ctx_progress(ctx, stage, progress, message, metrics),
    )
    if not training.gs_ply.exists() or training.gs_ply.stat().st_size <= 0:
        raise FineFailure("ARTIFACT_NOT_FOUND", f"ARTDECO did not create non-empty point_clouds/gs.ply: {training.gs_ply}")

    ctx.final_ply.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(training.gs_ply, ctx.final_ply)

    ctx_progress(ctx, "final_spz_converting", 88, "converting ARTDECO gs.ply to Spark-readable final_web.spz")
    try:
        splat_count = convert_ply_to_spz(ctx.final_ply, ctx.final_spz)
    except PreviewFailure as exc:
        raise FineFailure(exc.code, exc.message) from exc

    warnings: list[str] = []
    lod_rad = lod_builder(ctx)
    if ctx.lod_rad and lod_rad is None:
        warnings.append("RAD LOD builder is not configured; final_lod.rad was not generated.")

    metrics: dict[str, Any] = {
        "pipeline": VIDEO_PIPELINE_NAME,
        "algorithm": "artdeco_vslam_h3dgsv3_speed3r_pi3",
        "source_version": ctx.source_version,
        "source_commits": SOURCE_COMMITS_VIDEO_FINE,
        "input_video": str(ctx.input_video),
        "splat_count": splat_count,
        "final_ply_bytes": ctx.final_ply.stat().st_size,
        "final_spz_bytes": ctx.final_spz.stat().st_size,
        "lod_rad_bytes": lod_rad.stat().st_size if lod_rad else None,
        "warnings": warnings,
        **frames.metrics(),
        **intrinsics.metrics(),
        **training.metrics,
    }
    ctx.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    ctx.metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    ctx_progress(ctx, "fine_outputs_ready", 90, "validated ARTDECO final.ply and final_web.spz", metrics)
    return FineResult(
        final_ply=ctx.final_ply,
        final_spz=ctx.final_spz,
        metrics_json=ctx.metrics_json,
        lod_rad=lod_rad,
        splat_count=splat_count,
        source_commits=SOURCE_COMMITS_VIDEO_FINE,
        metrics=metrics,
    )


def ctx_progress(ctx: FineContext, stage: str, progress: int, message: str | None = None, metrics: dict[str, Any] | None = None) -> None:
    if ctx.progress:
        ctx.progress(stage, progress, message, metrics)
