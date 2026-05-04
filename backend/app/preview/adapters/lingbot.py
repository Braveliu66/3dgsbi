from __future__ import annotations

# LingBot adapter：负责视频/长序列预览，输出统一 Spark SPZ。

from app.preview.io.spz import convert_ply_to_spz
from app.preview.types import PreviewContext, PreviewResult, SOURCE_COMMITS
from app.preview.utils import StageTimer, require_file
from app.preview.vendor.lingbot_runtime import run_lingbot_pointcloud


def run(ctx: PreviewContext) -> PreviewResult:
    timer = StageTimer()
    weight = require_file(
        ctx.model_path("lingbot-map", "lingbot-map-long.pt"),
        "LINGBOT_WEIGHT_MISSING",
        "LingBot-Map weight",
    )
    ply_path = ctx.work_dir / "lingbot" / "recon.ply"
    fps = int(ctx.options.get("lingbot_fps") or 10)
    max_frames = int(ctx.options.get("lingbot_max_frames") or 512)
    confidence_quantile = float(ctx.options.get("lingbot_confidence_quantile") or 0.65)
    max_points = int(ctx.options.get("preview_max_points") or 15_000_000)

    def report(stage: str, progress: int, message: str) -> None:
        ctx.report(stage, progress, message)

    metrics = run_lingbot_pointcloud(
        input_dir=ctx.input_dir,
        input_video=ctx.input_video,
        checkpoint_path=weight,
        output_ply=ply_path,
        fps=fps,
        max_frames=max_frames,
        confidence_quantile=confidence_quantile,
        max_points=max_points,
        progress=report,
    )
    timer.mark("lingbot_inference")
    ctx.report("spz_conversion", 88, "converting LingBot point cloud PLY to Spark SPZ")
    splat_count = convert_ply_to_spz(ply_path, ctx.output_spz)
    timer.mark("spz_conversion")

    return PreviewResult(
        output_spz=ctx.output_spz,
        intermediate_ply=ply_path,
        splat_count=splat_count,
        metrics={
            **metrics,
            **timer.metrics(),
            "adapter": "lingbot_spz",
            "intermediate_ply_size": ply_path.stat().st_size,
        },
        source_commits={"LingBot-Map": SOURCE_COMMITS["LingBot-Map"], "Spark": SOURCE_COMMITS["Spark"]},
    )

