from __future__ import annotations

# LiteVGGT adapter：负责系统参数、权重路径、进度映射和 SPZ 转换。
# 真正的模型推理在 preview.vendor.litevggt_runtime 中，避免 worker 直接拼命令。

from app.preview.io.spz import convert_ply_to_spz
from app.preview.types import PreviewContext, PreviewResult, SOURCE_COMMITS
from app.preview.utils import StageTimer, require_file
from app.preview.vendor.litevggt_runtime import run_litevggt_pointcloud


def run(ctx: PreviewContext) -> PreviewResult:
    timer = StageTimer()
    weight = require_file(
        ctx.model_path("litevggt", "te_dict.pt"),
        "LITEVGGT_WEIGHT_MISSING",
        "LiteVGGT weight",
    )
    ply_path = ctx.work_dir / "litevggt" / "recon.ply"

    coverage_mode = str(ctx.options.get("litevggt_coverage_mode") or "complete")
    if coverage_mode == "complete":
        keep_ratio = float(ctx.options.get("litevggt_keep_ratio") or 0.95)
        spatial_keep_quantile = float(ctx.options.get("litevggt_spatial_keep_quantile") or 1.0)
        preserve_full_image = bool(ctx.options.get("litevggt_preserve_full_image", True))
        frame_selection = str(ctx.options.get("litevggt_frame_selection") or "scene")
        max_points = int(ctx.options.get("preview_max_points") or 25_000_000)
    elif coverage_mode == "balanced":
        keep_ratio = float(ctx.options.get("litevggt_keep_ratio") or 0.75)
        spatial_keep_quantile = float(ctx.options.get("litevggt_spatial_keep_quantile") or 0.999)
        preserve_full_image = bool(ctx.options.get("litevggt_preserve_full_image", True))
        frame_selection = str(ctx.options.get("litevggt_frame_selection") or "scene")
        max_points = int(ctx.options.get("preview_max_points") or 15_000_000)
    else:
        keep_ratio = float(ctx.options.get("litevggt_keep_ratio") or 0.42)
        spatial_keep_quantile = float(ctx.options.get("litevggt_spatial_keep_quantile") or 0.995)
        preserve_full_image = bool(ctx.options.get("litevggt_preserve_full_image", False))
        frame_selection = str(ctx.options.get("litevggt_frame_selection") or "head")
        max_points = int(ctx.options.get("preview_max_points") or 10_000_000)

    letterbox_size = int(ctx.options.get("litevggt_letterbox_size") or 518)

    max_input_frames_value = ctx.options.get("litevggt_max_input_frames")
    max_input_frames = int(max_input_frames_value) if max_input_frames_value else None

    min_scene_change = float(ctx.options.get("litevggt_min_scene_change") or 0.045)
    edge_keep_ratio = float(ctx.options.get("litevggt_edge_keep_ratio") or 0.15)
    axis_trim_low_quantile = float(ctx.options.get("litevggt_axis_trim_low_quantile") or 0.0005)
    axis_trim_high_quantile = float(ctx.options.get("litevggt_axis_trim_high_quantile") or 0.9995)
    selection_strategy = str(ctx.options.get("litevggt_point_selection_strategy") or "per_frame")

    def report(stage: str, progress: int, message: str) -> None:
        ctx.report(stage, progress, message)

    metrics = run_litevggt_pointcloud(
        input_dir=ctx.input_dir,
        checkpoint_path=weight,
        output_ply=ply_path,
        keep_ratio=keep_ratio,
        max_points=max_points,
        spatial_keep_quantile=spatial_keep_quantile,
        preserve_full_image=preserve_full_image,
        letterbox_size=letterbox_size,
        max_input_frames=max_input_frames,
        frame_selection=frame_selection,
        min_scene_change=min_scene_change,
        edge_keep_ratio=edge_keep_ratio,
        axis_trim_low_quantile=axis_trim_low_quantile,
        axis_trim_high_quantile=axis_trim_high_quantile,
        selection_strategy=selection_strategy,
        progress=report,
    )
    timer.mark("litevggt_inference")
    ctx.report("spz_conversion", 86, "converting LiteVGGT point cloud PLY to Spark SPZ")
    splat_count = convert_ply_to_spz(ply_path, ctx.output_spz)
    timer.mark("spz_conversion")

    return PreviewResult(
        output_spz=ctx.output_spz,
        intermediate_ply=ply_path,
        splat_count=splat_count,
        metrics={
            **metrics,
            **timer.metrics(),
            "adapter": "litevggt_spz",
            "intermediate_ply_size": ply_path.stat().st_size,
        },
        source_commits={"LiteVGGT": SOURCE_COMMITS["LiteVGGT"], "Spark": SOURCE_COMMITS["Spark"]},
    )
