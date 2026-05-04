from __future__ import annotations

# EDGS adapter：系统管线为 LiteVGGT + EDGS + Spark SPZ。
# v1 中 EDGS 使用其原始 COLMAP/RoMA correspondence 初始化路径训练预览 Gaussian；
# LiteVGGT 直接路径仍独立可选，后续可把 LiteVGGT 相机/点云进一步接入 EDGS init。

from pathlib import Path

from app.preview.io.spz import convert_ply_to_spz
from app.preview.types import PreviewContext, PreviewResult, SOURCE_COMMITS
from app.preview.utils import StageTimer, require_file
from app.preview.vendor.edgs_runtime import run_edgs_preview


def run(ctx: PreviewContext) -> PreviewResult:
    timer = StageTimer()
    scene_dir = ctx.work_dir / "edgs_scene"
    output_dir = ctx.work_dir / "edgs_output"
    num_ref_views = int(ctx.options.get("edgs_num_ref_views") or 16)
    num_corrs = int(ctx.options.get("edgs_num_corrs_per_view") or 20_000)
    num_steps = int(ctx.options.get("edgs_preview_steps") or 1_000)
    max_size = int(ctx.options.get("edgs_max_image_size") or 1024)
    roma_weight = require_file(ctx.model_path("roma", "roma_indoor.pth"), "ROMA_WEIGHT_MISSING", "RoMA indoor weight")
    dinov2_weight = require_file(ctx.model_path("roma", "dinov2_vitl14_pretrain.pth"), "DINOV2_WEIGHT_MISSING", "DINOv2 ViT-L/14 weight")

    def report(stage: str, progress: int, message: str) -> None:
        ctx.report(stage, progress, message)

    result = run_edgs_preview(
        input_dir=ctx.input_dir,
        scene_dir=scene_dir,
        output_dir=output_dir,
        num_ref_views=num_ref_views,
        num_corrs_per_view=num_corrs,
        num_steps=num_steps,
        max_size=max_size,
        roma_weight=roma_weight,
        dinov2_weight=dinov2_weight,
        progress=report,
    )
    timer.mark("edgs_training")
    ply_path = Path(result["ply_path"])

    ctx.report("spz_conversion", 88, "converting EDGS Gaussian PLY to Spark SPZ")
    splat_count = convert_ply_to_spz(ply_path, ctx.output_spz)
    timer.mark("spz_conversion")

    metrics = {key: value for key, value in result.items() if key != "ply_path"}
    return PreviewResult(
        output_spz=ctx.output_spz,
        intermediate_ply=ply_path,
        splat_count=splat_count,
        metrics={
            **metrics,
            **timer.metrics(),
            "adapter": "litevggt_edgs",
            "intermediate_ply_size": ply_path.stat().st_size,
            "license_notice": "EDGS is limited to non-commercial academic/personal use.",
        },
        source_commits={"EDGS": SOURCE_COMMITS["EDGS"], "Spark": SOURCE_COMMITS["Spark"]},
    )
