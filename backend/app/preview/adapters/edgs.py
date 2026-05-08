from __future__ import annotations

# EDGS adapter：系统管线为 LiteVGGT + EDGS + Spark SPZ。
# v1 中 EDGS 使用其原始 COLMAP/RoMA correspondence 初始化路径训练预览 Gaussian；
# LiteVGGT 直接路径仍独立可选，后续可把 LiteVGGT 相机/点云进一步接入 EDGS init。

from pathlib import Path

from app.preview.io.spz import convert_ply_to_spz
from app.preview.types import PreviewContext, PreviewResult, SOURCE_COMMITS
from app.preview.utils import StageTimer, image_files, require_file
from app.preview.vendor.edgs_runtime import run_edgs_preview


def run(ctx: PreviewContext) -> PreviewResult:
    timer = StageTimer()
    scene_dir = ctx.work_dir / "edgs_scene"
    output_dir = ctx.work_dir / "edgs_output"
    input_count = len(image_files(ctx.input_dir))
    num_ref_views = _read_positive_int(ctx.options.get("edgs_num_ref_views"), _default_ref_views(input_count))
    num_corrs = _read_positive_int(ctx.options.get("edgs_num_corrs_per_view") or ctx.options.get("edgs_matches_per_ref"), 20_000)
    num_steps = _read_positive_int(ctx.options.get("edgs_preview_steps"), 1_500 if input_count < 16 else 3_000)
    max_size = _read_positive_int(ctx.options.get("edgs_max_image_size"), 1024)
    selected_count = max(1, min(num_ref_views, input_count))
    nns_per_ref = _read_positive_int(ctx.options.get("edgs_nns_per_ref"), 3)
    scaling_factor = _read_positive_float(ctx.options.get("edgs_scaling_factor"), 0.001)
    colmap_max_features = _read_positive_int(ctx.options.get("edgs_colmap_max_num_features"), 4096)
    colmap_max_size = _read_positive_int(ctx.options.get("edgs_colmap_max_image_size"), 1024)
    colmap_min_model_size = _read_positive_int(ctx.options.get("edgs_colmap_min_model_size"), max(3, min(10, selected_count)))
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
        colmap_max_num_features=colmap_max_features,
        colmap_max_image_size=colmap_max_size,
        colmap_min_model_size=colmap_min_model_size,
        nns_per_ref=nns_per_ref,
        scaling_factor=scaling_factor,
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


def _default_ref_views(image_count: int) -> int:
    if image_count <= 24:
        return max(1, image_count)
    if image_count <= 96:
        return 64
    return 96


def _read_positive_int(value, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _read_positive_float(value, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback
