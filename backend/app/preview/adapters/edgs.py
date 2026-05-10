from __future__ import annotations

# EDGS adapter：系统管线为 LiteVGGT + EDGS + Spark SPZ。
# v1 中 EDGS 使用其原始 COLMAP/RoMA correspondence 初始化路径训练预览 Gaussian；
# LiteVGGT 直接路径仍独立可选，后续可把 LiteVGGT 相机/点云进一步接入 EDGS init。

from pathlib import Path

from app.preview.io.spz import convert_ply_to_spz
from app.preview.types import PreviewContext, PreviewResult, SOURCE_COMMITS
from app.preview.utils import StageTimer, image_files, require_file
from app.preview.vendor.edgs_runtime import run_edgs_preview
from app.preview.vendor.litevggt_runtime import build_litevggt_colmap_scene


def run(ctx: PreviewContext) -> PreviewResult:
    timer = StageTimer()
    scene_dir = ctx.work_dir / "edgs_scene"
    output_dir = ctx.work_dir / "edgs_output"
    input_count = len(image_files(ctx.input_dir))
    num_ref_views = _read_positive_int(ctx.options.get("edgs_num_ref_views"), _default_ref_views(input_count))
    num_corrs = _read_positive_int(ctx.options.get("edgs_num_corrs_per_view") or ctx.options.get("edgs_matches_per_ref"), 20_000)
    num_steps = _read_positive_int(ctx.options.get("edgs_preview_steps"), 1_500 if input_count < 16 else 3_000)
    nns_per_ref = _read_positive_int(ctx.options.get("edgs_nns_per_ref"), 3)
    scaling_factor = _read_positive_float(ctx.options.get("edgs_scaling_factor"), 0.001)
    litevggt_keep_ratio = _read_positive_float(ctx.options.get("litevggt_keep_ratio"), 0.75)
    litevggt_spatial_keep_quantile = _read_positive_float(ctx.options.get("litevggt_spatial_keep_quantile"), 1.0)
    litevggt_max_points = _read_positive_int(ctx.options.get("litevggt_edgs_max_points") or ctx.options.get("preview_max_points"), 1_000_000)
    litevggt_letterbox_size = _read_positive_int(ctx.options.get("litevggt_letterbox_size"), 518)
    litevggt_max_input_frames = _read_optional_positive_int(ctx.options.get("litevggt_max_input_frames"))
    litevggt_frame_selection = str(ctx.options.get("litevggt_frame_selection") or "scene")
    litevggt_min_scene_change = _read_positive_float(ctx.options.get("litevggt_min_scene_change"), 0.045)
    litevggt_edge_keep_ratio = _read_positive_float(ctx.options.get("litevggt_edge_keep_ratio"), 0.15)
    litevggt_axis_trim_low_quantile = _read_positive_float(ctx.options.get("litevggt_axis_trim_low_quantile"), 0.0005)
    litevggt_axis_trim_high_quantile = _read_positive_float(ctx.options.get("litevggt_axis_trim_high_quantile"), 0.9995)
    litevggt_selection_strategy = str(ctx.options.get("litevggt_point_selection_strategy") or "per_frame")
    litevggt_weight = require_file(ctx.model_path("litevggt", "te_dict.pt"), "LITEVGGT_WEIGHT_MISSING", "LiteVGGT weight")
    roma_weight = require_file(ctx.model_path("roma", "roma_indoor.pth"), "ROMA_WEIGHT_MISSING", "RoMA indoor weight")
    dinov2_weight = require_file(ctx.model_path("roma", "dinov2_vitl14_pretrain.pth"), "DINOV2_WEIGHT_MISSING", "DINOv2 ViT-L/14 weight")

    def report(stage: str, progress: int, message: str) -> None:
        ctx.report(stage, progress, message)

    scene_metrics = build_litevggt_colmap_scene(
        input_dir=ctx.input_dir,
        checkpoint_path=litevggt_weight,
        scene_dir=scene_dir,
        keep_ratio=litevggt_keep_ratio,
        max_points=litevggt_max_points,
        spatial_keep_quantile=litevggt_spatial_keep_quantile,
        letterbox_size=litevggt_letterbox_size,
        max_input_frames=litevggt_max_input_frames,
        frame_selection=litevggt_frame_selection,
        min_scene_change=litevggt_min_scene_change,
        edge_keep_ratio=litevggt_edge_keep_ratio,
        axis_trim_low_quantile=litevggt_axis_trim_low_quantile,
        axis_trim_high_quantile=litevggt_axis_trim_high_quantile,
        selection_strategy=litevggt_selection_strategy,
        progress=report,
    )
    timer.mark("litevggt_scene_init")

    result = run_edgs_preview(
        scene_dir=scene_dir,
        output_dir=output_dir,
        num_ref_views=num_ref_views,
        num_corrs_per_view=num_corrs,
        num_steps=num_steps,
        roma_weight=roma_weight,
        dinov2_weight=dinov2_weight,
        progress=report,
        nns_per_ref=nns_per_ref,
        scaling_factor=scaling_factor,
    )
    timer.mark("edgs_training")
    ply_path = Path(result["ply_path"])

    ctx.report("spz_conversion", 88, "converting EDGS Gaussian PLY to Spark SPZ")
    splat_count = convert_ply_to_spz(ply_path, ctx.output_spz)
    timer.mark("spz_conversion")

    metrics = {**scene_metrics, **{key: value for key, value in result.items() if key != "ply_path"}}
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
        source_commits={"LiteVGGT": SOURCE_COMMITS["LiteVGGT"], "EDGS": SOURCE_COMMITS["EDGS"], "Spark": SOURCE_COMMITS["Spark"]},
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


def _read_optional_positive_int(value) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
