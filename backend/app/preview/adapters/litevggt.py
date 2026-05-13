from __future__ import annotations

# LiteVGGT adapter：负责系统参数、权重路径、进度映射和 SPZ 转换。
# 真正的模型推理在 preview.vendor.litevggt_runtime 中，避免 worker 直接拼命令。

from typing import Any

from app.preview.io.spz import convert_ply_to_spz
from app.preview.types import PreviewContext, PreviewResult, SOURCE_COMMITS
from app.preview.utils import StageTimer, image_files, require_file
from app.preview.vendor.litevggt_runtime import run_litevggt_pointcloud


def run(ctx: PreviewContext) -> PreviewResult:
    timer = StageTimer()
    weight = require_file(
        ctx.model_path("litevggt", "te_dict.pt"),
        "LITEVGGT_WEIGHT_MISSING",
        "LiteVGGT weight",
    )
    ply_path = ctx.work_dir / "litevggt" / "recon.ply"
    files = image_files(ctx.input_dir)

    coverage_mode = str(ctx.options.get("litevggt_coverage_mode") or "complete")
    if coverage_mode == "complete":
        keep_ratio = float(ctx.options.get("litevggt_keep_ratio") or 0.42)
        spatial_keep_quantile = float(ctx.options.get("litevggt_spatial_keep_quantile") or 1.0)
        preserve_full_image = _read_bool(ctx.options.get("litevggt_preserve_full_image"), True)
        frame_selection = str(ctx.options.get("litevggt_frame_selection") or "scene")
        max_points = int(ctx.options.get("preview_max_points") or 3_000_000)
    elif coverage_mode == "balanced":
        keep_ratio = float(ctx.options.get("litevggt_keep_ratio") or 0.75)
        spatial_keep_quantile = float(ctx.options.get("litevggt_spatial_keep_quantile") or 0.999)
        preserve_full_image = _read_bool(ctx.options.get("litevggt_preserve_full_image"), True)
        frame_selection = str(ctx.options.get("litevggt_frame_selection") or "scene")
        max_points = int(ctx.options.get("preview_max_points") or 2_000_000)
    else:
        keep_ratio = float(ctx.options.get("litevggt_keep_ratio") or 0.42)
        spatial_keep_quantile = float(ctx.options.get("litevggt_spatial_keep_quantile") or 0.995)
        preserve_full_image = _read_bool(ctx.options.get("litevggt_preserve_full_image"), True)
        frame_selection = str(ctx.options.get("litevggt_frame_selection") or "head")
        max_points = int(ctx.options.get("preview_max_points") or 1_000_000)

    letterbox_size = int(ctx.options.get("litevggt_letterbox_size") or 518)
    window_size = int(ctx.options.get("litevggt_window_size") or 48)
    window_overlap = int(ctx.options.get("litevggt_window_overlap") or 16)
    oom_window_sizes = _read_int_list(ctx.options.get("litevggt_oom_window_sizes"), [32, 16, 8])

    max_input_frames_value = ctx.options.get("litevggt_max_input_frames")
    max_input_frames = int(max_input_frames_value) if max_input_frames_value else None

    min_scene_change = float(ctx.options.get("litevggt_min_scene_change") or 0.045)
    edge_keep_ratio = float(ctx.options.get("litevggt_edge_keep_ratio") or 0.0)
    axis_trim_low_quantile = float(ctx.options.get("litevggt_axis_trim_low_quantile") or 0.0005)
    axis_trim_high_quantile = float(ctx.options.get("litevggt_axis_trim_high_quantile") or 0.9995)
    selection_strategy = str(ctx.options.get("litevggt_point_selection_strategy") or "global")
    inference_mode = str(ctx.options.get("litevggt_inference_mode") or "auto")
    single_frame_limit = int(ctx.options.get("litevggt_single_frame_limit") or 192)
    global_keyframe_count = int(ctx.options.get("litevggt_global_keyframe_count") or 192)
    hierarchical_enable = _read_bool(ctx.options.get("litevggt_hierarchical_enable"), False)
    chunk_size = int(ctx.options.get("litevggt_chunk_size") or 64)
    chunk_overlap = int(ctx.options.get("litevggt_chunk_overlap") or 16)
    anchor_count = int(ctx.options.get("litevggt_anchor_count") or 8)
    alignment_max_rel_median = float(ctx.options.get("litevggt_alignment_max_rel_median") or 0.05)
    alignment_max_rel_p90 = float(ctx.options.get("litevggt_alignment_max_rel_p90") or 0.12)
    alignment_min_scale = float(ctx.options.get("litevggt_alignment_min_scale") or 0.25)
    alignment_max_scale = float(ctx.options.get("litevggt_alignment_max_scale") or 4.0)
    params: dict[str, Any] = {
        "coverage_mode": coverage_mode,
        "keep_ratio": keep_ratio,
        "spatial_keep_quantile": spatial_keep_quantile,
        "preserve_full_image": preserve_full_image,
        "frame_selection": frame_selection,
        "max_points": max_points,
        "letterbox_size": letterbox_size,
        "window_size": window_size,
        "window_overlap": window_overlap,
        "oom_window_sizes": oom_window_sizes,
        "max_input_frames": max_input_frames,
        "min_scene_change": min_scene_change,
        "edge_keep_ratio": edge_keep_ratio,
        "axis_trim_low_quantile": axis_trim_low_quantile,
        "axis_trim_high_quantile": axis_trim_high_quantile,
        "selection_strategy": selection_strategy,
        "inference_mode": inference_mode,
        "single_frame_limit": single_frame_limit,
        "global_keyframe_count": global_keyframe_count,
        "hierarchical_enable": hierarchical_enable,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "anchor_count": anchor_count,
        "alignment_max_rel_median": alignment_max_rel_median,
        "alignment_max_rel_p90": alignment_max_rel_p90,
        "alignment_min_scale": alignment_min_scale,
        "alignment_max_scale": alignment_max_scale,
    }
    print(
        "[litevggt-preview] adapter params "
        f"task_id={ctx.task_id} project_id={ctx.project_id} input_dir={ctx.input_dir} "
        f"image_count={len(files)} first_images={_first_names(files)} weight={weight} "
        f"weight_bytes={weight.stat().st_size} output_ply={ply_path} output_spz={ctx.output_spz} "
        + " ".join(f"{key}={value}" for key, value in params.items()),
        flush=True,
    )
    ctx.report(
        "litevggt_preflight",
        22,
        (
            "LiteVGGT params: "
            f"images={len(files)} coverage={coverage_mode} keep_ratio={keep_ratio} "
            f"max_points={max_points} window={window_size}/{window_overlap} "
            f"frame_selection={frame_selection}"
        ),
    )

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
        inference_mode=inference_mode,
        single_frame_limit=single_frame_limit,
        global_keyframe_count=global_keyframe_count,
        hierarchical_enable=hierarchical_enable,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        anchor_count=anchor_count,
        alignment_max_rel_median=alignment_max_rel_median,
        alignment_max_rel_p90=alignment_max_rel_p90,
        alignment_min_scale=alignment_min_scale,
        alignment_max_scale=alignment_max_scale,
        window_size=window_size,
        window_overlap=window_overlap,
        oom_window_sizes=oom_window_sizes,
        progress=report,
    )
    timer.mark("litevggt_inference")
    print(
        "[litevggt-preview] inference metrics "
        f"task_id={ctx.task_id} output_ply={ply_path} "
        f"ply_exists={ply_path.exists()} ply_bytes={ply_path.stat().st_size if ply_path.exists() else None} "
        f"metrics={_format_metrics(metrics)}",
        flush=True,
    )
    ctx.report("spz_conversion", 86, "converting LiteVGGT point cloud PLY to Spark SPZ")
    splat_count = convert_ply_to_spz(ply_path, ctx.output_spz)
    timer.mark("spz_conversion")
    print(
        "[litevggt-preview] conversion complete "
        f"task_id={ctx.task_id} output_spz={ctx.output_spz} "
        f"spz_exists={ctx.output_spz.exists()} spz_bytes={ctx.output_spz.stat().st_size if ctx.output_spz.exists() else None} "
        f"splat_count={splat_count} stage_durations={timer.metrics().get('stage_durations')}",
        flush=True,
    )

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


def _read_bool(value, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return fallback


def _read_int_list(value, fallback: list[int]) -> list[int]:
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = str(value).replace(";", ",").split(",")

    parsed: list[int] = []
    for item in values:
        try:
            parsed.append(int(str(item).strip()))
        except (TypeError, ValueError):
            continue
    return parsed or fallback


def _first_names(paths, limit: int = 8) -> str:
    names = [path.name for path in paths[:limit]]
    suffix = "" if len(paths) <= limit else f", ... +{len(paths) - limit}"
    return "[" + ", ".join(names) + suffix + "]"


def _format_metrics(metrics: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in sorted(metrics.items()))
