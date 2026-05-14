from __future__ import annotations

from typing import Any

from app.preview.io.ply import convert_pointcloud_ply_to_fixed_splat_ply
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
    points_ply_path = ctx.work_dir / "litevggt" / "preview_points.ply"
    splats_ply_path = ctx.work_dir / "litevggt" / "preview_splats.ply"
    meta_path = ctx.work_dir / "litevggt" / "preview_meta.json"
    files = image_files(ctx.input_dir)

    keep_ratio = _read_float(ctx.options.get("litevggt_keep_ratio"), 0.85)
    max_points = _read_int(ctx.options.get("preview_max_points"), 15_000_000)
    max_input_frames_value = ctx.options.get("litevggt_max_input_frames")
    max_input_frames = _read_optional_int(max_input_frames_value) if max_input_frames_value is not None else 64
    target_size = _read_int(ctx.options.get("litevggt_target_size"), 518)
    frame_stride = _read_optional_int(ctx.options.get("litevggt_frame_stride"))
    depth_conf_thresh = _read_optional_float(ctx.options.get("litevggt_depth_conf_thresh"), None)
    preprocess_mode = str(ctx.options.get("litevggt_preprocess_mode") or "pad")
    point_selection_strategy = str(ctx.options.get("litevggt_point_selection_strategy") or "scene_coverage")
    axis_trim_low_quantile = _read_float(ctx.options.get("litevggt_axis_trim_low_quantile"), 0.0)
    axis_trim_high_quantile = _read_float(ctx.options.get("litevggt_axis_trim_high_quantile"), 1.0)
    spatial_keep_quantile = _read_float(ctx.options.get("litevggt_spatial_keep_quantile"), 1.0)
    params: dict[str, Any] = {
        "keep_ratio": keep_ratio,
        "max_points": max_points,
        "max_input_frames": max_input_frames,
        "target_size": target_size,
        "frame_stride": frame_stride if frame_stride is not None else "auto",
        "depth_conf_thresh": depth_conf_thresh,
        "preprocess_mode": preprocess_mode,
        "point_selection_strategy": point_selection_strategy,
        "axis_trim_low_quantile": axis_trim_low_quantile,
        "axis_trim_high_quantile": axis_trim_high_quantile,
        "spatial_keep_quantile": spatial_keep_quantile,
    }
    print(
        "[litevggt-preview] adapter params "
        f"task_id={ctx.task_id} project_id={ctx.project_id} input_dir={ctx.input_dir} "
        f"image_count={len(files)} first_images={_first_names(files)} weight={weight} "
        f"weight_bytes={weight.stat().st_size} output_points_ply={points_ply_path} "
        f"output_splats_ply={splats_ply_path} output_spz={ctx.output_spz} "
        + " ".join(f"{key}={value}" for key, value in params.items()),
        flush=True,
    )
    ctx.report(
        "litevggt_preflight",
        22,
        f"LiteVGGT official single path: images={len(files)} keep_ratio={params['keep_ratio']} max_points={max_points}",
    )

    def report(stage: str, progress: int, message: str) -> None:
        ctx.report(stage, progress, message)

    metrics = run_litevggt_pointcloud(
        input_dir=ctx.input_dir,
        checkpoint_path=weight,
        output_ply=points_ply_path,
        output_meta_json=meta_path,
        keep_ratio=keep_ratio,
        max_points=max_points,
        max_input_frames=max_input_frames,
        target_size=target_size,
        frame_stride=frame_stride,
        depth_conf_thresh=depth_conf_thresh,
        preprocess_mode=preprocess_mode,
        selection_strategy=point_selection_strategy,
        axis_trim_low_quantile=axis_trim_low_quantile,
        axis_trim_high_quantile=axis_trim_high_quantile,
        spatial_keep_quantile=spatial_keep_quantile,
        progress=report,
    )
    timer.mark("litevggt_inference")
    print(
        "[litevggt-preview] inference metrics "
        f"task_id={ctx.task_id} output_points_ply={points_ply_path} "
        f"points_ply_exists={points_ply_path.exists()} "
        f"points_ply_bytes={points_ply_path.stat().st_size if points_ply_path.exists() else None} "
        f"metrics={_format_metrics(metrics)}",
        flush=True,
    )

    base_point_radius = _read_bounded_float(metrics.get("litevggt_preview_point_radius"), 0.002, 1e-8, 1.0)
    point_radius_scale = _read_bounded_float(ctx.options.get("litevggt_point_radius_scale"), 1.0, 0.1, 20.0)
    point_radius = base_point_radius * point_radius_scale
    ctx.report("splat_conversion", 82, "converting LiteVGGT point cloud PLY to fixed Gaussian PLY")
    converted_splat_count = convert_pointcloud_ply_to_fixed_splat_ply(
        points_ply_path,
        splats_ply_path,
        point_radius=point_radius,
    )
    timer.mark("fixed_splat_conversion")

    ctx.report("spz_conversion", 86, "converting LiteVGGT fixed Gaussian PLY to Spark SPZ")
    splat_count = convert_ply_to_spz(splats_ply_path, ctx.output_spz)
    timer.mark("spz_conversion")
    print(
        "[litevggt-preview] conversion complete "
        f"task_id={ctx.task_id} output_splats_ply={splats_ply_path} output_spz={ctx.output_spz} "
        f"splats_ply_exists={splats_ply_path.exists()} "
        f"splats_ply_bytes={splats_ply_path.stat().st_size if splats_ply_path.exists() else None} "
        f"spz_exists={ctx.output_spz.exists()} spz_bytes={ctx.output_spz.stat().st_size if ctx.output_spz.exists() else None} "
        f"splat_count={splat_count} stage_durations={timer.metrics().get('stage_durations')}",
        flush=True,
    )

    return PreviewResult(
        output_spz=ctx.output_spz,
        intermediate_ply=points_ply_path,
        splat_count=splat_count,
        metrics={
            **metrics,
            **timer.metrics(),
            "adapter": "litevggt_spz",
            "intermediate_ply_size": points_ply_path.stat().st_size,
            "intermediate_points_ply": str(points_ply_path),
            "intermediate_splats_ply": str(splats_ply_path),
            "preview_meta_json": str(meta_path),
            "intermediate_points_ply_size": points_ply_path.stat().st_size,
            "intermediate_splats_ply_size": splats_ply_path.stat().st_size,
            "preview_meta_json_size": meta_path.stat().st_size,
            "fixed_splat_base_point_radius": base_point_radius,
            "fixed_splat_point_radius_scale": point_radius_scale,
            "fixed_splat_point_radius": point_radius,
            "fixed_splat_count": converted_splat_count,
        },
        source_commits={"LiteVGGT": SOURCE_COMMITS["LiteVGGT"], "Spark": SOURCE_COMMITS["Spark"]},
    )


def _first_names(paths, limit: int = 8) -> str:
    names = [path.name for path in paths[:limit]]
    suffix = "" if len(paths) <= limit else f", ... +{len(paths) - limit}"
    return "[" + ", ".join(names) + suffix + "]"


def _format_metrics(metrics: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in sorted(metrics.items()))


def _read_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "auto", "none"}:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _read_int(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _read_float(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _read_bounded_float(value: Any, fallback: float, minimum: float, maximum: float) -> float:
    parsed = _read_float(value, fallback)
    return max(minimum, min(maximum, parsed))


def _read_optional_float(value: Any, fallback: float | None) -> float | None:
    if value is None:
        return fallback
    normalized = str(value).strip().lower()
    if normalized in {"", "auto"}:
        return fallback
    if normalized in {"none", "off", "false"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback
