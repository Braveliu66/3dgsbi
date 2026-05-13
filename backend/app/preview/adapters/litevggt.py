from __future__ import annotations

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

    keep_ratio_value = ctx.options.get("litevggt_keep_ratio")
    keep_ratio = float(keep_ratio_value) if keep_ratio_value is not None else None
    max_points = int(ctx.options.get("preview_max_points") or 3_000_000)
    max_input_frames_value = ctx.options.get("litevggt_max_input_frames")
    max_input_frames = int(max_input_frames_value) if max_input_frames_value else None
    target_size_value = ctx.options.get("litevggt_target_size")
    target_size = int(target_size_value) if target_size_value else None
    params: dict[str, Any] = {
        "keep_ratio": keep_ratio if keep_ratio is not None else "auto",
        "max_points": max_points,
        "max_input_frames": max_input_frames,
        "target_size": target_size if target_size is not None else "auto",
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
        f"LiteVGGT official single path: images={len(files)} keep_ratio={params['keep_ratio']} max_points={max_points}",
    )

    def report(stage: str, progress: int, message: str) -> None:
        ctx.report(stage, progress, message)

    metrics = run_litevggt_pointcloud(
        input_dir=ctx.input_dir,
        checkpoint_path=weight,
        output_ply=ply_path,
        keep_ratio=keep_ratio,
        max_points=max_points,
        max_input_frames=max_input_frames,
        target_size=target_size,
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


def _first_names(paths, limit: int = 8) -> str:
    names = [path.name for path in paths[:limit]]
    suffix = "" if len(paths) <= limit else f", ... +{len(paths) - limit}"
    return "[" + ", ".join(names) + suffix + "]"


def _format_metrics(metrics: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in sorted(metrics.items()))
