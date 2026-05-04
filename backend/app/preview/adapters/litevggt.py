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
    keep_ratio = float(ctx.options.get("litevggt_keep_ratio") or 0.42)
    max_points = int(ctx.options.get("preview_max_points") or 15_000_000)

    def report(stage: str, progress: int, message: str) -> None:
        ctx.report(stage, progress, message)

    metrics = run_litevggt_pointcloud(
        input_dir=ctx.input_dir,
        checkpoint_path=weight,
        output_ply=ply_path,
        keep_ratio=keep_ratio,
        max_points=max_points,
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

