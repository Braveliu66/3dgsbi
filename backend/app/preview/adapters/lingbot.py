from __future__ import annotations

# LingBot adapter: export video/long-sequence preview as Spark SPZ.
from app.preview.io.spz import convert_ply_to_spz
from app.preview.types import PreviewContext, PreviewResult, SOURCE_COMMITS
from app.preview.utils import StageTimer, require_file


def load_lingbot_runtime():
    from app.preview.vendor.lingbot_runtime import run_lingbot_pointcloud

    return run_lingbot_pointcloud


def run(ctx: PreviewContext) -> PreviewResult:
    timer = StageTimer()
    weight = require_file(
        ctx.model_path("lingbot-map", "lingbot-map-long.pt"),
        "LINGBOT_WEIGHT_MISSING",
        "LingBot-Map weight",
    )
    ply_path = ctx.work_dir / "lingbot" / "recon.ply"
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    realtime = "segment_index" in ctx.options
    offline_video = bool(ctx.input_video and not realtime)
    fps = read_int_option(first_option(ctx.options, "lingbot_fps", "fps"), 5 if realtime or offline_video else 10)
    max_frames = read_int_option(ctx.options.get("lingbot_max_frames"), 96 if realtime else 0)
    confidence_quantile = read_float_option(ctx.options.get("lingbot_confidence_quantile"), 0.8 if offline_video else 0.65)
    max_points = read_int_option(ctx.options.get("preview_max_points"), 0 if offline_video else 15_000_000)

    def report(stage: str, progress: int, message: str, metrics: dict | None = None) -> None:
        ctx.report(stage, progress, message, metrics)

    metrics = load_lingbot_runtime()(
        input_dir=ctx.input_dir,
        input_video=ctx.input_video,
        checkpoint_path=weight,
        output_ply=ply_path,
        fps=fps,
        max_frames=max_frames,
        confidence_quantile=confidence_quantile,
        max_points=max_points,
        progress=report,
        runtime_options={
            **ctx.options,
            "lingbot_compile_cache_dir": str(ctx.model_cache_dir / "torchinductor"),
            "lingbot_input_mode": "realtime_camera" if realtime else "offline_video" if ctx.input_video else "image_sequence",
        },
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
            "output_spz": str(ctx.output_spz),
            "output_spz_size": ctx.output_spz.stat().st_size if ctx.output_spz.exists() else None,
            "intermediate_ply_size": ply_path.stat().st_size,
        },
        source_commits={"LingBot-Map": SOURCE_COMMITS["LingBot-Map"], "Spark": SOURCE_COMMITS["Spark"]},
    )


def read_int_option(value, fallback: int) -> int:
    if value is None or value == "":
        return fallback
    return int(value)


def read_float_option(value, fallback: float) -> float:
    if value is None or value == "":
        return fallback
    return float(value)


def first_option(options: dict, *names: str):
    for name in names:
        value = options.get(name)
        if value is not None and value != "":
            return value
    return None
