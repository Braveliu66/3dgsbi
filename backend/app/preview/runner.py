from __future__ import annotations

from app.preview.adapters import lingbot_pointcloud, litevggt
from app.preview.types import PreviewContext, PreviewFailure, PreviewResult


def run_preview_pipeline(ctx: PreviewContext) -> PreviewResult:
    if ctx.pipeline == "litevggt_spz":
        return litevggt.run(ctx)

    if ctx.pipeline in {
        "lingbot",
        "lingbot_map",
        "lingbot_map_spz",
        "video_lingbot",
        "lingbot_video_pointcloud_fast",
        "lingbot_pointcloud",
    }:
        return lingbot_pointcloud.run(ctx)

    raise PreviewFailure("UNKNOWN_PREVIEW_PIPELINE", f"unsupported preview pipeline: {ctx.pipeline}")
