from __future__ import annotations

from app.preview.adapters import edgs, lingbot, litevggt
from app.preview.types import PreviewContext, PreviewFailure, PreviewResult


def run_preview_pipeline(ctx: PreviewContext) -> PreviewResult:
    """按标准管线名分发到内置 adapter。"""

    if ctx.pipeline == "litevggt_spz":
        return litevggt.run(ctx)
    if ctx.pipeline == "litevggt_edgs":
        return edgs.run(ctx)
    if ctx.pipeline == "lingbot_spz":
        return lingbot.run(ctx)
    raise PreviewFailure("UNKNOWN_PREVIEW_PIPELINE", f"unsupported preview pipeline: {ctx.pipeline}")

