from __future__ import annotations

from pathlib import Path
from typing import Any

from app.fine.types import FineFailure
from app.fine.video.types import CameraIntrinsics


def resolve_video_intrinsics(width: int, height: int, options: dict[str, Any]) -> CameraIntrinsics:
    explicit = options.get("fine_video_intrinsics") or options.get("camera_intrinsics") or {}
    if isinstance(explicit, dict):
        values = {
            "fx": explicit.get("fx"),
            "fy": explicit.get("fy"),
            "cx": explicit.get("cx"),
            "cy": explicit.get("cy"),
        }
        if all(value is not None for value in values.values()):
            try:
                fx = float(values["fx"])
                fy = float(values["fy"])
                cx = float(values["cx"])
                cy = float(values["cy"])
            except (TypeError, ValueError) as exc:
                raise FineFailure("INVALID_INTRINSICS", "Video intrinsics must be numeric fx/fy/cx/cy values") from exc
            if fx <= 0 or fy <= 0:
                raise FineFailure("INVALID_INTRINSICS", "Video intrinsics fx/fy must be positive")
            return CameraIntrinsics(width, height, fx, fy, cx, cy, "user", False)

    focal = 0.9 * max(width, height)
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=focal,
        fy=focal,
        cx=width / 2.0,
        cy=height / 2.0,
        source="default_pinhole_0.9max_center",
        optimize_focal=True,
    )


def write_artdeco_intrinsics(path: Path, intrinsics: CameraIntrinsics) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"width: {intrinsics.width}",
            f"height: {intrinsics.height}",
            f"calibration:  [{intrinsics.fx}, {intrinsics.fy}, {intrinsics.cx}, {intrinsics.cy}]",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return path
