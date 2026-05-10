from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ProgressCallback = Callable[[str, int, str | None, dict[str, Any] | None], None]


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    source: str
    optimize_focal: bool

    def metrics(self) -> dict[str, Any]:
        return {
            "intrinsics_source": self.source,
            "intrinsics_width": self.width,
            "intrinsics_height": self.height,
            "intrinsics_fx": self.fx,
            "intrinsics_fy": self.fy,
            "intrinsics_cx": self.cx,
            "intrinsics_cy": self.cy,
            "artdeco_optimize_focal": self.optimize_focal,
        }


@dataclass(frozen=True, slots=True)
class ExtractedVideoFrames:
    frames_dir: Path
    dataset_root: Path
    count: int
    width: int
    height: int
    fps: float | None
    source_video: Path

    def metrics(self) -> dict[str, Any]:
        return {
            "video_frames": self.count,
            "video_frame_width": self.width,
            "video_frame_height": self.height,
            "video_source_fps": self.fps,
        }


@dataclass(frozen=True, slots=True)
class ArtdecoTrainingResult:
    output_dir: Path
    gs_ply: Path
    metrics: dict[str, Any]
