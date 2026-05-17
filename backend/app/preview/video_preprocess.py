from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.preview.image_preprocess import normalize_image_directory
from app.preview.types import PreviewFailure
from app.preview.utils import image_files


@dataclass(slots=True)
class PreviewVideoPreprocessResult:
    output_dir: Path
    metrics: dict[str, Any]


def preprocess_preview_video(
    video_path: Path,
    work_dir: Path,
    *,
    scene_type: str,
    max_side: int,
    jpeg_quality: int,
    fps: float | None = None,
    max_frames: int | None = None,
    min_frames: int = 8,
) -> PreviewVideoPreprocessResult:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PreviewFailure("FFMPEG_UNAVAILABLE", "ffmpeg executable was not found in PATH")
    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise PreviewFailure("VIDEO_INPUT_NOT_FOUND", f"video file not found: {video_path}")

    scene = "outdoor" if str(scene_type).strip().lower() == "outdoor" else "indoor"
    resolved_fps = _read_positive_float(fps, 1.0 if scene == "outdoor" else 2.0)
    resolved_max_frames = _read_positive_int(max_frames, 300)

    raw_dir = work_dir / "video_frames_raw"
    selected_dir = work_dir / "video_frames_selected"
    normalized_dir = work_dir / "video_frames_normalized"
    for path in (raw_dir, selected_dir, normalized_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={resolved_fps:.6g}",
        "-q:v",
        "2",
        str(raw_dir / "%06d.jpg"),
    ]
    print("[preview-video] command " + " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    print(completed.stdout or "", flush=True)
    if completed.returncode != 0:
        raise PreviewFailure("VIDEO_FRAME_EXTRACTION_FAILED", (completed.stdout or "").strip() or "ffmpeg frame extraction failed")

    extracted = image_files(raw_dir)
    if len(extracted) < min_frames:
        raise PreviewFailure(
            "INSUFFICIENT_VIDEO_FRAMES",
            f"video extraction produced {len(extracted)} frames; at least {min_frames} are required",
        )

    selected = select_uniform_frames(extracted, resolved_max_frames)
    for index, source in enumerate(selected):
        shutil.copy2(source, selected_dir / f"{index:06d}.jpg")

    normalized = normalize_image_directory(
        selected_dir,
        normalized_dir,
        max_side=max_side,
        jpeg_quality=jpeg_quality,
    )
    metrics = {
        "preview_input_type": "video",
        "video_preprocess_source_path": str(video_path),
        "video_preprocess_sampled_fps": resolved_fps,
        "video_preprocess_max_frames": resolved_max_frames,
        "video_preprocess_extracted_frames": len(extracted),
        "video_preprocess_selected_frames": len(selected),
        **normalized.metrics(),
    }
    return PreviewVideoPreprocessResult(output_dir=normalized.output_dir, metrics=metrics)


def select_uniform_frames(paths: list[Path], max_frames: int) -> list[Path]:
    ordered = sorted(paths, key=lambda path: path.name)
    if max_frames <= 0 or len(ordered) <= max_frames:
        return ordered
    if max_frames == 1:
        return [ordered[0]]
    last = len(ordered) - 1
    indices = sorted({round(index * last / (max_frames - 1)) for index in range(max_frames)})
    return [ordered[index] for index in indices]


def _read_positive_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _read_positive_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback
