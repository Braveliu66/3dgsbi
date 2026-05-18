from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.preview.image_preprocess import normalize_image_directory
from app.preview.types import PreviewFailure
from app.preview.utils import image_files


PREVIEW_VIDEO_SPEED_MAX_FRAMES = 64
PREVIEW_VIDEO_SPEED_FALLBACK_FPS = 2.0


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

    resolved_max_frames = max(min_frames, _read_positive_int(max_frames, PREVIEW_VIDEO_SPEED_MAX_FRAMES))
    duration_seconds = _read_video_duration_seconds(video_path)
    resolved_fps = _resolve_video_sample_fps(fps, duration_seconds, resolved_max_frames, min_frames)

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
        "video_preprocess_duration_seconds": duration_seconds,
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


def _resolve_video_sample_fps(
    value: Any,
    duration_seconds: float | None,
    max_frames: int,
    min_frames: int,
) -> float:
    if value is not None:
        return _read_positive_float(value, PREVIEW_VIDEO_SPEED_FALLBACK_FPS)
    if duration_seconds is None or duration_seconds <= 0:
        return PREVIEW_VIDEO_SPEED_FALLBACK_FPS
    min_fps = max(0.001, float(min_frames) / duration_seconds)
    target_fps = max(0.001, float(max_frames) / duration_seconds)
    return max(min_fps, min(PREVIEW_VIDEO_SPEED_FALLBACK_FPS, target_fps))


def _read_video_duration_seconds(video_path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    try:
        duration = float((completed.stdout or "").strip().splitlines()[0])
    except (IndexError, ValueError):
        return None
    return duration if duration > 0 else None


def _read_positive_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback
