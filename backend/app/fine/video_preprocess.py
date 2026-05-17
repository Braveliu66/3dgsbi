from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.fine.preprocess import (
    Image,
    ImageOps,
    classify_blur_scores,
    score_blur_images,
    should_reject_for_training,
    summarize_blur_scores,
)
from app.fine.types import FineFailure
from app.preview.utils import image_files


Progress = Callable[[str, int, str], None]


@dataclass(slots=True)
class FineVideoPreprocessResult:
    output_dir: Path
    metrics: dict[str, Any]


def preprocess_fine_video(
    video_path: Path,
    output_dir: Path,
    *,
    scene_type: str,
    quality_mode: str = "auto",
    min_frames: int = 8,
    progress: Progress | None = None,
) -> FineVideoPreprocessResult:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FineFailure("FFMPEG_UNAVAILABLE", "ffmpeg executable was not found in PATH")
    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise FineFailure("VIDEO_INPUT_NOT_FOUND", f"video file not found: {video_path}")

    duration = probe_video_duration_seconds(video_path)
    fps, max_side = choose_video_sampling(scene_type, duration, quality_mode)
    raw_dir = output_dir.with_name(f"{output_dir.name}_raw")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        progress("fine_video_extracting", 12, f"extracting video frames at {fps:.3g} fps")
    vf = f"fps={fps:.6g},scale='min({max_side},iw)':-2"
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-q:v",
        "2",
        str(raw_dir / "image_%06d.jpg"),
    ]
    print("[fine-video] command " + " ".join(command), flush=True)
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
        raise FineFailure("VIDEO_FRAME_EXTRACTION_FAILED", (completed.stdout or "").strip() or "ffmpeg frame extraction failed")

    extracted = image_files(raw_dir)
    if len(extracted) < min_frames:
        raise FineFailure(
            "INSUFFICIENT_VIDEO_FRAMES",
            f"video extraction produced {len(extracted)} frames; at least {min_frames} are required",
        )

    if progress:
        progress("fine_video_filtering", 14, f"filtering {len(extracted)} extracted video frames")
    kept, filter_metrics = filter_video_frames(raw_dir, output_dir, min_frames=min_frames)
    if len(kept) < min_frames:
        raise FineFailure(
            "INSUFFICIENT_VIDEO_FRAMES",
            f"video preprocessing kept {len(kept)} usable frames; at least {min_frames} are required",
        )

    metrics = {
        "fine_input_type": "video",
        "video_source_path": str(video_path),
        "video_source_duration_seconds": round(duration, 3) if duration > 0 else None,
        "video_sampled_fps": fps,
        "video_frame_max_side": max_side,
        "video_extracted_frames": len(extracted),
        "video_kept_frames": len(kept),
        "video_duplicate_frames_removed": filter_metrics["duplicate_frames_removed"],
        "video_quality_frames_removed": filter_metrics["quality_frames_removed"],
        "video_frame_filter_metrics": filter_metrics,
    }
    if progress:
        progress("fine_video_ready", 16, f"prepared {len(kept)} video frames for COLMAP fine reconstruction")
    return FineVideoPreprocessResult(output_dir=output_dir, metrics=metrics)


def choose_video_sampling(scene_type: str, duration_seconds: float, quality_mode: str = "auto") -> tuple[float, int]:
    scene = "outdoor" if scene_type == "outdoor" else "indoor"
    quality = quality_mode if quality_mode in {"quality", "speed"} else ("quality" if scene == "indoor" else "speed")
    if duration_seconds <= 0:
        return (3.0 if scene == "indoor" else 1.5), (2600 if scene == "indoor" else 2200)

    if duration_seconds < 60:
        fps = 3.0 if scene == "indoor" else 2.0
    elif duration_seconds <= 300:
        fps = 2.0 if scene == "indoor" else 1.0
    else:
        target = 1200 if scene == "indoor" else 3000
        fps = target / duration_seconds

    if quality == "quality":
        fps *= 1.15
    elif quality == "speed":
        fps *= 0.85
    fps = max(0.25, min(4.0 if scene == "indoor" else 3.0, fps))
    max_side = 2600 if scene == "indoor" else 2200
    if quality == "speed":
        max_side = min(max_side, 2200)
    return fps, max_side


def probe_video_duration_seconds(video_path: Path) -> float:
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                return 0.0
            frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            return frames / fps if frames > 0 and fps > 0 else 0.0
        finally:
            cap.release()
    except Exception:
        return 0.0


def filter_video_frames(raw_dir: Path, output_dir: Path, *, min_frames: int) -> tuple[list[Path], dict[str, Any]]:
    scores = score_blur_images(raw_dir)
    classifications = classify_blur_scores(scores)
    summary = summarize_blur_scores(scores, reject_ratio=0.0, min_images=min_frames)
    score_by_path = {item.path: item for item in scores}
    quality_values = sorted(item.quality for item in scores)
    median_quality = quality_values[len(quality_values) // 2] if quality_values else 0.0

    duplicate_paths = consecutive_duplicate_paths([item.path for item in scores])
    quality_reject_paths = {
        item.path
        for item in scores
        if should_reject_for_training(item, classifications[item.path], median_quality=median_quality)
    }
    candidates = [path for path in sorted(score_by_path) if path not in duplicate_paths and path not in quality_reject_paths]
    if len(candidates) < min_frames:
        candidates = [path for path in sorted(score_by_path) if path not in duplicate_paths]
    if len(candidates) < min_frames:
        candidates = sorted(score_by_path, key=lambda path: score_by_path[path].quality, reverse=True)[:min_frames]
        candidates = sorted(candidates, key=lambda path: path.name)

    kept_paths: list[Path] = []
    for index, source in enumerate(candidates):
        target = output_dir / f"{index:06d}.jpg"
        shutil.copy2(source, target)
        kept_paths.append(target)

    removed_paths = set(score_by_path) - set(candidates)
    metrics = {
        "extracted_frames": len(scores),
        "kept_frames": len(kept_paths),
        "duplicate_frames_removed": len(duplicate_paths & removed_paths),
        "quality_frames_removed": len(quality_reject_paths & removed_paths),
        "blurred_frames": summary.blurred_images,
        "mean_laplacian": round(summary.mean_laplacian, 4),
        "mean_texture_density": round(summary.mean_texture_density, 6),
        "mean_exposure_bad_ratio": round(summary.mean_exposure_bad_ratio, 6),
        "kept_frame_names": [path.name for path in kept_paths],
    }
    return kept_paths, metrics


def consecutive_duplicate_paths(paths: list[Path], *, threshold: int = 4) -> set[Path]:
    duplicates: set[Path] = set()
    previous_hash: int | None = None
    for path in sorted(paths, key=lambda item: item.name):
        current_hash = dhash(path)
        if previous_hash is not None and hamming_distance(previous_hash, current_hash) <= threshold:
            duplicates.add(path)
            continue
        previous_hash = current_hash
    return duplicates


def dhash(path: Path) -> int:
    if Image is None or ImageOps is None:
        return 0
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("L").resize((9, 8))
    pixels = list(image.getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            value = (value << 1) | int(left > right)
    return value


def hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())
