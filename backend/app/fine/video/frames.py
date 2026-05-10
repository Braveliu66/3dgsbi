from __future__ import annotations

from pathlib import Path

from app.fine.types import FineFailure
from app.fine.video.types import ExtractedVideoFrames


def extract_video_frames(
    video_path: Path,
    dataset_root: Path,
    *,
    max_frames: int = 240,
    max_side: int = 1600,
    jpeg_quality: int = 92,
) -> ExtractedVideoFrames:
    try:
        import cv2
    except Exception as exc:
        raise FineFailure("VIDEO_RUNTIME_UNAVAILABLE", f"OpenCV video runtime is unavailable: {exc}") from exc

    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise FineFailure("VIDEO_INPUT_MISSING", f"Missing non-empty input video: {video_path}")

    images_dir = dataset_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FineFailure("VIDEO_DECODE_FAILED", f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or None
    stride = max(1, frame_count // max_frames) if frame_count > 0 and max_frames > 0 else 1
    written = 0
    frame_index = 0
    output_width = 0
    output_height = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue
            frame = resize_frame_if_needed(frame, max_side, cv2)
            output_height, output_width = frame.shape[:2]
            timestamp = frame_index / fps if fps else float(frame_index)
            target = images_dir / f"{timestamp:012.3f}.jpg"
            if not cv2.imwrite(str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]):
                raise FineFailure("VIDEO_FRAME_WRITE_FAILED", f"Could not write extracted frame: {target}")
            written += 1
            frame_index += 1
            if max_frames > 0 and written >= max_frames:
                break
    finally:
        capture.release()

    if written <= 0:
        raise FineFailure("VIDEO_DECODE_FAILED", "Video did not yield any readable frames")
    return ExtractedVideoFrames(
        frames_dir=images_dir,
        dataset_root=dataset_root,
        count=written,
        width=output_width,
        height=output_height,
        fps=fps,
        source_video=video_path,
    )


def resize_frame_if_needed(frame, max_side: int, cv2):
    if max_side <= 0:
        return frame
    height, width = frame.shape[:2]
    side = max(width, height)
    if side <= max_side:
        return frame
    scale = max_side / side
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
