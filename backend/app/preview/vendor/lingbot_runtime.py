from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.preview.io.ply import write_gaussian_splat_ply
from app.preview.types import PreviewFailure


Progress = Callable[[str, int, str], None]
LINGBOT_MAP_COMMIT = "4cd986009b9adeded8a4e740919221940dedeffe"


@dataclass(frozen=True, slots=True)
class ExtractedLingBotFrames:
    frames_dir: Path
    count: int
    source_fps: float | None
    sampled_fps: int
    width: int
    height: int


def run_lingbot_video_preview(
    *,
    video_path: Path,
    model_path: Path,
    output_ply: Path,
    work_dir: Path,
    fps: int,
    max_frames: int,
    image_size: int,
    mode: str,
    keyframe_interval: int | None,
    camera_iterations: int,
    num_scale_frames: int,
    window_size: int,
    overlap_keyframes: int,
    max_points: int,
    conf_threshold: float,
    compile_model: bool,
    progress: Progress,
) -> dict[str, Any]:
    if not model_path.exists() or model_path.stat().st_size <= 0:
        raise PreviewFailure("LINGBOT_WEIGHT_MISSING", f"LingBot-Map weight not found: {model_path}")

    started = time.monotonic()
    frames = extract_video_frames(video_path, work_dir / "lingbot_frames", fps=fps, max_frames=max_frames)
    progress("lingbot_frames_ready", 28, f"sampled {frames.count} video frames for LingBot-Map")

    try:
        import torch
        from lingbot_map.utils.load_fn import load_and_preprocess_images
    except Exception as exc:
        raise PreviewFailure("LINGBOT_RUNTIME_UNAVAILABLE", f"LingBot-Map runtime import failed: {exc}") from exc

    if not torch.cuda.is_available():
        raise PreviewFailure("GPU_RESOURCE_UNAVAILABLE", "LingBot-Map video preview requires CUDA")

    frame_paths = sorted(frames.frames_dir.glob("*.jpg"))
    if len(frame_paths) < 2:
        raise PreviewFailure("LINGBOT_NOT_ENOUGH_FRAMES", "LingBot-Map preview requires at least 2 sampled frames")

    progress("lingbot_preprocess", 34, f"preprocessing {len(frame_paths)} frames at image size {image_size}")
    try:
        images = load_and_preprocess_images(
            [str(path) for path in frame_paths],
            mode="crop",
            image_size=image_size,
            patch_size=14,
        )
    except Exception as exc:
        raise PreviewFailure("LINGBOT_PREPROCESS_FAILED", f"LingBot-Map preprocessing failed: {exc}") from exc

    device = torch.device("cuda:0")
    use_sdpa = not flashinfer_available()
    resolved_mode = resolve_mode(mode, int(images.shape[0]))
    resolved_keyframe_interval = resolve_keyframe_interval(keyframe_interval, resolved_mode, int(images.shape[0]))
    model = load_lingbot_model(
        model_path,
        device,
        mode=resolved_mode,
        image_size=image_size,
        use_sdpa=use_sdpa,
        camera_iterations=camera_iterations,
        num_scale_frames=num_scale_frames,
        window_size=window_size,
    )
    if compile_model:
        model = compile_lingbot_model(model)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    output_device = torch.device("cpu")
    torch.cuda.reset_peak_memory_stats()
    progress("lingbot_inference", 42, f"running LingBot-Map {resolved_mode} inference")
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            if resolved_mode == "windowed":
                predictions = model.inference_windowed(
                    images,
                    window_size=window_size,
                    overlap_keyframes=overlap_keyframes,
                    num_scale_frames=num_scale_frames,
                    keyframe_interval=resolved_keyframe_interval,
                    output_device=output_device,
                )
            else:
                predictions = model.inference_streaming(
                    images,
                    num_scale_frames=num_scale_frames,
                    keyframe_interval=resolved_keyframe_interval,
                    output_device=output_device,
                )
    except Exception as exc:
        raise PreviewFailure("LINGBOT_INFERENCE_FAILED", f"LingBot-Map inference failed: {exc}") from exc
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    progress("lingbot_pointcloud", 72, "filtering LingBot-Map world points")
    points, colors, confidence, point_metrics = predictions_to_points(
        predictions,
        max_points=max_points,
        conf_threshold=conf_threshold,
    )
    point_count = write_gaussian_splat_ply(
        points,
        colors,
        output_ply,
        confidence=confidence,
        max_points=0,
        scale=0.002,
        opacity_logit=-2.0,
    )

    peak_mb = float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0
    return {
        "adapter": "lingbot_map_spz",
        "lingbot_commit": LINGBOT_MAP_COMMIT,
        "lingbot_model": model_path.name,
        "lingbot_sampled_frames": frames.count,
        "lingbot_source_fps": frames.source_fps,
        "lingbot_sampled_fps": frames.sampled_fps,
        "lingbot_frame_width": frames.width,
        "lingbot_frame_height": frames.height,
        "lingbot_image_size": int(image_size),
        "lingbot_inference_mode": resolved_mode,
        "lingbot_keyframe_interval": resolved_keyframe_interval,
        "lingbot_camera_iterations": int(camera_iterations),
        "lingbot_num_scale_frames": int(num_scale_frames),
        "lingbot_window_size": int(window_size) if resolved_mode == "windowed" else None,
        "lingbot_overlap_keyframes": int(overlap_keyframes) if resolved_mode == "windowed" else None,
        "lingbot_use_sdpa": bool(use_sdpa),
        "lingbot_compile": bool(compile_model),
        "point_count": int(point_count),
        "cuda_memory_peak_mb": round(peak_mb, 2),
        "lingbot_duration_seconds": round(time.monotonic() - started, 3),
        **point_metrics,
    }


def extract_video_frames(video_path: Path, output_dir: Path, *, fps: int, max_frames: int) -> ExtractedLingBotFrames:
    try:
        import cv2
    except Exception as exc:
        raise PreviewFailure("VIDEO_RUNTIME_UNAVAILABLE", f"OpenCV video runtime is unavailable: {exc}") from exc

    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise PreviewFailure("VIDEO_INPUT_MISSING", f"Missing non-empty input video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise PreviewFailure("VIDEO_DECODE_FAILED", f"Could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or None
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    interval = max(1, int(round((source_fps or fps) / max(1, fps))))
    if total_frames > 0 and max_frames > 0:
        interval = max(interval, math.ceil(total_frames / max_frames))

    written = 0
    frame_index = 0
    width = 0
    height = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % interval != 0:
                frame_index += 1
                continue
            height, width = frame.shape[:2]
            target = output_dir / f"{written:06d}.jpg"
            if not cv2.imwrite(str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
                raise PreviewFailure("VIDEO_FRAME_WRITE_FAILED", f"Could not write extracted frame: {target}")
            written += 1
            frame_index += 1
            if max_frames > 0 and written >= max_frames:
                break
    finally:
        capture.release()

    if written < 2:
        raise PreviewFailure("VIDEO_DECODE_FAILED", "Video did not yield enough readable frames")
    return ExtractedLingBotFrames(output_dir, written, source_fps, fps, width, height)


def flashinfer_available() -> bool:
    try:
        __import__("flashinfer")
        return True
    except Exception:
        try:
            __import__("flashinfer_python")
            return True
        except Exception:
            return False


def resolve_mode(value: str, frame_count: int) -> str:
    normalized = (value or "auto").strip().lower()
    if normalized in {"streaming", "windowed"}:
        return normalized
    return "streaming" if frame_count <= 512 else "windowed"


def resolve_keyframe_interval(value: int | None, mode: str, frame_count: int) -> int:
    if value is not None and value > 0:
        return int(value)
    if mode == "streaming" and frame_count > 320:
        return max(1, math.ceil(frame_count / 320))
    return 1


def load_lingbot_model(
    model_path: Path,
    device,
    *,
    mode: str,
    image_size: int,
    use_sdpa: bool,
    camera_iterations: int,
    num_scale_frames: int,
    window_size: int,
):
    try:
        import torch
        if mode == "windowed":
            from lingbot_map.models.gct_stream_window import GCTStream
        else:
            from lingbot_map.models.gct_stream import GCTStream
    except Exception as exc:
        raise PreviewFailure("LINGBOT_RUNTIME_UNAVAILABLE", f"LingBot-Map model import failed: {exc}") from exc

    model = GCTStream(
        img_size=image_size,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=max(1024, window_size * 16),
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=use_sdpa,
        camera_num_iterations=camera_iterations,
        enable_point=True,
        enable_depth=True,
    )
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=False)
    except Exception as exc:
        raise PreviewFailure("LINGBOT_WEIGHT_LOAD_FAILED", f"Could not load LingBot-Map checkpoint: {exc}") from exc
    return model.to(device).eval()


def compile_lingbot_model(model):
    try:
        import torch
    except Exception:
        return model
    try:
        aggregator = model.aggregator
        for index, block in enumerate(aggregator.frame_blocks):
            aggregator.frame_blocks[index] = torch.compile(block, mode="reduce-overhead")
        return model
    except Exception:
        return model


def predictions_to_points(
    predictions: dict[str, Any],
    *,
    max_points: int,
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    world_points = squeeze_array(predictions.get("world_points"))
    confidence = squeeze_array(predictions.get("world_points_conf"))
    images = squeeze_array(predictions.get("images"))
    if world_points is None:
        raise PreviewFailure("LINGBOT_EMPTY_POINT_CLOUD", "LingBot-Map did not return world_points")
    if confidence is None:
        confidence = np.ones(world_points.shape[:-1], dtype=np.float32)
    if images is None:
        raise PreviewFailure("LINGBOT_IMAGE_COLORS_MISSING", "LingBot-Map did not return source images")

    points = np.asarray(world_points, dtype=np.float32).reshape(-1, 3)
    conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
    image_array = np.asarray(images)
    if image_array.ndim != 4:
        raise PreviewFailure("LINGBOT_IMAGE_COLORS_INVALID", f"unexpected LingBot image shape: {image_array.shape}")
    colors = np.transpose(image_array, (0, 2, 3, 1)).reshape(-1, 3)
    if colors.dtype.kind == "f":
        colors = colors * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    valid = np.isfinite(points).all(axis=1) & np.isfinite(conf)
    threshold_valid = valid & (conf >= float(conf_threshold))
    used_threshold = True
    if not np.any(threshold_valid):
        threshold_valid = valid
        used_threshold = False
    points = points[threshold_valid]
    colors = colors[threshold_valid]
    conf = conf[threshold_valid]
    if points.shape[0] == 0:
        raise PreviewFailure("LINGBOT_EMPTY_POINT_CLOUD", "LingBot-Map produced no valid 3D points")

    before_downsample = int(points.shape[0])
    if max_points > 0 and points.shape[0] > max_points:
        keep = np.argsort(conf)[::-1][:max_points]
        points = points[keep]
        colors = colors[keep]
        conf = conf[keep]

    return points, colors, conf, {
        "lingbot_conf_threshold": float(conf_threshold),
        "lingbot_conf_threshold_used": used_threshold,
        "lingbot_points_before_downsample": before_downsample,
        "lingbot_points_after_downsample": int(points.shape[0]),
    }


def squeeze_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr
