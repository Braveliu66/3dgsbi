from __future__ import annotations

import gc
import json
import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from app.preview.io.ply import fixed_preview_radius, write_point_cloud_ply
from app.preview.types import PreviewFailure


Progress = Callable[[str, int, str], None]
LINGBOT_MAP_COMMIT = "4cd986009b9adeded8a4e740919221940dedeffe"
POINT_KEYS = ("world_points", "world_points_from_depth")
COLOR_KEYS = ("images", "image", "rgb", "colors")
MAX_REASONABLE_POINT_RADIUS = 1_000.0
SUSPICIOUS_POINT_RADIUS_FOR_DEPTH_COMPARE = 50.0
POINT_SOURCE_FALLBACK_RADIUS_RATIO = 10.0
SPATIAL_TRIM_LOW_PERCENTILE = 0.5
SPATIAL_TRIM_HIGH_PERCENTILE = 99.5
CONF_KEYS_BY_POINT = {
    "world_points": ("world_points_conf", "conf"),
    "world_points_from_depth": ("depth_conf", "world_points_conf", "conf"),
    "points": ("conf", "world_points_conf"),
}
_BATCHED_NDIMS = {
    "pose_enc": 3,
    "depth": 5,
    "depth_conf": 4,
    "world_points": 5,
    "world_points_conf": 4,
    "world_points_from_depth": 5,
    "extrinsic_w2c": 4,
    "extrinsic": 4,
    "intrinsic": 4,
    "chunk_scales": 2,
    "chunk_transforms": 4,
    "images": 5,
}
DEFAULT_LINGBOT_MODEL_IMAGE_SIZE = 518
DEFAULT_LINGBOT_MIN_INFERENCE_FPS = 3.0
DEFAULT_KV_CACHE_SLIDING_WINDOW = 32
AUTO_WINDOWED_FRAME_THRESHOLD = 320
DEFAULT_PREVIEW_SPLAT_SCALE = 0.006
DEFAULT_PREVIEW_OPACITY = 0.65
SH_C0 = np.float32(0.28209479177387814)


def _log(message: str) -> None:
    print(f"[lingbot-preview] {message}", flush=True)


@dataclass(frozen=True, slots=True)
class LingBotInferenceProfile:
    image_size: int
    target_width: int
    target_height: int
    max_frames: int
    mode: str
    keyframe_interval: int | None
    camera_iterations: int
    num_scale_frames: int
    window_size: int
    kv_cache_sliding_window: int
    overlap_size: int
    overlap_keyframes: int
    preprocess_mode: str


@dataclass(frozen=True, slots=True)
class ExtractedLingBotFrames:
    frames_dir: Path
    count: int
    source_fps: float | None
    sampled_fps: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ExtractedFramePoints:
    points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray | None
    point_source: str
    raw_count: int
    filtered_count: int


@dataclass(frozen=True, slots=True)
class PreparedPreviewPoints:
    points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray | None
    input_count: int
    valid_count: int
    final_count: int


@dataclass(frozen=True, slots=True)
class PointArrayQuality:
    valid: bool
    finite_count: int
    finite_ratio: float
    radius: float


@dataclass(frozen=True, slots=True)
class PointCloudVideoConfig:
    profile: str = "stable_fast"
    mode: str = "auto"
    fps: float = 10.0
    image_size: int = 518
    target_width: int = 518
    target_height: int = 378
    window_size: int = 64
    keyframe_interval: int = 6
    overlap_keyframes: int = 8
    num_scale_frames: int = 4
    camera_iterations_fast: int = 4
    camera_iterations_retry: int = 4
    pixel_stride_fast: int = 5
    pixel_stride_full: int = 3
    conf_percentile_fast: float = 65.0
    conf_percentile_full: float = 35.0
    min_conf: float = 1e-5
    use_sdpa: bool = True
    allow_sdpa_fallback: bool = False
    compile_model: bool = False
    write_progressive_preview: bool = True
    voxel_target_fast: int = 3000
    voxel_target_full: int = 5200
    coverage_keyframes: bool = True
    coverage_rotation_degrees: float = 12.0
    coverage_translation: float = 0.35
    retry_overlap_pose_jump: float = 2.0
    save_debug_predictions: bool = False


@dataclass(frozen=True, slots=True)
class LingBotVideoWindow:
    index: int
    frame_indices: tuple[int, ...]
    frames: tuple[np.ndarray, ...]


class StreamingVoxelMap:
    def __init__(self, *, voxel_target: int) -> None:
        self.voxel_target = max(1, int(voxel_target))
        self.voxel_size: float | None = None
        self.voxels: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, float]] = {}
        self.input_points = 0

    def add_points(self, points: np.ndarray, colors: np.ndarray, confidence: np.ndarray | None) -> None:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if points.shape[0] == 0:
            return
        if confidence is None or np.asarray(confidence).reshape(-1).shape[0] != points.shape[0]:
            conf = np.ones(points.shape[0], dtype=np.float32)
        else:
            conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
        valid = np.isfinite(points).all(axis=1) & np.isfinite(conf)
        points = points[valid]
        colors = colors[valid]
        conf = conf[valid]
        if points.shape[0] == 0:
            return
        self.input_points += int(points.shape[0])
        if self.voxel_size is None:
            self.voxel_size = auto_voxel_size(points, self.voxel_target)
        keys = np.floor(points / float(self.voxel_size)).astype(np.int64)
        for key_array, point, color, quality in zip(keys, points, colors, conf):
            key = (int(key_array[0]), int(key_array[1]), int(key_array[2]))
            old = self.voxels.get(key)
            if old is None or float(quality) > old[2]:
                self.voxels[key] = (
                    point.astype(np.float32, copy=True),
                    color.astype(np.uint8, copy=True),
                    float(quality),
                )

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.voxels:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.float32),
            )
        values = list(self.voxels.values())
        return (
            np.stack([value[0] for value in values]).astype(np.float32, copy=False),
            np.stack([value[1] for value in values]).astype(np.uint8, copy=False),
            np.asarray([value[2] for value in values], dtype=np.float32),
        )

    def write_ply(self, output_ply: Path) -> int:
        points, colors, confidence = self.arrays()
        if points.shape[0] <= 0:
            raise PreviewFailure("LINGBOT_EMPTY_POINT_CLOUD", "LingBot-Map produced no valid voxelized points")
        return write_point_cloud_ply(points, colors, output_ply, confidence=confidence, max_points=0, include_confidence=False)

    def metrics(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_voxel_size": self.voxel_size,
            f"{prefix}_input_points": int(self.input_points),
            f"{prefix}_voxel_points": int(len(self.voxels)),
            f"{prefix}_points_removed_by_voxel": int(max(0, self.input_points - len(self.voxels))),
        }


def auto_voxel_size(points: np.ndarray, voxel_target: int) -> float:
    bbox_min = np.nanmin(points, axis=0)
    bbox_max = np.nanmax(points, axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    if not np.isfinite(diag) or diag <= 0:
        return 0.01
    return max(diag / max(1, int(voxel_target)), 1e-6)


def lingbot_window_frame_span(*, window_size: int, num_scale_frames: int, keyframe_interval: int) -> int:
    return int(num_scale_frames) + max(0, int(window_size) - int(num_scale_frames)) * max(1, int(keyframe_interval))


def lingbot_window_overlap_span(*, overlap_keyframes: int, keyframe_interval: int) -> int:
    return max(0, int(overlap_keyframes)) * max(1, int(keyframe_interval))


def effective_pointcloud_config(config: PointCloudVideoConfig, frame_count: int) -> PointCloudVideoConfig:
    mode = resolve_mode(config.mode, frame_count)
    return replace(config, mode=mode)


def iter_lingbot_video_windows(
    frame_iter: Iterator[tuple[int, np.ndarray]],
    *,
    window_size: int,
    num_scale_frames: int,
    keyframe_interval: int,
    overlap_keyframes: int,
) -> Iterator[LingBotVideoWindow]:
    span = lingbot_window_frame_span(
        window_size=window_size,
        num_scale_frames=num_scale_frames,
        keyframe_interval=keyframe_interval,
    )
    overlap = lingbot_window_overlap_span(overlap_keyframes=overlap_keyframes, keyframe_interval=keyframe_interval)
    step = max(1, span - overlap)
    window_start = 0
    selected: list[tuple[int, np.ndarray]] = []
    window_index = 0

    def emit_ready() -> LingBotVideoWindow:
        return LingBotVideoWindow(
            index=window_index,
            frame_indices=tuple(index for index, _frame in selected),
            frames=tuple(frame for _index, frame in selected),
        )

    for frame_index, frame in frame_iter:
        while frame_index >= window_start + span and len(selected) >= 2:
            yield emit_ready()
            window_index += 1
            window_start += step
            selected = [(index, old_frame) for index, old_frame in selected if index >= window_start]
        selected.append((frame_index, frame))
    if len(selected) >= 2:
        yield emit_ready()


def iter_video_frames_ffmpeg(
    video_path: Path,
    *,
    fps: float = 10.0,
    width: int = 518,
    height: int = 378,
) -> Iterator[tuple[int, np.ndarray]]:
    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise PreviewFailure("VIDEO_INPUT_MISSING", f"Missing non-empty input video: {video_path}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise PreviewFailure("VIDEO_RUNTIME_UNAVAILABLE", "ffmpeg is required for phone-video streaming decode")
    vf = f"fps={float(fps):g},scale={int(width)}:{int(height)}:force_original_aspect_ratio=increase,crop={int(width)}:{int(height)}"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    _log(f"video stream decode path={video_path} fps={fps} size={width}x{height} autorotate=true")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    frame_size = int(width) * int(height) * 3
    index = 0
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if not buf:
                break
            if len(buf) < frame_size:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(int(height), int(width), 3)
            yield index, frame
            index += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        stderr = b""
        if proc.stderr is not None:
            stderr = proc.stderr.read()
            proc.stderr.close()
        return_code = proc.wait()
    if return_code != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise PreviewFailure("VIDEO_DECODE_FAILED", f"ffmpeg streaming decode failed: {message or return_code}")


def estimate_video_sampled_frames(video_path: Path, *, fps: float) -> int | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    if not np.isfinite(duration) or duration <= 0:
        return None
    return max(1, int(math.ceil(duration * float(fps))))


def estimate_lingbot_window_count(sampled_frames: int | None, *, span: int, overlap: int) -> int | None:
    if sampled_frames is None or sampled_frames <= 0:
        return None
    step = max(1, int(span) - int(overlap))
    if sampled_frames <= span:
        return 1
    return 1 + int(math.ceil((sampled_frames - span) / step))


def preprocess_rgb_frames(frames: tuple[np.ndarray, ...] | list[np.ndarray], *, torch_module: Any) -> Any:
    if len(frames) < 2:
        raise PreviewFailure("LINGBOT_NOT_ENOUGH_FRAMES", "LingBot-Map preview requires at least 2 sampled frames")
    arrays = []
    for frame in frames:
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise PreviewFailure("LINGBOT_PREPROCESS_FAILED", f"RGB frame has invalid shape: {rgb.shape}")
        arrays.append(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)))
    return torch_module.from_numpy(np.stack(arrays, axis=0)).contiguous()


def run_lingbot_video_preview(
    *,
    video_path: Path,
    model_path: Path,
    work_dir: Path,
    output_ply: Path | None = None,
    output_points_ply: Path | None = None,
    output_splats_ply: Path | None = None,
    output_meta_json: Path | None = None,
    output_official_predictions_npz: Path | None = None,
    fps: int,
    max_frames: int,
    image_size: int,
    target_width: int,
    target_height: int,
    mode: str,
    keyframe_interval: int | None,
    camera_iterations: int,
    num_scale_frames: int,
    preprocess_mode: str,
    window_size: int,
    overlap_size: int,
    overlap_keyframes: int,
    max_points: int,
    frame_stride: int,
    pixel_stride: int,
    conf_percentile: float,
    min_conf: float,
    save_predictions: bool,
    compile_model: bool,
    progress: Progress,
    keyframes_only_points: bool = False,
    use_sdpa: bool = True,
    allow_sdpa_fallback: bool = False,
    min_inference_fps: float = DEFAULT_LINGBOT_MIN_INFERENCE_FPS,
) -> dict[str, Any]:
    if not model_path.exists() or model_path.stat().st_size <= 0:
        raise PreviewFailure("LINGBOT_WEIGHT_MISSING", f"LingBot-Map weight not found: {model_path}")

    started = time.monotonic()
    _log(
        "runtime start "
        f"video={video_path} model={model_path} fps={fps} max_frames={max_frames} image_size={image_size} "
        f"target_size={target_width}x{target_height} "
        f"mode={mode} keyframe_interval={keyframe_interval} camera_iterations={camera_iterations} "
        f"num_scale_frames={num_scale_frames} preprocess_mode={preprocess_mode} "
        f"window_size={window_size} overlap_size={overlap_size} overlap_keyframes={overlap_keyframes} "
        f"frame_stride={frame_stride} pixel_stride={pixel_stride} conf_percentile={conf_percentile} "
        f"min_conf={min_conf} max_points={max_points} save_predictions={save_predictions} "
        f"compile={compile_model} keyframes_only_points={keyframes_only_points} "
        f"use_sdpa={use_sdpa} allow_sdpa_fallback={allow_sdpa_fallback} min_inference_fps={min_inference_fps}"
    )
    frames = extract_video_frames(video_path, work_dir / "lingbot_frames", fps=fps, max_frames=max_frames)
    frame_message = (
        f"sampled {frames.count} LingBot frames from source_fps={frames.source_fps} "
        f"target_fps={frames.sampled_fps} size={frames.width}x{frames.height}"
    )
    progress("lingbot_frames_ready", 28, frame_message)
    _log(f"frames ready {frame_message}")

    try:
        import torch
        from lingbot_map.utils.geometry import closed_form_inverse_se3_general, unproject_depth_map_to_point_map
        from lingbot_map.utils.load_fn import load_and_preprocess_images
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    except Exception as exc:
        raise PreviewFailure("LINGBOT_RUNTIME_UNAVAILABLE", f"LingBot-Map runtime import failed: {exc}") from exc

    if not torch.cuda.is_available():
        raise PreviewFailure("GPU_RESOURCE_UNAVAILABLE", "LingBot-Map video preview requires CUDA")

    frame_paths = sorted(frames.frames_dir.glob("*.jpg"))
    if len(frame_paths) < 2:
        raise PreviewFailure("LINGBOT_NOT_ENOUGH_FRAMES", "LingBot-Map preview requires at least 2 sampled frames")

    device = torch.device("cuda:0")
    use_sdpa, flashinfer_found = resolve_lingbot_attention_backend(
        allow_sdpa_fallback=allow_sdpa_fallback,
        use_sdpa=use_sdpa,
    )
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    profile = LingBotInferenceProfile(
        image_size=image_size,
        target_width=target_width,
        target_height=target_height,
        max_frames=max_frames,
        mode=mode,
        keyframe_interval=keyframe_interval,
        camera_iterations=camera_iterations,
        num_scale_frames=num_scale_frames,
        window_size=window_size,
        kv_cache_sliding_window=resolve_kv_cache_sliding_window(window_size),
        overlap_size=overlap_size,
        overlap_keyframes=overlap_keyframes,
        preprocess_mode=preprocess_mode,
    )

    try:
        predictions, images, inference_metrics = run_lingbot_inference_profile(
            frame_paths=frame_paths,
            model_path=model_path,
            device=device,
            profile=profile,
            use_sdpa=use_sdpa,
            flashinfer_found=flashinfer_found,
            allow_sdpa_fallback=allow_sdpa_fallback,
            dtype=dtype,
            compile_requested=bool(compile_model),
            min_inference_fps=min_inference_fps,
            load_and_preprocess_images=load_and_preprocess_images,
            torch_module=torch,
            progress=progress,
        )
    except PreviewFailure:
        raise
    except Exception as exc:
        if is_cuda_out_of_memory(exc, torch_module=torch):
            release_cuda_exception(exc, torch_module=torch)
            raise PreviewFailure("LINGBOT_CUDA_OOM", "LingBot-Map inference ran out of CUDA memory") from exc
        raise PreviewFailure("LINGBOT_INFERENCE_FAILED", f"LingBot-Map inference failed: {exc}") from exc

    progress("lingbot_predictions", 66, "preparing LingBot-Map per-frame predictions")
    pred_np = predictions_to_visualization_np(
        predictions,
        images,
        pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
        closed_form_inverse_se3_general=closed_form_inverse_se3_general,
        torch_module=torch,
    )
    if output_official_predictions_npz is not None:
        save_official_predictions_npz(pred_np, output_official_predictions_npz)
    attach_depth_world_points(
        pred_np,
        unproject_depth_map_to_point_map=unproject_depth_map_to_point_map,
    )
    _log(
        "prediction keys "
        f"depth={'depth' in pred_np} extrinsic={'extrinsic' in pred_np} intrinsic={'intrinsic' in pred_np} "
        f"world_points_from_depth={'world_points_from_depth' in pred_np} "
        f"depth_conf={'depth_conf' in pred_np} world_points={'world_points' in pred_np} "
        f"images={'images' in pred_np}"
    )
    pred_np["is_keyframe"] = build_keyframe_mask(
        int(images.shape[0]),
        num_scale_frames=int(inference_metrics["lingbot_num_scale_frames"]),
        keyframe_interval=int(inference_metrics["lingbot_keyframe_interval"]),
    )

    predictions_dir = save_predictions_npz(pred_np, work_dir / "predictions") if save_predictions else None
    points_ply = output_points_ply or output_ply or output_splats_ply
    if points_ply is None:
        points_ply = work_dir / "preview_points.ply"
    progress("lingbot_pointcloud", 72, "writing LingBot-Map point-cloud PLY")
    if predictions_dir is not None:
        point_metrics = write_spark_plain_ply_from_npz(
            predictions_dir,
            points_ply,
            frame_stride=frame_stride,
            pixel_stride=pixel_stride,
            conf_percentile=conf_percentile,
            min_conf=min_conf,
            max_points=max_points,
            keyframes_only_points=keyframes_only_points,
            output_meta_json=output_meta_json,
        )
    else:
        point_metrics = write_spark_plain_ply_from_arrays(
            pred_np,
            points_ply,
            frame_stride=frame_stride,
            pixel_stride=pixel_stride,
            conf_percentile=conf_percentile,
            min_conf=min_conf,
            max_points=max_points,
            keyframes_only_points=keyframes_only_points,
            output_meta_json=output_meta_json,
        )
    point_count = int(point_metrics["point_count"])

    del pred_np
    del predictions
    clear_cuda_cache(torch)

    return {
        "adapter": "lingbot_map_spz",
        "lingbot_commit": LINGBOT_MAP_COMMIT,
        "lingbot_model": model_path.name,
        "lingbot_sampled_frames": frames.count,
        "lingbot_source_fps": frames.source_fps,
        "lingbot_sampled_fps": frames.sampled_fps,
        "lingbot_frame_width": frames.width,
        "lingbot_frame_height": frames.height,
        **inference_metrics,
        "lingbot_frame_stride": int(frame_stride),
        "lingbot_pixel_stride": int(pixel_stride),
        "lingbot_conf_percentile": float(conf_percentile),
        "lingbot_min_conf": float(min_conf),
        "lingbot_max_points": int(max_points),
        "lingbot_save_predictions": bool(save_predictions),
        "lingbot_official_predictions_npz": str(output_official_predictions_npz) if output_official_predictions_npz else None,
        "lingbot_keyframes_only_points": bool(keyframes_only_points),
        "lingbot_predictions_dir": str(predictions_dir) if predictions_dir else None,
        "quality_warning": point_metrics.get("quality_warning"),
        "point_count": point_count,
        "lingbot_duration_seconds": round(time.monotonic() - started, 3),
        **point_metrics,
    }


def run_lingbot_video_pointcloud_fast(
    *,
    video_path: Path,
    model_path: Path,
    work_dir: Path,
    output_fast_ply: Path,
    output_full_ply: Path,
    output_camera_path_json: Path,
    output_metrics_json: Path,
    output_meta_json: Path,
    config: PointCloudVideoConfig,
    progress: Progress,
) -> dict[str, Any]:
    if not model_path.exists() or model_path.stat().st_size <= 0:
        raise PreviewFailure("LINGBOT_WEIGHT_MISSING", f"LingBot-Map weight not found: {model_path}")

    started = time.monotonic()
    work_dir.mkdir(parents=True, exist_ok=True)
    _log(
        "pointcloud runtime start "
        f"video={video_path} model={model_path} profile={config.profile} fps={config.fps} requested_mode={config.mode} "
        f"window_size={config.window_size} keyframe_interval={config.keyframe_interval} "
        f"overlap_keyframes={config.overlap_keyframes} camera_iterations={config.camera_iterations_fast} "
        f"pixel_stride_fast={config.pixel_stride_fast} pixel_stride_full={config.pixel_stride_full} "
        f"conf_fast={config.conf_percentile_fast} conf_full={config.conf_percentile_full} "
        f"min_conf={config.min_conf} allow_sdpa_fallback={config.allow_sdpa_fallback}"
    )

    try:
        import torch
        from lingbot_map.utils.geometry import closed_form_inverse_se3_general, unproject_depth_map_to_point_map
        from lingbot_map.utils.load_fn import load_and_preprocess_images
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    except Exception as exc:
        raise PreviewFailure("LINGBOT_RUNTIME_UNAVAILABLE", f"LingBot-Map runtime import failed: {exc}") from exc

    if not torch.cuda.is_available():
        raise PreviewFailure("GPU_RESOURCE_UNAVAILABLE", "LingBot-Map video preview requires CUDA")

    frames = extract_video_frames(video_path, work_dir / "lingbot_frames", fps=config.fps, max_frames=0)
    frame_paths = sorted(frames.frames_dir.glob("*.jpg"))
    if len(frame_paths) < 2:
        raise PreviewFailure("LINGBOT_NOT_ENOUGH_FRAMES", "LingBot-Map preview requires at least 2 sampled frames")
    config = effective_pointcloud_config(config, len(frame_paths))
    frame_message = (
        f"sampled {frames.count} LingBot frames with official preprocessing path "
        f"target_fps={frames.sampled_fps} size={frames.width}x{frames.height}"
    )
    progress("lingbot_frames_ready", 28, frame_message)
    _log(f"frames ready {frame_message}")

    device = torch.device("cuda:0")
    use_sdpa, flashinfer_found = resolve_lingbot_attention_backend(
        allow_sdpa_fallback=config.allow_sdpa_fallback,
        use_sdpa=config.use_sdpa,
    )
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    kv_cache_sliding_window = resolve_kv_cache_sliding_window(config.window_size)

    span = lingbot_window_frame_span(
        window_size=config.window_size,
        num_scale_frames=config.num_scale_frames,
        keyframe_interval=config.keyframe_interval,
    )
    overlap = lingbot_window_overlap_span(overlap_keyframes=config.overlap_keyframes, keyframe_interval=config.keyframe_interval)
    estimated_window_count = estimate_lingbot_window_count(frames.count, span=span, overlap=overlap)
    _log(
        "pointcloud official sequence plan "
        f"sampled_frames={frames.count} estimated_internal_windows={estimated_window_count} "
        f"profile={config.profile} mode={config.mode} "
        f"window_size={config.window_size} keyframe_interval={config.keyframe_interval} "
        f"overlap_keyframes={config.overlap_keyframes} overlap_source_frames={overlap} "
        "input_frames_are_prefiltered=false external_stitching=false"
    )

    profile = LingBotInferenceProfile(
        image_size=config.image_size,
        target_width=config.target_width,
        target_height=config.target_height,
        max_frames=0,
        mode=config.mode,
        keyframe_interval=config.keyframe_interval,
        camera_iterations=config.camera_iterations_fast,
        num_scale_frames=config.num_scale_frames,
        window_size=config.window_size,
        kv_cache_sliding_window=kv_cache_sliding_window,
        overlap_size=overlap,
        overlap_keyframes=config.overlap_keyframes,
        preprocess_mode="crop",
    )

    try:
        predictions, images, inference_metrics = run_lingbot_inference_profile(
            frame_paths=frame_paths,
            model_path=model_path,
            device=device,
            profile=profile,
            use_sdpa=use_sdpa,
            flashinfer_found=flashinfer_found,
            allow_sdpa_fallback=config.allow_sdpa_fallback,
            dtype=dtype,
            compile_requested=bool(config.compile_model),
            min_inference_fps=0.0,
            load_and_preprocess_images=load_and_preprocess_images,
            torch_module=torch,
            progress=progress,
        )
    except PreviewFailure:
        raise
    except Exception as exc:
        if is_cuda_out_of_memory(exc, torch_module=torch):
            release_cuda_exception(exc, torch_module=torch)
            raise PreviewFailure("LINGBOT_CUDA_OOM", "LingBot-Map official-semantics inference ran out of CUDA memory") from exc
        if is_cuda_illegal_memory_access(exc):
            release_cuda_exception(exc, torch_module=torch)
            raise PreviewFailure(
                "LINGBOT_CUDA_ILLEGAL_MEMORY_ACCESS",
                "LingBot-Map CUDA inference hit an illegal memory access; retry with windowed or low_mem preview settings.",
            ) from exc
        raise PreviewFailure("LINGBOT_INFERENCE_FAILED", f"LingBot-Map official-semantics inference failed: {exc}") from exc

    progress("lingbot_predictions", 66, "preparing official LingBot predictions for depth-world point cloud")
    pred_np = predictions_to_visualization_np(
        predictions,
        images,
        pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
        closed_form_inverse_se3_general=closed_form_inverse_se3_general,
        torch_module=torch,
    )
    if needs_depth_world_points_fallback(pred_np):
        attach_depth_world_points(pred_np, unproject_depth_map_to_point_map=unproject_depth_map_to_point_map)
    point_source = choose_lingbot_point_source(pred_np)
    frame_count = prediction_frame_count(pred_np) or frames.count
    if "is_keyframe" not in pred_np:
        pred_np["is_keyframe"] = build_keyframe_mask(
            frame_count,
            num_scale_frames=config.num_scale_frames,
            keyframe_interval=config.keyframe_interval,
        )
    debug_predictions_path = work_dir / "official_predictions.npz"
    if config.save_debug_predictions:
        save_official_predictions_npz(pred_np, debug_predictions_path)

    voxel_fast = StreamingVoxelMap(voxel_target=config.voxel_target_fast)
    voxel_full = StreamingVoxelMap(voxel_target=config.voxel_target_full)
    camera_path: list[dict[str, Any]] = []
    seen_camera_indices: set[int] = set()
    global_pose_by_source: dict[int, np.ndarray] = {}
    previous_export_pose: np.ndarray | None = None
    frame_indices = tuple(range(frame_count))
    previous_export_pose, used_frames, raw_points, filtered_points = add_window_points_to_voxels(
        pred_np,
        frame_indices,
        voxel_fast=voxel_fast,
        voxel_full=voxel_full,
        previous_export_pose=previous_export_pose,
        config=config,
    )
    recommended_view = first_camera_view(pred_np)
    add_camera_path_entries(pred_np, frame_indices, camera_path, seen_camera_indices, global_pose_by_source)
    if config.write_progressive_preview and len(voxel_fast.voxels) > 0:
        voxel_fast.write_ply(output_fast_ply)
    _log(
        "official sequence points done "
        f"used_point_frames={used_frames} raw_points={raw_points} filtered_points={filtered_points} "
        f"fast_voxels={len(voxel_fast.voxels)} full_voxels={len(voxel_full.voxels)} "
        f"fast_voxel_size={voxel_fast.voxel_size} full_voxel_size={voxel_full.voxel_size}"
    )
    progress(
        "lingbot_window_points",
        70,
        f"finished official sequence: used_frames={used_frames}, fast_points={len(voxel_fast.voxels)}, full_points={len(voxel_full.voxels)}",
    )
    window_metrics = [
        {
            "window_index": 0,
            "frame_start": 0,
            "frame_end": int(max(0, frame_count - 1)),
            "inference_frames": int(frame_count),
            "used_point_frames": used_frames,
            "raw_points": raw_points,
            "filtered_points": filtered_points,
            "retried": False,
            "bad_reasons": [],
            "official_sequence_semantics": True,
        }
    ]
    clear_cuda_cache(torch)

    if len(voxel_fast.voxels) <= 0 or len(voxel_full.voxels) <= 0:
        raise PreviewFailure("LINGBOT_EMPTY_POINT_CLOUD", "LingBot-Map produced no valid point cloud")

    progress("lingbot_pointcloud", 72, "writing LingBot point-cloud LODs")
    fast_count = voxel_fast.write_ply(output_fast_ply)
    full_count = voxel_full.write_ply(output_full_ply)
    fast_points, _fast_colors, _fast_conf = voxel_fast.arrays()
    bbox = preview_bounds(fast_points)
    validate_pointcloud_preview(fast_count=fast_count, bbox=bbox, camera_path=camera_path)
    write_preview_meta_json(
        output_meta_json,
        point_source=point_source,
        point_count_raw=voxel_fast.input_points,
        point_count_exported=fast_count,
        bbox=bbox,
        recommended_view=recommended_view,
    )
    output_camera_path_json.parent.mkdir(parents=True, exist_ok=True)
    output_camera_path_json.write_text(
        json.dumps(camera_path_payload(camera_path, fps=config.fps), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    metrics = {
        **inference_metrics,
        "adapter": "lingbot_video_pointcloud_fast",
        "lingbot_commit": LINGBOT_MAP_COMMIT,
        "lingbot_model": model_path.name,
        "lingbot_sampled_frames": int(frames.count),
        "lingbot_sampled_fps": float(config.fps),
        "lingbot_frame_width": int(frames.width),
        "lingbot_frame_height": int(frames.height),
        "lingbot_image_size": int(config.image_size),
        "lingbot_target_width": int(config.target_width),
        "lingbot_target_height": int(config.target_height),
        "lingbot_inference_mode": config.mode,
        "lingbot_official_sequence_semantics": True,
        "lingbot_external_stitching": False,
        "lingbot_keyframe_interval": int(config.keyframe_interval),
        "lingbot_camera_iterations": int(config.camera_iterations_fast),
        "lingbot_camera_iterations_retry": int(config.camera_iterations_retry),
        "lingbot_num_scale_frames": int(config.num_scale_frames),
        "lingbot_window_size": int(config.window_size),
        "lingbot_kv_cache_sliding_window": int(kv_cache_sliding_window),
        "lingbot_overlap_keyframes": int(config.overlap_keyframes),
        "lingbot_use_sdpa": bool(use_sdpa),
        "lingbot_flashinfer_available": bool(flashinfer_found),
        "lingbot_allow_sdpa_fallback": bool(config.allow_sdpa_fallback),
        "lingbot_sdpa_fallback_active": bool(use_sdpa),
        "lingbot_aggregator_dtype": inference_metrics.get("lingbot_aggregator_dtype"),
        "lingbot_compile": bool(inference_metrics.get("lingbot_compile", False)),
        "lingbot_max_frames": 0,
        "lingbot_point_source": point_source,
        "point_source": point_source,
        "lingbot_depth_reprojection_fallback": bool(point_source == "world_points_from_depth"),
        "lingbot_keyframes_only_points": True,
        "lingbot_save_predictions": False,
        "lingbot_pixel_stride_fast": int(config.pixel_stride_fast),
        "lingbot_pixel_stride_full": int(config.pixel_stride_full),
        "lingbot_conf_percentile_fast": float(config.conf_percentile_fast),
        "lingbot_conf_percentile_full": float(config.conf_percentile_full),
        "lingbot_min_conf": float(config.min_conf),
        "lingbot_window_count": len(window_metrics),
        "lingbot_retry_window_count": 0,
        "lingbot_bad_window_count": 0,
        "lingbot_inference_seconds": inference_metrics.get("lingbot_inference_seconds"),
        "lingbot_inference_fps": inference_metrics.get("lingbot_inference_fps"),
        "lingbot_duration_seconds": round(time.monotonic() - started, 3),
        "cuda_memory_peak_mb": inference_metrics.get("cuda_memory_peak_mb"),
        "lingbot_official_predictions_npz": str(debug_predictions_path) if config.save_debug_predictions else None,
        "lingbot_official_predictions_npz_size": debug_predictions_path.stat().st_size if config.save_debug_predictions and debug_predictions_path.exists() else None,
        "point_count": int(fast_count),
        "point_count_raw": int(voxel_fast.input_points),
        "point_count_exported": int(fast_count),
        "preview_full_point_count": int(full_count),
        "preview_fast_ply": str(output_fast_ply),
        "preview_full_ply": str(output_full_ply),
        "camera_path_json": str(output_camera_path_json),
        "metrics_json": str(output_metrics_json),
        "preview_meta_json": str(output_meta_json),
        "intermediate_ply_size": output_fast_ply.stat().st_size,
        "preview_fast_ply_size": output_fast_ply.stat().st_size,
        "preview_full_ply_size": output_full_ply.stat().st_size,
        "camera_path_json_size": output_camera_path_json.stat().st_size,
        "camera_path_pose_count": len(camera_path),
        "preview_meta_json_size": output_meta_json.stat().st_size,
        "bbox_min": bbox["bbox_min"],
        "bbox_max": bbox["bbox_max"],
        "bbox_center": bbox["center"],
        "bbox_radius": bbox["radius"],
        "quality_warning": None,
        "lingbot_window_metrics": window_metrics,
        **voxel_fast.metrics("preview_fast"),
        **voxel_full.metrics("preview_full"),
    }
    output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_json.write_text(json.dumps(metrics, ensure_ascii=True, indent=2, default=json_default), encoding="utf-8")
    metrics["metrics_json_size"] = output_metrics_json.stat().st_size
    progress("lingbot_pointcloud_ready", 88, f"wrote fast={fast_count} full={full_count} depth-world points")
    return metrics


def run_lingbot_rgb_window(
    model: Any,
    window: LingBotVideoWindow,
    *,
    config: PointCloudVideoConfig,
    dtype: Any,
    device: Any,
    torch_module: Any,
    pose_encoding_to_extri_intri: Any,
    closed_form_inverse_se3_general: Any,
    unproject_depth_map_to_point_map: Any,
    overlap_keyframes: int | None = None,
    progress: Progress | None = None,
    progress_value: int = 42,
    window_label: str | None = None,
) -> tuple[dict[str, np.ndarray], float, float]:
    images = preprocess_rgb_frames(window.frames, torch_module=torch_module)
    if hasattr(images, "to"):
        images = images.to(device)
    label = window_label or f"window {window.index + 1}"
    _log(
        "upstream inference start "
        f"{label} input_frames={len(window.frames)} source_frames={window.frame_indices[0]}..{window.frame_indices[-1]} "
        f"window_size={config.window_size} upstream_keyframe_interval={config.keyframe_interval} "
        f"overlap_keyframes={overlap_keyframes or config.overlap_keyframes}"
    )
    output_device = torch_module.device("cpu")
    torch_module.cuda.reset_peak_memory_stats()
    inference_started = time.perf_counter()
    stop_heartbeat = threading.Event()
    heartbeat = threading.Thread(
        target=log_inference_heartbeat,
        args=(stop_heartbeat, label, inference_started, progress, progress_value),
        name=f"lingbot-inference-heartbeat-{window.index}",
        daemon=True,
    )
    heartbeat.start()
    try:
        with torch_module.no_grad(), torch_module.amp.autocast("cuda", dtype=dtype):
            predictions = run_lingbot_inference(
                model,
                images,
                resolved_mode="windowed",
                window_size=config.window_size,
                overlap_size=lingbot_window_overlap_span(
                    overlap_keyframes=overlap_keyframes or config.overlap_keyframes,
                    keyframe_interval=config.keyframe_interval,
                ),
                overlap_keyframes=overlap_keyframes or config.overlap_keyframes,
                num_scale_frames=config.num_scale_frames,
                keyframe_interval=config.keyframe_interval,
                output_device=output_device,
                torch_module=torch_module,
            )
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=1.0)
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    seconds = time.perf_counter() - inference_started
    peak_mb = float(torch_module.cuda.max_memory_allocated() / 1024 / 1024) if torch_module.cuda.is_available() else 0.0
    pred_np = predictions_to_visualization_np(
        predictions,
        images,
        pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
        closed_form_inverse_se3_general=closed_form_inverse_se3_general,
        torch_module=torch_module,
    )
    if needs_depth_world_points_fallback(pred_np):
        attach_depth_world_points(pred_np, unproject_depth_map_to_point_map=unproject_depth_map_to_point_map)
    return pred_np, seconds, peak_mb


def log_inference_heartbeat(
    stop_event: threading.Event,
    window_label: str,
    started: float,
    progress: Progress | None,
    progress_value: int,
    stage: str = "lingbot_window_inference",
) -> None:
    while not stop_event.wait(15.0):
        elapsed = time.perf_counter() - started
        message = f"LingBot inference still running: {window_label}, elapsed={elapsed:.0f}s"
        _log(message)
        if progress is not None:
            try:
                progress(stage, progress_value, message)
            except Exception:
                pass


def format_window_label(index: int, total: int | None) -> str:
    if total is None:
        return f"window {index + 1}"
    return f"window {index + 1}/{total}"


def window_progress_percent(
    window_index: int,
    source_frame_end: int,
    estimated_window_count: int | None,
    estimated_sampled_frames: int | None,
) -> int:
    if estimated_window_count and estimated_window_count > 0:
        fraction = min(1.0, max(0.0, float(window_index) / float(estimated_window_count)))
    elif estimated_sampled_frames and estimated_sampled_frames > 0:
        fraction = min(1.0, max(0.0, float(source_frame_end) / float(estimated_sampled_frames)))
    else:
        fraction = min(0.9, float(window_index) * 0.05)
    return int(min(70, max(34, round(34 + fraction * 36))))


def estimate_window_to_global_transform(
    pred_np: dict[str, np.ndarray],
    frame_indices: tuple[int, ...],
    global_pose_by_source: dict[int, np.ndarray],
) -> np.ndarray:
    frame_count = prediction_frame_count(pred_np) or 0
    src_positions = []
    dst_positions = []
    extrinsics = pred_np.get("extrinsic")
    if not isinstance(extrinsics, np.ndarray):
        return np.eye(4, dtype=np.float32)
    for local_index in range(min(frame_count, len(frame_indices))):
        source_index = int(frame_indices[local_index])
        previous_pose = global_pose_by_source.get(source_index)
        if previous_pose is None:
            continue
        pose = c2w_4x4(extrinsics[local_index])
        src_positions.append(pose[:3, 3])
        dst_positions.append(previous_pose[:3, 3])
    if not src_positions:
        return np.eye(4, dtype=np.float32)
    return estimate_similarity_transform(np.stack(src_positions), np.stack(dst_positions))


def estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float32).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float32).reshape(-1, 3)
    transform = np.eye(4, dtype=np.float32)
    if src.shape[0] == 0 or dst.shape[0] != src.shape[0]:
        return transform
    if src.shape[0] < 3:
        transform[:3, 3] = np.mean(dst - src, axis=0)
        return transform
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = (dst_centered.T @ src_centered) / src.shape[0]
    try:
        u, singular_values, vh = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        transform[:3, 3] = dst_mean - src_mean
        return transform
    sign = np.sign(np.linalg.det(u @ vh))
    correction = np.diag([1.0, 1.0, sign]).astype(np.float32)
    rotation = u @ correction @ vh
    variance = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    scale = float(np.sum(singular_values * np.diag(correction)) / max(variance, 1e-8))
    transform[:3, :3] = (scale * rotation).astype(np.float32)
    transform[:3, 3] = (dst_mean - scale * rotation @ src_mean).astype(np.float32)
    return transform


def apply_world_transform(pred_np: dict[str, np.ndarray], transform: np.ndarray) -> None:
    transform = np.asarray(transform, dtype=np.float32).reshape(4, 4)
    for point_key in POINT_KEYS:
        points = pred_np.get(point_key)
        if isinstance(points, np.ndarray):
            flat = points.reshape(-1, 3)
            valid = np.isfinite(flat).all(axis=1)
            transformed = flat.copy()
            transformed[valid] = (flat[valid] @ transform[:3, :3].T) + transform[:3, 3]
            pred_np[point_key] = transformed.reshape(points.shape).astype(np.float32, copy=False)
    extrinsics = pred_np.get("extrinsic")
    if isinstance(extrinsics, np.ndarray):
        transformed_ext = []
        for extrinsic in extrinsics:
            pose = transform @ c2w_4x4(extrinsic)
            transformed_ext.append(pose[:3, :4])
        pred_np["extrinsic"] = np.asarray(transformed_ext, dtype=np.float32)


def window_alignment_bad(
    pred_np: dict[str, np.ndarray],
    frame_indices: tuple[int, ...],
    global_pose_by_source: dict[int, np.ndarray],
    *,
    config: PointCloudVideoConfig,
) -> tuple[bool, list[str]]:
    reasons = []
    extrinsics = pred_np.get("extrinsic")
    if not isinstance(extrinsics, np.ndarray):
        return False, reasons
    jumps = []
    for local_index in range(min(len(frame_indices), extrinsics.shape[0])):
        previous_pose = global_pose_by_source.get(int(frame_indices[local_index]))
        if previous_pose is None:
            continue
        pose = c2w_4x4(extrinsics[local_index])
        jumps.append(float(np.linalg.norm(pose[:3, 3] - previous_pose[:3, 3])))
    if jumps and float(np.median(jumps)) > float(config.retry_overlap_pose_jump):
        reasons.append("overlap_pose_jump")
    return bool(reasons), reasons


def add_window_points_to_voxels(
    pred_np: dict[str, np.ndarray],
    frame_indices: tuple[int, ...],
    *,
    voxel_fast: StreamingVoxelMap,
    voxel_full: StreamingVoxelMap,
    previous_export_pose: np.ndarray | None,
    config: PointCloudVideoConfig,
) -> tuple[np.ndarray | None, int, int, int]:
    used_frames = 0
    raw_points = 0
    filtered_points = 0
    for frame_index, frame in iter_prediction_frames(pred_np, strided_frame_indices(prediction_frame_count(pred_np), 1)):
        source_index = int(frame_indices[frame_index]) if frame_index < len(frame_indices) else frame_index
        pose = c2w_4x4(frame["extrinsic"]) if "extrinsic" in frame else None
        if not should_export_point_frame(
            frame,
            source_index=source_index,
            pose=pose,
            previous_export_pose=previous_export_pose,
            keyframe_interval=config.keyframe_interval,
            coverage_keyframes=config.coverage_keyframes,
            rotation_threshold_degrees=config.coverage_rotation_degrees,
            translation_threshold=config.coverage_translation,
        ):
            continue
        fast = extract_frame_points_for_export(
            frame,
            pixel_stride=config.pixel_stride_fast,
            conf_percentile=config.conf_percentile_fast,
            min_conf=config.min_conf,
            source_name=f"window frame {source_index}",
        )
        full = extract_frame_points_for_export(
            frame,
            pixel_stride=config.pixel_stride_full,
            conf_percentile=config.conf_percentile_full,
            min_conf=config.min_conf,
            source_name=f"window frame {source_index}",
        )
        voxel_fast.add_points(fast.points, fast.colors, fast.confidence)
        voxel_full.add_points(full.points, full.colors, full.confidence)
        raw_points += fast.raw_count
        filtered_points += fast.filtered_count
        used_frames += 1
        if pose is not None:
            previous_export_pose = pose
    return previous_export_pose, used_frames, raw_points, filtered_points


def should_export_point_frame(
    frame: Any,
    *,
    source_index: int,
    pose: np.ndarray | None,
    previous_export_pose: np.ndarray | None,
    keyframe_interval: int,
    coverage_keyframes: bool,
    rotation_threshold_degrees: float,
    translation_threshold: float,
) -> bool:
    if prediction_frame_is_keyframe(frame):
        return True
    if source_index % max(1, int(keyframe_interval)) == 0:
        return True
    if not coverage_keyframes or pose is None or previous_export_pose is None:
        return False
    translation = float(np.linalg.norm(pose[:3, 3] - previous_export_pose[:3, 3]))
    rotation = rotation_angle_degrees(pose[:3, :3], previous_export_pose[:3, :3])
    return rotation > float(rotation_threshold_degrees) or translation > float(translation_threshold)


def rotation_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32).T
    trace = float(np.trace(relative))
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def add_camera_path_entries(
    pred_np: dict[str, np.ndarray],
    frame_indices: tuple[int, ...],
    camera_path: list[dict[str, Any]],
    seen_camera_indices: set[int],
    global_pose_by_source: dict[int, np.ndarray],
) -> None:
    extrinsics = pred_np.get("extrinsic")
    intrinsics = pred_np.get("intrinsic")
    if not isinstance(extrinsics, np.ndarray):
        return
    for local_index in range(min(extrinsics.shape[0], len(frame_indices))):
        source_index = int(frame_indices[local_index])
        pose = c2w_4x4(extrinsics[local_index])
        global_pose_by_source[source_index] = pose
        if source_index in seen_camera_indices:
            continue
        seen_camera_indices.add(source_index)
        entry = {
            "source_frame_index": source_index,
            "position": [float(value) for value in pose[:3, 3]],
            "c2w": pose[:3, :4].astype(float).tolist(),
        }
        if isinstance(intrinsics, np.ndarray) and local_index < intrinsics.shape[0]:
            entry["intrinsic"] = np.asarray(intrinsics[local_index], dtype=float).tolist()
        camera_path.append(entry)


def choose_lingbot_point_source(pred_np: dict[str, np.ndarray]) -> str:
    qualities = {
        point_key: lingbot_point_array_quality(pred_np.get(point_key))
        for point_key in POINT_KEYS
        if point_key in pred_np
    }
    native_quality = qualities.get("world_points")
    depth_quality = qualities.get("world_points_from_depth")
    if (
        native_quality is not None
        and depth_quality is not None
        and native_quality.valid
        and depth_quality.valid
        and should_fallback_from_native_world_points(native_quality, depth_quality)
    ):
        return "world_points_from_depth"
    for point_key in POINT_KEYS:
        quality = qualities.get(point_key)
        if quality is not None and quality.valid:
            return point_key
    raise PreviewFailure("LINGBOT_POINTS_MISSING", f"LingBot predictions did not contain any supported point source: {POINT_KEYS}")


def is_valid_lingbot_point_array(value: Any) -> bool:
    return lingbot_point_array_quality(value).valid


def needs_depth_world_points_fallback(pred_np: dict[str, np.ndarray]) -> bool:
    if "world_points_from_depth" in pred_np:
        return False
    quality = lingbot_point_array_quality(pred_np.get("world_points"))
    return not quality.valid or quality.radius > SUSPICIOUS_POINT_RADIUS_FOR_DEPTH_COMPARE


def should_fallback_from_native_world_points(native: PointArrayQuality, depth: PointArrayQuality) -> bool:
    if native.radius > MAX_REASONABLE_POINT_RADIUS and depth.radius <= MAX_REASONABLE_POINT_RADIUS:
        return True
    if depth.radius > 0 and native.radius > max(depth.radius * POINT_SOURCE_FALLBACK_RADIUS_RATIO, depth.radius + 1.0):
        return True
    if native.finite_ratio < 0.25 and depth.finite_ratio > native.finite_ratio:
        return True
    return False


def lingbot_point_array_quality(value: Any) -> PointArrayQuality:
    if not isinstance(value, np.ndarray) or value.size <= 0:
        return PointArrayQuality(False, 0, 0.0, math.inf)
    array = np.asarray(value)
    if array.ndim < 3 or array.shape[-1] != 3:
        return PointArrayQuality(False, 0, 0.0, math.inf)
    flat = array.reshape(-1, 3)
    finite_mask = np.isfinite(flat).all(axis=1)
    finite_count = int(finite_mask.sum())
    finite_ratio = finite_count / max(1, int(flat.shape[0]))
    if finite_count <= 0 or finite_ratio < 0.01:
        return PointArrayQuality(False, finite_count, finite_ratio, math.inf)
    finite = flat[finite_mask]
    if finite_count >= 8:
        bbox_min = np.percentile(finite, 1, axis=0)
        bbox_max = np.percentile(finite, 99, axis=0)
    else:
        bbox_min = np.min(finite, axis=0)
        bbox_max = np.max(finite, axis=0)
    radius = float(np.linalg.norm(bbox_max - bbox_min) * 0.5)
    valid = bool(np.isfinite(radius) and radius <= MAX_REASONABLE_POINT_RADIUS)
    return PointArrayQuality(valid, finite_count, finite_ratio, radius)


def camera_path_payload(camera_path: list[dict[str, Any]], *, fps: float) -> dict[str, Any]:
    poses = []
    for entry in camera_path:
        c2w = entry.get("c2w")
        pose = np.asarray(c2w, dtype=np.float32) if c2w is not None else None
        if pose is None or pose.shape != (3, 4):
            continue
        item: dict[str, Any] = {
            "position": [float(value) for value in pose[:3, 3]],
            "quaternion": [float(value) for value in rotation_matrix_to_quaternion_xyzw(pose[:3, :3])],
        }
        fov_y = camera_fov_y_degrees(entry.get("intrinsic"))
        if fov_y is not None:
            item["fov_y_deg"] = fov_y
        poses.append(item)
    return {"fps": float(fps), "poses": poses, "frames": camera_path}


def camera_fov_y_degrees(intrinsic: Any) -> float | None:
    if intrinsic is None:
        return None
    matrix = np.asarray(intrinsic, dtype=np.float32)
    if matrix.shape != (3, 3):
        return None
    fy = float(matrix[1, 1])
    cy = float(matrix[1, 2])
    image_height = max(1.0, cy * 2.0)
    if not np.isfinite(fy) or fy <= 1e-6:
        return None
    fov = float(np.degrees(2.0 * np.arctan((image_height / 2.0) / fy)))
    return fov if np.isfinite(fov) and 5.0 < fov < 170.0 else None


def rotation_matrix_to_quaternion_xyzw(matrix: np.ndarray) -> list[float]:
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return (quat / norm).astype(float).tolist()


def validate_pointcloud_preview(*, fast_count: int, bbox: dict[str, Any], camera_path: list[dict[str, Any]]) -> None:
    if int(fast_count) <= 0:
        raise PreviewFailure("LINGBOT_EMPTY_POINT_CLOUD", "LingBot-Map produced no valid point cloud")
    radius = float(bbox.get("radius") or 0.0)
    vectors = [bbox.get("bbox_min"), bbox.get("bbox_max"), bbox.get("center")]
    finite_bbox = all(is_vec3(value) for value in vectors) and np.isfinite(radius)
    if not finite_bbox or radius <= 1e-6 or radius > 1e6:
        raise PreviewFailure("LINGBOT_INVALID_POINT_CLOUD", "LingBot-Map point cloud bounds are invalid")
    if len(camera_path) < 2:
        raise PreviewFailure("LINGBOT_CAMERA_PATH_INVALID", "LingBot-Map produced fewer than 2 camera poses")


def is_vec3(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    return all(np.isfinite(float(item)) for item in value)


def first_camera_view(pred_np: dict[str, np.ndarray]) -> dict[str, Any] | None:
    for _frame_index, frame in iter_prediction_frames(pred_np, [0]):
        return lingbot_camera_view_from_frame(frame, radius_hint=1.0)
    return None


def c2w_4x4(extrinsic: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    ext = np.asarray(extrinsic, dtype=np.float32)
    matrix[:3, :4] = ext[:3, :4]
    return matrix


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def extract_video_frames(video_path: Path, output_dir: Path, *, fps: int, max_frames: int) -> ExtractedLingBotFrames:
    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise PreviewFailure("VIDEO_INPUT_MISSING", f"Missing non-empty input video: {video_path}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise PreviewFailure("VIDEO_RUNTIME_UNAVAILABLE", "ffmpeg is required for phone-video autorotation")

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.jpg"):
        stale.unlink()
    tmp_dir = output_dir.with_name(output_dir.name + "_ffmpeg_all")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pattern = tmp_dir / "%06d.jpg"
    target_fps = str(fps) if fps > 0 else "native"
    _log(
        "video decode "
        f"path={video_path} decoder=ffmpeg target_fps={target_fps} max_frames={max_frames} autorotate=true"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
    ]
    if fps > 0:
        command.extend(["-vf", f"fps={fps}"])
    command.extend(["-q:v", "2", str(pattern)])

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise PreviewFailure("VIDEO_DECODE_FAILED", f"ffmpeg could not decode video: {exc}") from exc

    all_frames = sorted(tmp_dir.glob("*.jpg"))
    if len(all_frames) < 2:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise PreviewFailure("VIDEO_DECODE_FAILED", "Video did not yield enough readable frames")

    if max_frames > 0 and len(all_frames) > max_frames:
        keep_idx = np.linspace(0, len(all_frames) - 1, max_frames, dtype=np.int64)
        selected = [all_frames[int(index)] for index in keep_idx]
    else:
        selected = all_frames
    if len(selected) < 2:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise PreviewFailure("VIDEO_DECODE_FAILED", "Video did not yield enough selected frames")

    for index, src in enumerate(selected):
        shutil.move(str(src), str(output_dir / f"{index:06d}.jpg"))
    shutil.rmtree(tmp_dir, ignore_errors=True)

    try:
        import cv2
    except Exception as exc:
        raise PreviewFailure("VIDEO_RUNTIME_UNAVAILABLE", f"OpenCV image runtime is unavailable: {exc}") from exc

    first = cv2.imread(str(output_dir / "000000.jpg"))
    if first is None:
        raise PreviewFailure("VIDEO_DECODE_FAILED", "Could not read extracted frame")
    height, width = first.shape[:2]
    _log(
        "video sampled "
        f"written_frames={len(selected)} decoded_frames={len(all_frames)} size={width}x{height} "
        f"target_fps={target_fps} max_frames={max_frames} autorotate=true"
    )
    return ExtractedLingBotFrames(output_dir, len(selected), None, fps, width, height)


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
    return "windowed" if frame_count > AUTO_WINDOWED_FRAME_THRESHOLD else "streaming"


def resolve_keyframe_interval(value: int | None, mode: str, frame_count: int) -> int:
    if value is not None and value > 0:
        return int(value)
    if frame_count > 320:
        return max(1, math.ceil(frame_count / 320))
    return 1


def resolve_preprocess_mode(mode: str, width: int, height: int) -> str:
    normalized = (mode or "auto").strip().lower()
    return normalized if normalized in {"crop", "pad"} else "crop"


def resolve_lingbot_target_dimensions(
    width: int,
    height: int,
    *,
    target_width: int,
    target_height: int,
    patch_size: int,
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return align_to_patch(target_width, patch_size), align_to_patch(target_height, patch_size)

    width, height = crop_portrait_to_square_size(width, height)
    max_width = align_to_patch(target_width, patch_size)
    max_height = align_to_patch(target_height, patch_size)
    scale = min(max_width / width, max_height / height)
    new_width = max(patch_size, int(round(width * scale / patch_size)) * patch_size)
    new_height = max(patch_size, int(round(height * scale / patch_size)) * patch_size)
    return min(new_width, max_width), min(new_height, max_height)


def align_to_patch(value: int, patch_size: int) -> int:
    return max(patch_size, int(value) // patch_size * patch_size)


def crop_portrait_to_square_size(width: int, height: int) -> tuple[int, int]:
    if height > width:
        return width, width
    return width, height


def crop_portrait_to_square_image(img: Any) -> Any:
    width, height = img.size
    if height <= width:
        return img
    top = (height - width) // 2
    return img.crop((0, top, width, top + width))


def load_and_preprocess_images_to_target_box(
    image_path_list: list[str],
    *,
    target_width: int,
    target_height: int,
    patch_size: int,
):
    if not image_path_list:
        raise ValueError("At least 1 image is required")

    try:
        import torch
        from PIL import Image, ImageOps
    except Exception as exc:
        raise PreviewFailure("LINGBOT_RUNTIME_UNAVAILABLE", f"LingBot-Map image preprocessing import failed: {exc}") from exc

    images = []
    for image_path in image_path_list:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")
        img = crop_portrait_to_square_image(img)

        width, height = img.size
        new_width, new_height = resolve_lingbot_target_dimensions(
            width,
            height,
            target_width=target_width,
            target_height=target_height,
            patch_size=patch_size,
        )
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        array = np.asarray(img, dtype=np.float32) / 255.0
        images.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())

    shapes = {(int(img.shape[1]), int(img.shape[2])) for img in images}
    if len(shapes) > 1:
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        padded = []
        for img in images:
            h_padding = max_height - int(img.shape[1])
            w_padding = max_width - int(img.shape[2])
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            padded.append(img)
        images = padded

    return torch.stack(images)


def select_lingbot_frame_paths(frame_paths: list[Path], max_frames: int) -> list[Path]:
    if max_frames <= 0 or len(frame_paths) <= max_frames:
        return frame_paths
    indices = np.linspace(0, len(frame_paths) - 1, max_frames, dtype=np.int64)
    return [frame_paths[int(index)] for index in indices]


def resolve_kv_cache_sliding_window(window_size: int) -> int:
    return max(4, min(int(window_size), DEFAULT_KV_CACHE_SLIDING_WINDOW))


def resolve_lingbot_attention_backend(
    *,
    allow_sdpa_fallback: bool,
    use_sdpa: bool = True,
    flashinfer_probe: Callable[[], bool] = flashinfer_available,
) -> tuple[bool, bool]:
    flashinfer_found = flashinfer_probe()
    if use_sdpa:
        return True, flashinfer_found
    if not flashinfer_found and not allow_sdpa_fallback:
        raise PreviewFailure(
            "LINGBOT_FLASHINFER_UNAVAILABLE",
            "LingBot-Map fast preview requires FlashInfer; install flashinfer-python or explicitly enable SDPA fallback",
        )
    return (not flashinfer_found), flashinfer_found


def run_lingbot_inference_profile(
    *,
    frame_paths: list[Path],
    model_path: Path,
    device: Any,
    profile: LingBotInferenceProfile,
    use_sdpa: bool,
    flashinfer_found: bool,
    allow_sdpa_fallback: bool,
    dtype: Any,
    compile_requested: bool,
    min_inference_fps: float,
    load_and_preprocess_images: Any,
    torch_module: Any,
    progress: Progress,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    compile_cache = configure_torch_compile_cache(model_path)
    selected_frame_paths = select_lingbot_frame_paths(frame_paths, profile.max_frames)
    if selected_frame_paths:
        try:
            import cv2

            first_frame = cv2.imread(str(selected_frame_paths[0]))
        except Exception as exc:
            raise PreviewFailure("LINGBOT_PREPROCESS_FAILED", f"Could not inspect LingBot frame size: {exc}") from exc
        if first_frame is None:
            raise PreviewFailure("LINGBOT_PREPROCESS_FAILED", f"Could not read LingBot frame: {selected_frame_paths[0]}")
        source_height, source_width = first_frame.shape[:2]
    else:
        source_width = 0
        source_height = 0
    resolved_preprocess_mode = resolve_preprocess_mode(profile.preprocess_mode, source_width, source_height)
    target_width = align_to_patch(profile.target_width, 14)
    target_height = align_to_patch(profile.target_height, 14)
    progress(
        "lingbot_preprocess",
        34,
        f"preprocessing {len(selected_frame_paths)} frames with official LingBot {resolved_preprocess_mode} mode",
    )
    try:
        images = load_and_preprocess_images(
            [str(path) for path in selected_frame_paths],
            mode=resolved_preprocess_mode,
            image_size=profile.image_size,
            patch_size=14,
        )
        if isinstance(images, tuple):
            images = images[0]
    except Exception as exc:
        raise PreviewFailure("LINGBOT_PREPROCESS_FAILED", f"LingBot-Map preprocessing failed: {exc}") from exc

    frame_count = int(images.shape[0])
    preprocessed_height = int(images.shape[-2])
    preprocessed_width = int(images.shape[-1])
    resolved_mode = resolve_mode(profile.mode, frame_count)
    resolved_keyframe_interval = resolve_keyframe_interval(profile.keyframe_interval, resolved_mode, frame_count)
    _log(
        "resolved inference "
        f"selected_frames={len(selected_frame_paths)} available_frames={len(frame_paths)} "
        f"requested_max_frames={profile.max_frames} requested_mode={profile.mode} "
        f"resolved_mode={resolved_mode} resolved_keyframe_interval={resolved_keyframe_interval} "
        f"preprocess_mode={resolved_preprocess_mode} source_size={source_width}x{source_height} "
        f"target_size={target_width}x{target_height} preprocessed_size={preprocessed_width}x{preprocessed_height} "
        f"image_size={profile.image_size} camera_iterations={profile.camera_iterations} "
        f"num_scale_frames={profile.num_scale_frames} window_size={profile.window_size} "
        f"overlap_size={profile.overlap_size} overlap_keyframes={profile.overlap_keyframes} "
        f"kv_cache_sliding_window={profile.kv_cache_sliding_window} "
        f"compile_requested={compile_requested} use_sdpa={use_sdpa} flashinfer_available={flashinfer_found} "
        f"allow_sdpa_fallback={allow_sdpa_fallback} dtype={dtype} "
        f"torchinductor_cache_dir={compile_cache['torchinductor_cache_dir']}"
    )
    model = load_lingbot_model(
        model_path,
        device,
        mode=resolved_mode,
        image_size=profile.image_size,
        use_sdpa=use_sdpa,
        camera_iterations=profile.camera_iterations,
        num_scale_frames=profile.num_scale_frames,
        window_size=profile.window_size,
        kv_cache_sliding_window=profile.kv_cache_sliding_window,
        max_frame_num=frame_count,
    )
    model_image_size = int(getattr(model, "_lingbot_model_image_size", DEFAULT_LINGBOT_MODEL_IMAGE_SIZE))
    aggregator_dtype = cast_lingbot_aggregator_for_inference(model, dtype)
    compile_active = False
    compile_fallback = False
    if should_compile_lingbot_model(compile_requested, resolved_mode):
        model = compile_lingbot_model(model)
        compile_active = True

    def _infer_once() -> tuple[dict[str, Any], float, float, float]:
        output_device = torch_module.device("cpu")
        if compile_active:
            warm_lingbot_model_once(
                model,
                images,
                num_scale_frames=profile.num_scale_frames,
                keyframe_interval=resolved_keyframe_interval,
                output_device=output_device,
                dtype=dtype,
                torch_module=torch_module,
                progress=progress,
            )
        torch_module.cuda.reset_peak_memory_stats()
        progress("lingbot_inference", 42, f"running LingBot-Map {resolved_mode} inference")
        inference_started = time.perf_counter()
        stop_heartbeat = threading.Event()
        heartbeat = threading.Thread(
            target=log_inference_heartbeat,
            args=(stop_heartbeat, f"official sequence, frames={frame_count}, mode={resolved_mode}", inference_started, progress, 42, "lingbot_inference"),
            name="lingbot-official-sequence-heartbeat",
            daemon=True,
        )
        heartbeat.start()
        try:
            with torch_module.no_grad(), torch_module.amp.autocast("cuda", dtype=dtype):
                predictions = run_lingbot_inference(
                    model,
                    images,
                    resolved_mode=resolved_mode,
                    window_size=profile.window_size,
                    overlap_size=profile.overlap_size,
                    overlap_keyframes=profile.overlap_keyframes,
                    num_scale_frames=profile.num_scale_frames,
                    keyframe_interval=resolved_keyframe_interval,
                    output_device=output_device,
                    torch_module=torch_module,
                )
        finally:
            stop_heartbeat.set()
            heartbeat.join(timeout=1.0)
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        inference_fps = frame_count / max(inference_seconds, 1e-6)
        validate_lingbot_inference_fps(inference_fps, min_inference_fps)
        peak_mb = float(torch_module.cuda.max_memory_allocated() / 1024 / 1024) if torch_module.cuda.is_available() else 0.0
        return predictions, inference_seconds, inference_fps, peak_mb

    try:
        predictions, inference_seconds, inference_fps, peak_mb = _infer_once()
    except Exception as exc:
        if not compile_active or not is_cudagraph_overwrite_error(exc):
            raise
        compile_active = False
        compile_fallback = True
        progress("lingbot_inference_retry", 48, "torch.compile CUDA Graph conflict; retrying without compile")
        clear_cuda_cache(torch_module)
        model = load_lingbot_model(
            model_path,
            device,
            mode=resolved_mode,
            image_size=profile.image_size,
            use_sdpa=use_sdpa,
            camera_iterations=profile.camera_iterations,
            num_scale_frames=profile.num_scale_frames,
            window_size=profile.window_size,
            kv_cache_sliding_window=profile.kv_cache_sliding_window,
            max_frame_num=frame_count,
        )
        model_image_size = int(getattr(model, "_lingbot_model_image_size", DEFAULT_LINGBOT_MODEL_IMAGE_SIZE))
        aggregator_dtype = cast_lingbot_aggregator_for_inference(model, dtype)
        predictions, inference_seconds, inference_fps, peak_mb = _infer_once()
    finally:
        clear_cuda_cache(torch_module)

    metrics = {
        "lingbot_image_size": int(profile.image_size),
        "lingbot_target_width": int(target_width),
        "lingbot_target_height": int(target_height),
        "lingbot_preprocessed_width": int(preprocessed_width),
        "lingbot_preprocessed_height": int(preprocessed_height),
        "lingbot_model_image_size": model_image_size,
        "lingbot_inference_frames": frame_count,
        "lingbot_inference_mode": resolved_mode,
        "lingbot_keyframe_interval": int(resolved_keyframe_interval),
        "lingbot_preprocess_mode": resolved_preprocess_mode,
        "lingbot_camera_iterations": int(profile.camera_iterations),
        "lingbot_num_scale_frames": int(profile.num_scale_frames),
        "lingbot_window_size": int(profile.window_size) if resolved_mode == "windowed" else None,
        "lingbot_kv_cache_sliding_window": int(profile.kv_cache_sliding_window),
        "lingbot_overlap_size": int(profile.overlap_size) if resolved_mode == "windowed" else None,
        "lingbot_overlap_keyframes": int(profile.overlap_keyframes) if resolved_mode == "windowed" else None,
        "lingbot_use_sdpa": bool(use_sdpa),
        "lingbot_flashinfer_available": bool(flashinfer_found),
        "lingbot_allow_sdpa_fallback": bool(allow_sdpa_fallback),
        "lingbot_sdpa_fallback_active": bool(use_sdpa),
        "lingbot_enable_point": None,
        "lingbot_aggregator_dtype": str(aggregator_dtype).replace("torch.", ""),
        "lingbot_compile": bool(compile_active),
        "lingbot_compile_requested": bool(compile_requested),
        "lingbot_compile_cudagraphs": False,
        "lingbot_compile_fallback": compile_fallback,
        "lingbot_compile_cache_dir": compile_cache["torchinductor_cache_dir"],
        "lingbot_torch_extensions_dir": compile_cache["torch_extensions_dir"],
        "lingbot_torchinductor_fx_graph_cache": compile_cache["torchinductor_fx_graph_cache"],
        "lingbot_torchinductor_autograd_cache": compile_cache["torchinductor_autograd_cache"],
        "lingbot_max_frames": int(profile.max_frames),
        "lingbot_inference_seconds": round(inference_seconds, 3),
        "lingbot_inference_fps": round(inference_fps, 3),
        "cuda_memory_peak_mb": round(peak_mb, 2),
    }
    _log(
        "inference metrics "
        f"seconds={metrics['lingbot_inference_seconds']} fps={metrics['lingbot_inference_fps']} "
        f"peak_mb={metrics['cuda_memory_peak_mb']} model_image_size={metrics['lingbot_model_image_size']} "
        f"dtype={metrics['lingbot_aggregator_dtype']} compile={metrics['lingbot_compile']} "
        f"compile_fallback={metrics['lingbot_compile_fallback']}"
    )
    return predictions, images, metrics


def configure_torch_compile_cache(model_path: Path) -> dict[str, str]:
    cache_root = model_path.parent.parent if model_path.parent.name == "lingbot" else model_path.parent
    torchinductor_cache_dir = os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        str(cache_root / "torchinductor"),
    )
    torch_extensions_dir = os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR",
        str(cache_root / "torch_extensions"),
    )
    fx_graph_cache = os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    autograd_cache = os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    for path in (torchinductor_cache_dir, torch_extensions_dir):
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    return {
        "torchinductor_cache_dir": torchinductor_cache_dir,
        "torch_extensions_dir": torch_extensions_dir,
        "torchinductor_fx_graph_cache": fx_graph_cache,
        "torchinductor_autograd_cache": autograd_cache,
    }


def should_compile_lingbot_model(compile_requested: bool, resolved_mode: str) -> bool:
    return bool(compile_requested) and str(resolved_mode).strip().lower() == "streaming"


def infer_lingbot_model_image_size_from_state_dict(
    state_dict: dict[str, Any],
    *,
    patch_size: int = 14,
    fallback: int = DEFAULT_LINGBOT_MODEL_IMAGE_SIZE,
) -> int:
    for key in ("aggregator.patch_embed.pos_embed", "module.aggregator.patch_embed.pos_embed"):
        value = state_dict.get(key)
        shape = getattr(value, "shape", None)
        if not shape or len(shape) < 2:
            continue
        token_count = int(shape[1])
        for special_tokens in (1, 0):
            patch_tokens = token_count - special_tokens
            if patch_tokens <= 0:
                continue
            patch_side = math.isqrt(patch_tokens)
            if patch_side * patch_side == patch_tokens:
                return int(patch_side * patch_size)
    return int(fallback)


def warm_lingbot_model_once(
    model: Any,
    images: Any,
    *,
    num_scale_frames: int,
    keyframe_interval: int,
    output_device: Any,
    dtype: Any,
    torch_module: Any,
    progress: Progress,
) -> None:
    frame_count = int(images.shape[0])
    warm_count = min(frame_count, max(2, min(num_scale_frames + 1, 4)))
    if warm_count < 2:
        return
    warm_scale_frames = min(max(1, int(num_scale_frames)), warm_count)
    if warm_scale_frames >= warm_count:
        warm_scale_frames = max(1, warm_count - 1)
    progress("lingbot_warmup", 38, "warming LingBot-Map kernels before timed inference")
    try:
        if hasattr(model, "clean_kv_cache"):
            model.clean_kv_cache()
        with torch_module.no_grad(), torch_module.amp.autocast("cuda", dtype=dtype):
            warm_predictions = model.inference_streaming(
                images[:warm_count],
                num_scale_frames=warm_scale_frames,
                keyframe_interval=max(int(keyframe_interval), 1),
                output_device=output_device,
            )
        del warm_predictions
    finally:
        if hasattr(model, "clean_kv_cache"):
            model.clean_kv_cache()
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
        clear_cuda_cache(torch_module)


def cast_lingbot_aggregator_for_inference(model: Any, dtype: Any) -> Any:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and dtype == torch.float32:
        return dtype
    aggregator = getattr(model, "aggregator", None)
    if aggregator is not None:
        model.aggregator = aggregator.to(dtype=dtype)
    return dtype


def validate_lingbot_inference_fps(inference_fps: float, min_inference_fps: float) -> None:
    return None


def is_cuda_out_of_memory(error: Exception, *, torch_module: Any) -> bool:
    oom_type = getattr(getattr(torch_module, "cuda", None), "OutOfMemoryError", None)
    if oom_type is not None and isinstance(error, oom_type):
        return True
    message = str(error).lower()
    if "cuda" in message and "out of memory" in message:
        return True
    if "cudacachingallocator" in message:
        return True
    if "internal assert failed" in message and ("handles_.at" in message or "cuda" in message):
        return True
    if "cublas_status_alloc_failed" in message or "cusparse_status_alloc_failed" in message:
        return True
    return False


def is_cuda_illegal_memory_access(error: Exception) -> bool:
    message = str(error).lower()
    return "cuda" in message and "illegal memory access" in message


def release_cuda_exception(error: Exception, *, torch_module: Any) -> None:
    error.__traceback__ = None
    error.__context__ = None
    error.__cause__ = None
    gc.collect()
    clear_cuda_cache(torch_module)


def clear_cuda_cache(torch_module: Any) -> None:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return
    try:
        if cuda.is_available():
            cuda.empty_cache()
            ipc_collect = getattr(cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
    except Exception as exc:
        _log(f"CUDA cache cleanup skipped: {exc}")


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
    kv_cache_sliding_window: int,
    enable_point: bool | None = None,
    enable_depth: bool | None = None,
    max_frame_num: int | None = None,
):
    try:
        import torch
        if mode == "windowed":
            from lingbot_map.models.gct_stream_window import GCTStream
        else:
            from lingbot_map.models.gct_stream import GCTStream
    except Exception as exc:
        raise PreviewFailure("LINGBOT_RUNTIME_UNAVAILABLE", f"LingBot-Map model import failed: {exc}") from exc

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model_image_size = infer_lingbot_model_image_size_from_state_dict(state_dict if isinstance(state_dict, dict) else {})
        model_kwargs = {
            "img_size": model_image_size,
            "patch_size": 14,
            "enable_3d_rope": True,
            "max_frame_num": max(1024, window_size * 16, int(max_frame_num or 0)),
            "kv_cache_sliding_window": kv_cache_sliding_window,
            "kv_cache_scale_frames": num_scale_frames,
            "kv_cache_cross_frame_special": True,
            "kv_cache_include_scale_frames": True,
            "use_sdpa": use_sdpa,
            "camera_num_iterations": camera_iterations,
        }
        if enable_point is not None:
            model_kwargs["enable_point"] = enable_point
        if enable_depth is not None:
            model_kwargs["enable_depth"] = enable_depth
        model = GCTStream(**model_kwargs)
        model.load_state_dict(state_dict, strict=False)
    except Exception as exc:
        raise PreviewFailure("LINGBOT_WEIGHT_LOAD_FAILED", f"Could not load LingBot-Map checkpoint: {exc}") from exc
    setattr(model, "_lingbot_model_image_size", model_image_size)
    return model.to(device).eval()


def compile_lingbot_model(model):
    try:
        import torch
    except Exception:
        return model
    try:
        disable_torch_compile_cudagraphs(torch)
        aggregator = model.aggregator
        for index, block in enumerate(aggregator.frame_blocks):
            aggregator.frame_blocks[index] = torch.compile(
                block,
                mode="default",
                options={"triton.cudagraphs": False},
            )
        return model
    except Exception:
        return model


def disable_torch_compile_cudagraphs(torch_module: Any) -> None:
    try:
        torch_module._inductor.config.triton.cudagraphs = False
    except Exception:
        pass


def run_lingbot_inference(
    model: Any,
    images: Any,
    *,
    resolved_mode: str,
    window_size: int,
    overlap_size: int,
    overlap_keyframes: int,
    num_scale_frames: int,
    keyframe_interval: int,
    output_device: Any,
    torch_module: Any,
) -> dict[str, Any]:
    try:
        torch_module.compiler.cudagraph_mark_step_begin()
    except Exception:
        pass
    if resolved_mode == "windowed":
        return model.inference_windowed(
            images,
            window_size=window_size,
            overlap_size=overlap_size,
            num_scale_frames=num_scale_frames,
            keyframe_interval=keyframe_interval,
            output_device=output_device,
        )
    return model.inference_streaming(
        images,
        num_scale_frames=num_scale_frames,
        keyframe_interval=keyframe_interval,
        output_device=output_device,
    )


def is_cudagraph_overwrite_error(error: Exception) -> bool:
    message = str(error).lower()
    return "cudagraph" in message and "overwritten" in message


def predictions_to_visualization_np(
    predictions: dict[str, Any],
    images: Any,
    *,
    pose_encoding_to_extri_intri: Any,
    closed_form_inverse_se3_general: Any,
    torch_module: Any,
) -> dict[str, Any]:
    if "pose_enc" in predictions:
        extrinsic_w2c, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        extrinsic_4x4 = torch_module.zeros(
            (*extrinsic_w2c.shape[:-2], 4, 4),
            device=extrinsic_w2c.device,
            dtype=extrinsic_w2c.dtype,
        )
        extrinsic_4x4[..., :3, :4] = extrinsic_w2c
        extrinsic_4x4[..., 3, 3] = 1.0
        extrinsic_c2w_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
        predictions["extrinsic_w2c"] = extrinsic_w2c
        predictions["extrinsic"] = extrinsic_c2w_4x4[..., :3, :4]
        predictions["intrinsic"] = intrinsic
        predictions["extrinsic_convention"] = "c2w"

    predictions.pop("pose_enc_list", None)
    predictions.pop("images", None)

    visualized: dict[str, Any] = {}
    for key, value in predictions.items():
        if key == "extrinsic_convention":
            visualized[key] = value
            continue
        array = to_numpy_array(value, torch_module=torch_module)
        if array is not None:
            visualized[key] = squeeze_lingbot_batch(key, array)

    image_array = to_numpy_array(images, torch_module=torch_module)
    if image_array is not None:
        visualized["images"] = squeeze_lingbot_batch("images", image_array)
    return visualized


def attach_depth_world_points(
    predictions: dict[str, np.ndarray],
    *,
    unproject_depth_map_to_point_map: Any,
) -> None:
    depth = predictions.get("depth")
    intrinsic = predictions.get("intrinsic")
    extrinsic_w2c = predictions.get("extrinsic_w2c")
    if depth is None or intrinsic is None or extrinsic_w2c is None:
        return

    try:
        depth_points = unproject_depth_map_to_point_map(
            depth,
            extrinsic_w2c,
            intrinsic,
        )
    except Exception as exc:
        raise PreviewFailure("LINGBOT_DEPTH_REPROJECT_FAILED", f"LingBot depth reprojection failed: {exc}") from exc
    predictions["world_points_from_depth"] = np.asarray(depth_points, dtype=np.float32)
    predictions["world_points_from_depth_convention"] = "world_from_depth_using_w2c"
    if "depth_conf" not in predictions:
        depth = np.asarray(predictions["depth"])
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim >= 2:
            predictions["depth_conf"] = np.ones(depth.shape, dtype=np.float32)


def to_numpy_array(value: Any, *, torch_module: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch_module.Tensor):
        value = value.detach().to("cpu")
        if value.is_floating_point():
            value = value.float()
        return value.numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def squeeze_lingbot_batch(key: str, value: np.ndarray) -> np.ndarray:
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is not None and value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


def build_keyframe_mask(frame_count: int, *, num_scale_frames: int, keyframe_interval: int) -> np.ndarray:
    keyframe_interval = max(int(keyframe_interval), 1)
    scale_frames = min(max(int(num_scale_frames), 0), frame_count)
    mask = np.zeros(frame_count, dtype=np.bool_)
    mask[:scale_frames] = True
    if scale_frames < frame_count:
        frame_indices = np.arange(scale_frames, frame_count)
        mask[scale_frames:] = ((frame_indices - scale_frames) % keyframe_interval) == 0
    return mask


def save_predictions_npz(predictions: dict[str, np.ndarray], output_dir: Path) -> Path:
    if output_dir.exists():
        for old_frame in output_dir.glob("frame_*.npz"):
            old_frame.unlink()
        old_meta = output_dir / "meta.npz"
        if old_meta.exists():
            old_meta.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = prediction_frame_count(predictions)
    if frame_count is None:
        np.savez(output_dir / "frame_000000.npz", **predictions)
        return output_dir

    sequence_keys = [
        key
        for key, value in predictions.items()
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == frame_count
    ]
    metadata = {
        key: value
        for key, value in predictions.items()
        if isinstance(value, np.ndarray) and key not in sequence_keys
    }
    for frame_index in range(frame_count):
        frame_dict = {key: predictions[key][frame_index] for key in sequence_keys}
        np.savez(output_dir / f"frame_{frame_index:06d}.npz", **frame_dict)
    if metadata:
        np.savez(output_dir / "meta.npz", **metadata)
    return output_dir


def save_official_predictions_npz(predictions: dict[str, np.ndarray], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in predictions.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(output_path, **arrays)
    return output_path


def prediction_frame_count(predictions: dict[str, np.ndarray]) -> int | None:
    images = predictions.get("images")
    if isinstance(images, np.ndarray) and images.ndim >= 1:
        return int(images.shape[0])
    for value in predictions.values():
        if isinstance(value, np.ndarray) and value.ndim >= 3:
            return int(value.shape[0])
    return None


def iter_prediction_frames(
    predictions: dict[str, np.ndarray],
    frame_indices: list[int],
) -> Iterator[tuple[int, dict[str, Any]]]:
    frame_count = prediction_frame_count(predictions)
    if frame_count is None:
        if not frame_indices or 0 in frame_indices:
            yield 0, predictions
        return

    sequence_keys = [
        key
        for key, value in predictions.items()
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == frame_count
    ]
    metadata = {
        key: value
        for key, value in predictions.items()
        if isinstance(value, np.ndarray) and key not in sequence_keys
    }
    for frame_index in frame_indices:
        frame = dict(metadata)
        frame.update({key: predictions[key][frame_index] for key in sequence_keys})
        yield frame_index, frame


def prediction_frame_keys(data: Any) -> tuple[str, ...]:
    return tuple(data.files if hasattr(data, "files") else data.keys())


def prediction_frame_is_keyframe(data: Any) -> bool | None:
    if "is_keyframe" not in prediction_frame_keys(data):
        return None
    marker = np.asarray(data["is_keyframe"], dtype=np.bool_).reshape(-1)
    if marker.size == 0:
        return False
    return bool(marker[0])


def should_export_prediction_frame(data: Any, *, keyframes_only_points: bool) -> bool:
    if not keyframes_only_points:
        return True
    is_keyframe = prediction_frame_is_keyframe(data)
    return True if is_keyframe is None else is_keyframe


def strided_frame_indices(frame_count: int | None, frame_stride: int) -> list[int]:
    if frame_count is None:
        return [0]
    return list(range(0, frame_count, max(1, int(frame_stride))))


def write_spark_plain_ply_from_arrays(
    predictions: dict[str, np.ndarray],
    output_ply: Path,
    *,
    frame_stride: int,
    pixel_stride: int,
    conf_percentile: float,
    min_conf: float,
    max_points: int,
    keyframes_only_points: bool = False,
    output_points_ply: Path | None = None,
    output_meta_json: Path | None = None,
) -> dict[str, Any]:
    frame_indices = strided_frame_indices(prediction_frame_count(predictions), frame_stride)
    selected_indices: list[int] = []
    point_batches = []
    color_batches = []
    conf_batches = []
    point_source = None
    recommended_view = None
    raw_point_count = 0
    filtered_point_count = 0
    skipped_frame_count = 0
    for frame_index, frame in iter_prediction_frames(predictions, frame_indices):
        if not should_export_prediction_frame(frame, keyframes_only_points=keyframes_only_points):
            skipped_frame_count += 1
            continue
        batch = extract_frame_points_for_export(
            frame,
            pixel_stride=pixel_stride,
            conf_percentile=conf_percentile,
            min_conf=min_conf,
            source_name=f"in-memory frame {frame_index}",
        )
        raw_point_count += batch.raw_count
        filtered_point_count += batch.filtered_count
        point_batches.append(batch.points)
        color_batches.append(batch.colors)
        conf_batches.append(batch.confidence)
        point_source = point_source or batch.point_source
        if recommended_view is None:
            recommended_view = lingbot_camera_view_from_frame(frame, radius_hint=1.0)
        selected_indices.append(frame_index)

    return write_lingbot_preview_assets(
        point_batches,
        color_batches,
        conf_batches,
        output_ply,
        point_source=point_source,
        frame_count=len(selected_indices),
        max_points=max_points,
        empty_selection_message="No LingBot-Map frames selected for point export",
        output_points_ply=output_points_ply,
        output_meta_json=output_meta_json,
        source_frame_count=len(frame_indices),
        skipped_frame_count=skipped_frame_count,
        raw_point_count=raw_point_count,
        confidence_filtered_point_count=max(0, raw_point_count - filtered_point_count),
        recommended_view=recommended_view,
    )


def write_spark_plain_ply_from_npz(
    predictions_dir: Path,
    output_ply: Path,
    *,
    frame_stride: int,
    pixel_stride: int,
    conf_percentile: float,
    min_conf: float,
    max_points: int,
    keyframes_only_points: bool = False,
    output_points_ply: Path | None = None,
    output_meta_json: Path | None = None,
) -> dict[str, Any]:
    files = sorted(predictions_dir.glob("frame_*.npz"))
    files = files[:: max(1, int(frame_stride))]
    if not files:
        raise PreviewFailure("LINGBOT_PREDICTIONS_MISSING", f"No frame_*.npz found in {predictions_dir}")

    selected_files = []
    point_batches = []
    color_batches = []
    conf_batches = []
    point_source = None
    recommended_view = None
    raw_point_count = 0
    filtered_point_count = 0
    skipped_frame_count = 0
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if not should_export_prediction_frame(data, keyframes_only_points=keyframes_only_points):
                skipped_frame_count += 1
                continue
            batch = extract_frame_points_for_export(
                data,
                pixel_stride=pixel_stride,
                conf_percentile=conf_percentile,
                min_conf=min_conf,
                source_name=str(path),
            )
            raw_point_count += batch.raw_count
            filtered_point_count += batch.filtered_count
            point_batches.append(batch.points)
            color_batches.append(batch.colors)
            conf_batches.append(batch.confidence)
            point_source = point_source or batch.point_source
            if recommended_view is None:
                recommended_view = lingbot_camera_view_from_frame(data, radius_hint=1.0)
            selected_files.append(path)

    return write_lingbot_preview_assets(
        point_batches,
        color_batches,
        conf_batches,
        output_ply,
        point_source=point_source,
        frame_count=len(selected_files),
        max_points=max_points,
        empty_selection_message=f"No frame_*.npz selected in {predictions_dir}",
        output_points_ply=output_points_ply,
        output_meta_json=output_meta_json,
        source_frame_count=len(files),
        skipped_frame_count=skipped_frame_count,
        raw_point_count=raw_point_count,
        confidence_filtered_point_count=max(0, raw_point_count - filtered_point_count),
        recommended_view=recommended_view,
    )


def write_lingbot_preview_assets(
    point_batches: list[np.ndarray],
    color_batches: list[np.ndarray],
    conf_batches: list[np.ndarray | None],
    output_ply: Path,
    *,
    point_source: str | None,
    frame_count: int,
    max_points: int,
    empty_selection_message: str,
    output_points_ply: Path | None = None,
    output_meta_json: Path | None = None,
    source_frame_count: int | None = None,
    skipped_frame_count: int = 0,
    raw_point_count: int | None = None,
    confidence_filtered_point_count: int = 0,
    recommended_view: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if frame_count <= 0:
        raise PreviewFailure("LINGBOT_PREDICTIONS_MISSING", empty_selection_message)
    if not point_batches:
        raise PreviewFailure("LINGBOT_EMPTY_POINT_CLOUD", "LingBot-Map produced no valid 3D points")

    points = np.concatenate(point_batches, axis=0) if len(point_batches) > 1 else point_batches[0]
    colors = np.concatenate(color_batches, axis=0) if len(color_batches) > 1 else color_batches[0]
    conf = concatenate_optional_conf(conf_batches, points.shape[0])
    total_points = int(points.shape[0])
    if total_points <= 0:
        raise PreviewFailure("LINGBOT_EMPTY_POINT_CLOUD", "LingBot-Map produced no valid 3D points")

    prepared = prepare_preview_points(points, colors, conf, max_points=max_points)
    points = prepared.points
    colors = prepared.colors
    conf = prepared.confidence
    point_radius = fixed_preview_radius(points)
    bbox = preview_bounds(points)
    if recommended_view is not None:
        recommended_view = {**recommended_view}
        position = np.asarray(recommended_view.get("position"), dtype=np.float32)
        target_direction = np.asarray(recommended_view.pop("_target_direction", [0.0, 0.0, 1.0]), dtype=np.float32)
        if position.shape == (3,) and target_direction.shape == (3,):
            target = position + target_direction * max(float(bbox["radius"]), 0.35)
            recommended_view["target"] = [float(value) for value in target]

    points_ply = output_points_ply or output_ply
    points_count = write_lingbot_pointcloud_ply(points, colors, conf, points_ply)
    if output_meta_json is not None:
        write_preview_meta_json(
            output_meta_json,
            point_source=point_source,
            point_count_raw=raw_point_count if raw_point_count is not None else total_points,
            point_count_exported=points_count,
            bbox=bbox,
            recommended_view=recommended_view,
        )

    quality_warning = "LingBot preview used depth-reprojected world_points_from_depth fallback." if point_source == "world_points_from_depth" else None
    _log(
        "export metrics "
        f"source={point_source} depth_fallback={point_source == 'world_points_from_depth'} "
        f"source_frames={source_frame_count if source_frame_count is not None else frame_count} "
        f"used_frames={frame_count} skipped_frames={skipped_frame_count} "
        f"raw_points={raw_point_count if raw_point_count is not None else total_points} "
        f"confidence_filtered={confidence_filtered_point_count} after_confidence={total_points} "
        f"finite_points={prepared.valid_count} exported={points_count} "
        f"removed_by_limit={max(0, prepared.valid_count - prepared.final_count)} "
        f"bbox_min={bbox['bbox_min']} bbox_max={bbox['bbox_max']} bbox_center={bbox['center']} "
        f"bbox_radius={bbox['radius']} point_radius={point_radius} warning={quality_warning}"
    )
    return {
        "point_count": int(points_count),
        "point_count_raw": int(raw_point_count if raw_point_count is not None else total_points),
        "point_count_exported": int(points_count),
        "lingbot_point_source": point_source,
        "lingbot_depth_reprojection_fallback": bool(point_source == "world_points_from_depth"),
        "lingbot_ply_format": "rgb_point_cloud",
        "lingbot_point_source_frames": int(source_frame_count if source_frame_count is not None else frame_count),
        "lingbot_point_frame_count": int(frame_count),
        "lingbot_point_skipped_frames": int(skipped_frame_count),
        "lingbot_points_before_confidence_filter": int(raw_point_count if raw_point_count is not None else total_points),
        "lingbot_points_filtered_by_confidence": int(confidence_filtered_point_count),
        "lingbot_points_after_confidence_filter": int(total_points),
        "lingbot_points_before_downsample": int(prepared.valid_count),
        "lingbot_points_after_downsample": int(points_count),
        "lingbot_points_removed_by_limit": int(max(0, prepared.valid_count - prepared.final_count)),
        "lingbot_preview_point_radius": float(point_radius),
        "bbox_min": bbox["bbox_min"],
        "bbox_max": bbox["bbox_max"],
        "bbox_center": bbox["center"],
        "bbox_radius": bbox["radius"],
        "quality_warning": quality_warning,
    }


def write_lingbot_pointcloud_ply(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray | None,
    output_ply: Path,
) -> int:
    return write_point_cloud_ply(points, colors, output_ply, confidence=confidence, max_points=0, include_confidence=False)


def write_spark_plain_ply_records(
    points: np.ndarray,
    colors: np.ndarray,
    output_ply: Path,
    *,
    splat_scale: float,
    opacity: float,
) -> None:
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    dtype = gaussian_splat_record_dtype()
    point_count = int(points.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {point_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        + "".join(f"property float f_rest_{index}\n" for index in range(45))
        + "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    ).encode("ascii")

    with output_ply.open("wb") as handle:
        handle.write(header)
        records = gaussian_splat_records(points, colors, dtype=dtype, splat_scale=splat_scale, opacity=opacity)
        records.tofile(handle)


def gaussian_splat_record_dtype() -> np.dtype:
    fields: list[tuple[str, str]] = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("f_dc_0", "<f4"),
        ("f_dc_1", "<f4"),
        ("f_dc_2", "<f4"),
    ]
    fields.extend((f"f_rest_{index}", "<f4") for index in range(45))
    fields.extend(
        [
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )
    return np.dtype(fields)


def gaussian_splat_records(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    dtype: np.dtype,
    splat_scale: float = DEFAULT_PREVIEW_SPLAT_SCALE,
    opacity: float = DEFAULT_PREVIEW_OPACITY,
) -> np.ndarray:
    records = np.zeros(points.shape[0], dtype=dtype)
    records["x"] = points[:, 0]
    records["y"] = points[:, 1]
    records["z"] = points[:, 2]
    sh_dc = rgb_to_fdc(colors.astype(np.float32) / 255.0)
    records["f_dc_0"] = sh_dc[:, 0]
    records["f_dc_1"] = sh_dc[:, 1]
    records["f_dc_2"] = sh_dc[:, 2]
    log_scale = np.float32(np.log(max(float(splat_scale), 1e-6)))
    records["opacity"] = np.float32(logit(opacity))
    records["scale_0"] = log_scale
    records["scale_1"] = log_scale
    records["scale_2"] = log_scale
    records["rot_0"] = np.float32(1.0)
    return records


def rgb_to_fdc(rgb01: np.ndarray) -> np.ndarray:
    return (rgb01.astype(np.float32) - 0.5) / SH_C0


def logit(value: float) -> float:
    clipped = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(clipped / (1.0 - clipped))


def concatenate_optional_conf(conf_batches: list[np.ndarray | None], point_count: int) -> np.ndarray | None:
    if not conf_batches or any(conf is None for conf in conf_batches):
        return None
    conf = np.concatenate([np.asarray(item, dtype=np.float32).reshape(-1) for item in conf_batches if item is not None], axis=0)
    return conf if conf.shape[0] == point_count else None


def prepare_preview_points(
    points: np.ndarray,
    colors: np.ndarray,
    conf: np.ndarray | None,
    *,
    max_points: int,
) -> PreparedPreviewPoints:
    input_count = int(points.shape[0])
    mask = np.isfinite(points).all(axis=1)
    if conf is not None:
        mask &= np.isfinite(conf)
    points = points[mask].astype("<f4", copy=False)
    colors = colors[mask].astype(np.uint8, copy=False)
    conf = None if conf is None else conf[mask].astype(np.float32, copy=False)
    if points.shape[0] == 0:
        raise PreviewFailure("LINGBOT_EMPTY_POINT_CLOUD", "LingBot-Map produced no valid 3D points")

    valid_count = int(points.shape[0])
    limit = int(max_points)
    if limit > 0 and points.shape[0] > limit:
        keep = np.linspace(0, points.shape[0] - 1, limit, dtype=np.int64)
        points = points[keep]
        colors = colors[keep]
        conf = None if conf is None else conf[keep]
    return PreparedPreviewPoints(
        points=points,
        colors=colors,
        confidence=conf,
        input_count=input_count,
        valid_count=valid_count,
        final_count=int(points.shape[0]),
    )


def estimate_preview_scale(points: np.ndarray) -> float:
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.shape[0] < 100:
        return DEFAULT_PREVIEW_SPLAT_SCALE
    bbox_min = np.percentile(finite, 1, axis=0)
    bbox_max = np.percentile(finite, 99, axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    return max(diag / 900.0, 1e-4)


def preview_bounds(points: np.ndarray) -> dict[str, Any]:
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.shape[0] == 0:
        raise PreviewFailure("LINGBOT_EMPTY_POINT_CLOUD", "LingBot-Map produced no valid 3D points")
    bbox_min = np.percentile(finite, 1, axis=0).astype(np.float32)
    bbox_max = np.percentile(finite, 99, axis=0).astype(np.float32)
    center = ((bbox_min + bbox_max) * 0.5).astype(np.float32)
    radius = max(float(np.linalg.norm(bbox_max - bbox_min) * 0.5), 0.05)
    return {
        "bbox_min": [float(value) for value in bbox_min],
        "bbox_max": [float(value) for value in bbox_max],
        "center": [float(value) for value in center],
        "radius": radius,
    }


def write_debug_points_ply(points: np.ndarray, colors: np.ndarray, output_ply: Path) -> None:
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    records = np.empty(points.shape[0], dtype=dtype)
    records["x"] = points[:, 0]
    records["y"] = points[:, 1]
    records["z"] = points[:, 2]
    records["red"] = colors[:, 0]
    records["green"] = colors[:, 1]
    records["blue"] = colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with output_ply.open("wb") as handle:
        handle.write(header)
        records.tofile(handle)


def write_preview_meta_json(
    output_meta_json: Path,
    *,
    point_source: str | None,
    point_count_raw: int,
    point_count_exported: int,
    bbox: dict[str, Any],
    recommended_view: dict[str, Any] | None = None,
) -> None:
    output_meta_json.parent.mkdir(parents=True, exist_ok=True)
    radius = float(bbox["radius"])
    payload = {
        "asset_type": "lingbot_preview_points",
        "point_source": point_source,
        "num_points": int(point_count_exported),
        "point_count_raw": int(point_count_raw),
        "point_count_exported": int(point_count_exported),
        "bbox_min": bbox["bbox_min"],
        "bbox_max": bbox["bbox_max"],
        "center": bbox["center"],
        "radius": radius,
        "scale_applied": 1.0,
        "coordinate_system": "lingbot_world",
        "recommended_frontend": {
            "camera_distance": max(radius * 2.4, 0.5),
            "near": max(radius / 1000.0, 0.001),
            "far": max(radius * 100.0, 100.0),
        },
    }
    if recommended_view is not None:
        payload["recommended_view"] = recommended_view
    output_meta_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def lingbot_camera_view_from_frame(frame: Any, *, radius_hint: float) -> dict[str, Any] | None:
    try:
        if "extrinsic" not in frame or "intrinsic" not in frame:
            return None
        extrinsic = np.asarray(frame["extrinsic"], dtype=np.float32)
        intrinsic = np.asarray(frame["intrinsic"], dtype=np.float32)
        camera_to_world = extrinsic[:3, :4]
        rotation = camera_to_world[:3, :3]
        position = camera_to_world[:3, 3].astype(np.float32)
        forward = normalize_vector(rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        up = normalize_vector(rotation @ np.asarray([0.0, -1.0, 0.0], dtype=np.float32))
        if forward is None or up is None:
            return None
        view: dict[str, Any] = {
            "source": "first_frame_camera",
            "position": [float(value) for value in position],
            "_target_direction": [float(value) for value in forward],
            "target": [float(value) for value in position + forward * max(float(radius_hint), 0.35)],
            "up": [float(value) for value in up],
        }
        image = np.asarray(frame["images"]) if "images" in frame else None
        image_height = int(image.shape[-2]) if image is not None and image.ndim >= 2 else 0
        focal_y = float(intrinsic[1, 1]) if intrinsic.shape[0] >= 2 and intrinsic.shape[1] >= 2 else 0.0
        if image_height > 0 and focal_y > 0:
            view["fov_y_degrees"] = float(np.degrees(2.0 * np.arctan(float(image_height) / (2.0 * focal_y))))
        return view
    except Exception:
        return None


def normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-6:
        return None
    return (vector / norm).astype(np.float32)


def extract_prediction_frame_points(
    predictions: dict[str, np.ndarray],
    frame_index: int,
    *,
    pixel_stride: int,
    conf_percentile: float,
    min_conf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    for _, frame in iter_prediction_frames(predictions, [frame_index]):
        return extract_frame_points(
            frame,
            pixel_stride=pixel_stride,
            conf_percentile=conf_percentile,
            min_conf=min_conf,
            source_name=f"in-memory frame {frame_index}",
        )
    raise PreviewFailure("LINGBOT_PREDICTIONS_MISSING", f"No in-memory frame {frame_index}")


def uniform_chunk_indices(start: int, count: int, total: int, target: int) -> np.ndarray:
    if target >= total:
        return np.arange(count, dtype=np.int64)
    end = start + count
    first_k = (start * target + total - 1) // total
    last_k = (end * target + total - 1) // total
    if last_k <= first_k:
        return np.empty(0, dtype=np.int64)
    selected = (np.arange(first_k, last_k, dtype=np.int64) * total) // target
    selected = selected - start
    return selected[(selected >= 0) & (selected < count)]


def extract_npz_frame_points(
    path: Path,
    *,
    pixel_stride: int,
    conf_percentile: float,
    min_conf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    with np.load(path, allow_pickle=False) as data:
        return extract_frame_points(
            data,
            pixel_stride=pixel_stride,
            conf_percentile=conf_percentile,
            min_conf=min_conf,
            source_name=str(path),
        )


def extract_frame_points(
    data: Any,
    *,
    pixel_stride: int,
    conf_percentile: float,
    min_conf: float,
    source_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    batch = extract_frame_points_for_export(
        data,
        pixel_stride=pixel_stride,
        conf_percentile=conf_percentile,
        min_conf=min_conf,
        source_name=source_name,
    )
    return batch.points, batch.colors, batch.confidence, batch.point_source


def extract_frame_points_for_export(
    data: Any,
    *,
    pixel_stride: int,
    conf_percentile: float,
    min_conf: float,
    source_name: str,
) -> ExtractedFramePoints:
    keys = tuple(data.files if hasattr(data, "files") else data.keys())
    try:
        point_key, points_grid, confidence_grid = select_lingbot_points(data)
    except PreviewFailure as exc:
        if exc.code == "LINGBOT_POINTS_MISSING":
            raise PreviewFailure("LINGBOT_POINT_FIELD_MISSING", f"{source_name} has no point field") from exc
        raise

    stride = max(1, int(pixel_stride))
    if points_grid.ndim == 4 and points_grid.shape[0] == 1:
        points_grid = points_grid[0]
    if points_grid.ndim != 3 or points_grid.shape[-1] != 3:
        raise PreviewFailure("LINGBOT_POINT_FIELD_INVALID", f"{source_name}: {point_key} shape {points_grid.shape}")
    points = points_grid[::stride, ::stride, :].reshape(-1, 3)
    raw_count = int(points.shape[0])

    colors = None
    color_key = pick_key(keys, COLOR_KEYS)
    if color_key is not None:
        image = image_to_hwc_u8(data[color_key])
        if image is not None:
            colors = image[::stride, ::stride, :].reshape(-1, 3)
    if colors is None or colors.shape[0] != points.shape[0]:
        colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)

    mask = np.isfinite(points).all(axis=1)
    confidence = None
    if confidence_grid is not None:
        confidence = np.asarray(confidence_grid, dtype=np.float32)
        confidence = np.squeeze(confidence)
        if confidence.ndim == 2:
            confidence = confidence[::stride, ::stride].reshape(-1)
            if confidence.shape[0] == points.shape[0]:
                finite_conf = np.isfinite(confidence)
                threshold = float(min_conf)
                valid_confidence = confidence[finite_conf]
                if conf_percentile > 0 and valid_confidence.size > 0:
                    threshold = max(threshold, float(np.percentile(valid_confidence, conf_percentile)))
                mask &= finite_conf & (confidence >= threshold)
        else:
            confidence = None

    mask = apply_spatial_outlier_mask(points, mask)
    filtered_conf = None
    if confidence is not None and confidence.shape[0] == points.shape[0]:
        filtered_conf = confidence[mask].astype(np.float32, copy=False)
    filtered_points = points[mask].astype("<f4", copy=False)
    return ExtractedFramePoints(
        points=filtered_points,
        colors=colors[mask].astype(np.uint8, copy=False),
        confidence=filtered_conf,
        point_source=point_key,
        raw_count=raw_count,
        filtered_count=int(filtered_points.shape[0]),
    )


def apply_spatial_outlier_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected_count = int(mask.sum())
    if selected_count < 256:
        return mask
    selected = points[mask]
    bbox_min = np.percentile(selected, SPATIAL_TRIM_LOW_PERCENTILE, axis=0)
    bbox_max = np.percentile(selected, SPATIAL_TRIM_HIGH_PERCENTILE, axis=0)
    if not np.isfinite(bbox_min).all() or not np.isfinite(bbox_max).all():
        return mask
    spatial = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
    trimmed = mask & spatial
    if int(trimmed.sum()) < max(16, int(selected_count * 0.5)):
        return mask
    return trimmed


def select_lingbot_points(predictions: Any) -> tuple[str, np.ndarray, np.ndarray | None]:
    keys = tuple(predictions.files if hasattr(predictions, "files") else predictions.keys())
    available = set(keys)
    for key in POINT_KEYS:
        if key not in available:
            continue
        points = np.asarray(predictions[key], dtype=np.float32)
        if not is_valid_lingbot_point_array(points):
            continue
        conf = None
        for conf_key in CONF_KEYS_BY_POINT.get(key, ("conf", "world_points_conf", "depth_conf")):
            if conf_key in available:
                conf = np.asarray(predictions[conf_key], dtype=np.float32)
                break
        return key, points, conf
    raise PreviewFailure(
        "LINGBOT_POINTS_MISSING",
        f"LingBot predictions did not contain any supported point source: {POINT_KEYS}",
    )


def image_to_hwc_u8(image: Any) -> np.ndarray | None:
    array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.transpose(array[:3], (1, 2, 0))
    if array.ndim != 3 or array.shape[-1] < 3:
        return None

    array = array[..., :3]
    if array.dtype != np.uint8:
        if array.size > 0 and np.nanmax(array) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def pick_key(keys: tuple[str, ...], candidates: tuple[str, ...]) -> str | None:
    available = set(keys)
    for key in candidates:
        if key in available:
            return key
    return None
