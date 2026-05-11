from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.preview.types import PreviewFailure


Progress = Callable[[str, int, str], None]
LINGBOT_MAP_COMMIT = "4cd986009b9adeded8a4e740919221940dedeffe"
POINT_KEYS = ("world_points_from_depth", "world_points", "points")
COLOR_KEYS = ("images", "image", "rgb", "colors")
CONF_KEYS_BY_POINT = {
    "world_points_from_depth": ("depth_conf", "world_points_conf", "conf"),
    "world_points": ("world_points_conf", "depth_conf", "conf"),
    "points": ("conf", "world_points_conf", "depth_conf"),
}
_BATCHED_NDIMS = {
    "pose_enc": 3,
    "depth": 5,
    "depth_conf": 4,
    "world_points": 5,
    "world_points_conf": 4,
    "world_points_from_depth": 5,
    "extrinsic": 4,
    "intrinsic": 4,
    "chunk_scales": 2,
    "chunk_transforms": 4,
    "images": 5,
}


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
    frame_stride: int,
    pixel_stride: int,
    conf_percentile: float,
    min_conf: float,
    save_predictions: bool,
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
    compile_requested = bool(compile_model)
    compile_active = False
    compile_fallback = False
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
    if compile_requested:
        model = compile_lingbot_model(model)
        compile_active = True

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    output_device = torch.device("cpu")
    torch.cuda.reset_peak_memory_stats()
    progress("lingbot_inference", 42, f"running LingBot-Map {resolved_mode} inference")
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            predictions = run_lingbot_inference(
                model,
                images,
                resolved_mode=resolved_mode,
                window_size=window_size,
                overlap_keyframes=overlap_keyframes,
                num_scale_frames=num_scale_frames,
                keyframe_interval=resolved_keyframe_interval,
                output_device=output_device,
                torch_module=torch,
            )
    except Exception as exc:
        if not compile_active or not is_cudagraph_overwrite_error(exc):
            raise PreviewFailure("LINGBOT_INFERENCE_FAILED", f"LingBot-Map inference failed: {exc}") from exc
        compile_active = False
        compile_fallback = True
        progress("lingbot_inference_retry", 48, "torch.compile CUDA Graph conflict; retrying without compile")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                predictions = run_lingbot_inference(
                    model,
                    images,
                    resolved_mode=resolved_mode,
                    window_size=window_size,
                    overlap_keyframes=overlap_keyframes,
                    num_scale_frames=num_scale_frames,
                    keyframe_interval=resolved_keyframe_interval,
                    output_device=output_device,
                    torch_module=torch,
                )
        except Exception as retry_exc:
            raise PreviewFailure("LINGBOT_INFERENCE_FAILED", f"LingBot-Map inference failed after compile fallback: {retry_exc}") from retry_exc
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    progress("lingbot_predictions", 66, "preparing LingBot-Map per-frame predictions")
    pred_np = predictions_to_visualization_np(
        predictions,
        images,
        pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
        closed_form_inverse_se3_general=closed_form_inverse_se3_general,
        torch_module=torch,
    )
    attach_depth_world_points(
        pred_np,
        unproject_depth_map_to_point_map=unproject_depth_map_to_point_map,
    )
    pred_np["is_keyframe"] = build_keyframe_mask(
        int(images.shape[0]),
        num_scale_frames=num_scale_frames,
        keyframe_interval=resolved_keyframe_interval,
    )

    predictions_dir = save_predictions_npz(pred_np, work_dir / "predictions") if save_predictions else None
    progress("lingbot_pointcloud", 72, "writing Spark plain PLY from LingBot-Map predictions")
    if predictions_dir is not None:
        point_metrics = write_spark_plain_ply_from_npz(
            predictions_dir,
            output_ply,
            frame_stride=frame_stride,
            pixel_stride=pixel_stride,
            conf_percentile=conf_percentile,
            min_conf=min_conf,
            max_points=max_points,
        )
    else:
        point_metrics = write_spark_plain_ply_from_arrays(
            pred_np,
            output_ply,
            frame_stride=frame_stride,
            pixel_stride=pixel_stride,
            conf_percentile=conf_percentile,
            min_conf=min_conf,
            max_points=max_points,
        )
    point_count = int(point_metrics["point_count"])

    del pred_np
    del predictions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        "lingbot_compile": bool(compile_active),
        "lingbot_compile_requested": compile_requested,
        "lingbot_compile_cudagraphs": False,
        "lingbot_compile_fallback": compile_fallback,
        "lingbot_max_frames": int(max_frames),
        "lingbot_frame_stride": int(frame_stride),
        "lingbot_pixel_stride": int(pixel_stride),
        "lingbot_conf_percentile": float(conf_percentile),
        "lingbot_min_conf": float(min_conf),
        "lingbot_max_points": int(max_points),
        "lingbot_save_predictions": bool(save_predictions),
        "lingbot_predictions_dir": str(predictions_dir) if predictions_dir else None,
        "point_count": point_count,
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
    enable_point: bool = True,
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
        enable_point=enable_point,
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
            overlap_keyframes=overlap_keyframes,
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
) -> dict[str, np.ndarray]:
    if "pose_enc" in predictions:
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        extrinsic_4x4 = torch_module.zeros(
            (*extrinsic.shape[:-2], 4, 4),
            device=extrinsic.device,
            dtype=extrinsic.dtype,
        )
        extrinsic_4x4[..., :3, :4] = extrinsic
        extrinsic_4x4[..., 3, 3] = 1.0
        extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
        predictions["extrinsic"] = extrinsic_4x4[..., :3, :4]
        predictions["intrinsic"] = intrinsic

    predictions.pop("pose_enc_list", None)
    predictions.pop("images", None)

    visualized: dict[str, np.ndarray] = {}
    for key, value in predictions.items():
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
    if "world_points_from_depth" in predictions:
        return
    if not all(key in predictions for key in ("depth", "extrinsic", "intrinsic")):
        return
    try:
        depth_points = unproject_depth_map_to_point_map(
            predictions["depth"],
            predictions["extrinsic"],
            predictions["intrinsic"],
        )
    except Exception as exc:
        if "world_points" in predictions:
            return
        raise PreviewFailure("LINGBOT_DEPTH_REPROJECT_FAILED", f"LingBot depth reprojection failed: {exc}") from exc
    predictions["world_points_from_depth"] = np.asarray(depth_points, dtype=np.float32)


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


def prediction_frame_count(predictions: dict[str, np.ndarray]) -> int | None:
    images = predictions.get("images")
    if isinstance(images, np.ndarray) and images.ndim >= 1:
        return int(images.shape[0])
    for value in predictions.values():
        if isinstance(value, np.ndarray) and value.ndim >= 3:
            return int(value.shape[0])
    return None


def write_spark_plain_ply_from_arrays(
    predictions: dict[str, np.ndarray],
    output_ply: Path,
    *,
    frame_stride: int,
    pixel_stride: int,
    conf_percentile: float,
    min_conf: float,
    max_points: int,
) -> dict[str, Any]:
    predictions_dir = save_predictions_npz(predictions, output_ply.parent / "_predictions_for_ply")
    return write_spark_plain_ply_from_npz(
        predictions_dir,
        output_ply,
        frame_stride=frame_stride,
        pixel_stride=pixel_stride,
        conf_percentile=conf_percentile,
        min_conf=min_conf,
        max_points=max_points,
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
) -> dict[str, Any]:
    files = sorted(predictions_dir.glob("frame_*.npz"))
    files = files[:: max(1, int(frame_stride))]
    if not files:
        raise PreviewFailure("LINGBOT_PREDICTIONS_MISSING", f"No frame_*.npz found in {predictions_dir}")

    total_points = 0
    point_source = None
    for path in files:
        points, _, source = extract_npz_frame_points(
            path,
            pixel_stride=pixel_stride,
            conf_percentile=conf_percentile,
            min_conf=min_conf,
        )
        total_points += int(points.shape[0])
        point_source = point_source or source
    if total_points <= 0:
        raise PreviewFailure("LINGBOT_EMPTY_POINT_CLOUD", "LingBot-Map produced no valid 3D points")

    target_points = total_points if max_points <= 0 else min(total_points, int(max_points))
    written = write_spark_plain_ply_records(
        files,
        output_ply,
        total_points=total_points,
        target_points=target_points,
        pixel_stride=pixel_stride,
        conf_percentile=conf_percentile,
        min_conf=min_conf,
    )
    return {
        "point_count": int(written),
        "lingbot_point_source": point_source,
        "lingbot_ply_format": "plain_xyz_rgb",
        "lingbot_points_before_downsample": int(total_points),
        "lingbot_points_after_downsample": int(written),
    }


def write_spark_plain_ply_records(
    files: list[Path],
    output_ply: Path,
    *,
    total_points: int,
    target_points: int,
    pixel_stride: int,
    conf_percentile: float,
    min_conf: float,
) -> int:
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
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {target_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    written = 0
    global_start = 0
    with output_ply.open("wb") as handle:
        handle.write(header)
        for path in files:
            points, colors, _ = extract_npz_frame_points(
                path,
                pixel_stride=pixel_stride,
                conf_percentile=conf_percentile,
                min_conf=min_conf,
            )
            count = int(points.shape[0])
            if target_points < total_points:
                keep = uniform_chunk_indices(global_start, count, total_points, target_points)
                points = points[keep]
                colors = colors[keep]
            global_start += count
            if points.shape[0] == 0:
                continue

            records = np.empty(points.shape[0], dtype=dtype)
            records["x"] = points[:, 0]
            records["y"] = points[:, 1]
            records["z"] = points[:, 2]
            records["red"] = colors[:, 0]
            records["green"] = colors[:, 1]
            records["blue"] = colors[:, 2]
            records.tofile(handle)
            written += int(points.shape[0])

    if written != target_points:
        raise PreviewFailure("LINGBOT_PLY_WRITE_MISMATCH", f"wrote {written} points, expected {target_points}")
    return written


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
) -> tuple[np.ndarray, np.ndarray, str]:
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
) -> tuple[np.ndarray, np.ndarray, str]:
    keys = tuple(data.files if hasattr(data, "files") else data.keys())
    point_key = pick_key(keys, POINT_KEYS)
    if point_key is None:
        raise PreviewFailure("LINGBOT_POINT_FIELD_MISSING", f"{source_name} has no point field")

    stride = max(1, int(pixel_stride))
    points_grid = np.asarray(data[point_key], dtype=np.float32)
    if points_grid.ndim == 4 and points_grid.shape[0] == 1:
        points_grid = points_grid[0]
    if points_grid.ndim != 3 or points_grid.shape[-1] != 3:
        raise PreviewFailure("LINGBOT_POINT_FIELD_INVALID", f"{source_name}: {point_key} shape {points_grid.shape}")
    points = points_grid[::stride, ::stride, :].reshape(-1, 3)

    colors = None
    color_key = pick_key(keys, COLOR_KEYS)
    if color_key is not None:
        image = image_to_hwc_u8(data[color_key])
        if image is not None:
            colors = image[::stride, ::stride, :].reshape(-1, 3)
    if colors is None or colors.shape[0] != points.shape[0]:
        colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)

    mask = np.isfinite(points).all(axis=1)
    conf_key = pick_key(keys, CONF_KEYS_BY_POINT.get(point_key, ("conf",)))
    if conf_key is not None:
        confidence = np.asarray(data[conf_key], dtype=np.float32)
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

    return points[mask].astype("<f4", copy=False), colors[mask].astype(np.uint8, copy=False), point_key


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
