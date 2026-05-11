from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

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
DEFAULT_LINGBOT_MODEL_IMAGE_SIZE = 518
DEFAULT_LINGBOT_MIN_INFERENCE_FPS = 3.0
OOM_FALLBACK_IMAGE_SIZE = 448
OOM_FALLBACK_MAX_FRAMES = 96
OOM_FALLBACK_WINDOW_SIZE = 32
OOM_FALLBACK_KEYFRAME_INTERVAL = 2
DEFAULT_KV_CACHE_SLIDING_WINDOW = 16
OOM_FALLBACK_KV_CACHE_SLIDING_WINDOW = 8


@dataclass(frozen=True, slots=True)
class LingBotInferenceProfile:
    image_size: int
    max_frames: int
    mode: str
    keyframe_interval: int | None
    camera_iterations: int
    num_scale_frames: int
    window_size: int
    kv_cache_sliding_window: int
    overlap_keyframes: int
    oom_fallback: bool = False


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
    keyframes_only_points: bool = False,
    allow_sdpa_fallback: bool = False,
    min_inference_fps: float = DEFAULT_LINGBOT_MIN_INFERENCE_FPS,
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

    device = torch.device("cuda:0")
    use_sdpa, flashinfer_found = resolve_lingbot_attention_backend(
        allow_sdpa_fallback=allow_sdpa_fallback,
    )
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    profile = LingBotInferenceProfile(
        image_size=image_size,
        max_frames=max_frames,
        mode=mode,
        keyframe_interval=keyframe_interval,
        camera_iterations=camera_iterations,
        num_scale_frames=num_scale_frames,
        window_size=window_size,
        kv_cache_sliding_window=resolve_kv_cache_sliding_window(window_size),
        overlap_keyframes=overlap_keyframes,
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
        if not is_cuda_out_of_memory(exc, torch_module=torch):
            raise PreviewFailure("LINGBOT_INFERENCE_FAILED", f"LingBot-Map inference failed: {exc}") from exc
        release_cuda_exception(exc, torch_module=torch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fallback_profile = make_lingbot_oom_fallback_profile(profile)
        progress(
            "lingbot_inference_retry",
            46,
            (
                "CUDA out of memory; retrying LingBot-Map with "
                f"image_size={fallback_profile.image_size}, "
                f"max_frames={fallback_profile.max_frames}, "
                f"window_size={fallback_profile.window_size}, "
                f"keyframe_interval={fallback_profile.keyframe_interval}"
            ),
        )
        try:
            predictions, images, inference_metrics = run_lingbot_inference_profile(
                frame_paths=frame_paths,
                model_path=model_path,
                device=device,
                profile=fallback_profile,
                use_sdpa=use_sdpa,
                flashinfer_found=flashinfer_found,
                allow_sdpa_fallback=allow_sdpa_fallback,
                dtype=dtype,
                compile_requested=False,
                min_inference_fps=min_inference_fps,
                load_and_preprocess_images=load_and_preprocess_images,
                torch_module=torch,
                progress=progress,
            )
        except PreviewFailure:
            raise
        except Exception as retry_exc:
            if is_cuda_out_of_memory(retry_exc, torch_module=torch):
                raise PreviewFailure(
                    "LINGBOT_CUDA_OOM",
                    "LingBot-Map inference still ran out of CUDA memory after the safe fallback profile",
                ) from retry_exc
            raise PreviewFailure("LINGBOT_INFERENCE_FAILED", f"LingBot-Map inference failed after OOM fallback: {retry_exc}") from retry_exc

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
        num_scale_frames=int(inference_metrics["lingbot_num_scale_frames"]),
        keyframe_interval=int(inference_metrics["lingbot_keyframe_interval"]),
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
            keyframes_only_points=keyframes_only_points,
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
            keyframes_only_points=keyframes_only_points,
        )
    point_count = int(point_metrics["point_count"])

    del pred_np
    del predictions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        "lingbot_keyframes_only_points": bool(keyframes_only_points),
        "lingbot_predictions_dir": str(predictions_dir) if predictions_dir else None,
        "point_count": point_count,
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


def select_lingbot_frame_paths(frame_paths: list[Path], max_frames: int) -> list[Path]:
    if max_frames <= 0 or len(frame_paths) <= max_frames:
        return frame_paths
    indices = np.linspace(0, len(frame_paths) - 1, max_frames, dtype=np.int64)
    return [frame_paths[int(index)] for index in indices]


def make_lingbot_oom_fallback_profile(profile: LingBotInferenceProfile) -> LingBotInferenceProfile:
    fallback_max_frames = OOM_FALLBACK_MAX_FRAMES if profile.max_frames <= 0 else min(profile.max_frames, OOM_FALLBACK_MAX_FRAMES)
    fallback_keyframe_interval = max(profile.keyframe_interval or 0, OOM_FALLBACK_KEYFRAME_INTERVAL)
    return LingBotInferenceProfile(
        image_size=min(profile.image_size, OOM_FALLBACK_IMAGE_SIZE),
        max_frames=fallback_max_frames,
        mode=profile.mode,
        keyframe_interval=fallback_keyframe_interval,
        camera_iterations=1,
        num_scale_frames=min(profile.num_scale_frames, 2),
        window_size=min(profile.window_size, OOM_FALLBACK_WINDOW_SIZE),
        kv_cache_sliding_window=min(profile.kv_cache_sliding_window, OOM_FALLBACK_KV_CACHE_SLIDING_WINDOW),
        overlap_keyframes=min(profile.overlap_keyframes, 4),
        oom_fallback=True,
    )


def resolve_kv_cache_sliding_window(window_size: int) -> int:
    return max(4, min(int(window_size), DEFAULT_KV_CACHE_SLIDING_WINDOW))


def resolve_lingbot_attention_backend(
    *,
    allow_sdpa_fallback: bool,
    flashinfer_probe: Callable[[], bool] = flashinfer_available,
) -> tuple[bool, bool]:
    flashinfer_found = flashinfer_probe()
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
    selected_frame_paths = select_lingbot_frame_paths(frame_paths, profile.max_frames)
    progress(
        "lingbot_preprocess",
        34,
        f"preprocessing {len(selected_frame_paths)} frames at image size {profile.image_size}",
    )
    try:
        images = load_and_preprocess_images(
            [str(path) for path in selected_frame_paths],
            mode="crop",
            image_size=profile.image_size,
            patch_size=14,
        )
    except Exception as exc:
        raise PreviewFailure("LINGBOT_PREPROCESS_FAILED", f"LingBot-Map preprocessing failed: {exc}") from exc

    frame_count = int(images.shape[0])
    resolved_mode = resolve_mode(profile.mode, frame_count)
    resolved_keyframe_interval = resolve_keyframe_interval(profile.keyframe_interval, resolved_mode, frame_count)
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
        enable_point=False,
    )
    model_image_size = int(getattr(model, "_lingbot_model_image_size", DEFAULT_LINGBOT_MODEL_IMAGE_SIZE))
    aggregator_dtype = cast_lingbot_aggregator_for_inference(model, dtype)
    compile_active = False
    compile_fallback = False
    if compile_requested:
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
        with torch_module.no_grad(), torch_module.amp.autocast("cuda", dtype=dtype):
            predictions = run_lingbot_inference(
                model,
                images,
                resolved_mode=resolved_mode,
                window_size=profile.window_size,
                overlap_keyframes=profile.overlap_keyframes,
                num_scale_frames=profile.num_scale_frames,
                keyframe_interval=resolved_keyframe_interval,
                output_device=output_device,
                torch_module=torch_module,
            )
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
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
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
            enable_point=False,
        )
        model_image_size = int(getattr(model, "_lingbot_model_image_size", DEFAULT_LINGBOT_MODEL_IMAGE_SIZE))
        aggregator_dtype = cast_lingbot_aggregator_for_inference(model, dtype)
        predictions, inference_seconds, inference_fps, peak_mb = _infer_once()
    finally:
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()

    metrics = {
        "lingbot_image_size": int(profile.image_size),
        "lingbot_model_image_size": model_image_size,
        "lingbot_inference_frames": frame_count,
        "lingbot_inference_mode": resolved_mode,
        "lingbot_keyframe_interval": int(resolved_keyframe_interval),
        "lingbot_camera_iterations": int(profile.camera_iterations),
        "lingbot_num_scale_frames": int(profile.num_scale_frames),
        "lingbot_window_size": int(profile.window_size) if resolved_mode == "windowed" else None,
        "lingbot_kv_cache_sliding_window": int(profile.kv_cache_sliding_window),
        "lingbot_overlap_keyframes": int(profile.overlap_keyframes) if resolved_mode == "windowed" else None,
        "lingbot_use_sdpa": bool(use_sdpa),
        "lingbot_flashinfer_available": bool(flashinfer_found),
        "lingbot_allow_sdpa_fallback": bool(allow_sdpa_fallback),
        "lingbot_sdpa_fallback_active": bool(use_sdpa),
        "lingbot_enable_point": False,
        "lingbot_aggregator_dtype": str(aggregator_dtype).replace("torch.", ""),
        "lingbot_compile": bool(compile_active),
        "lingbot_compile_requested": bool(compile_requested),
        "lingbot_compile_cudagraphs": False,
        "lingbot_compile_fallback": compile_fallback,
        "lingbot_max_frames": int(profile.max_frames),
        "lingbot_oom_fallback": bool(profile.oom_fallback),
        "lingbot_inference_seconds": round(inference_seconds, 3),
        "lingbot_inference_fps": round(inference_fps, 3),
        "cuda_memory_peak_mb": round(peak_mb, 2),
    }
    return predictions, images, metrics


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
            torch_module.cuda.empty_cache()


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
    if min_inference_fps > 0 and inference_fps < min_inference_fps:
        raise PreviewFailure(
            "LINGBOT_INFERENCE_FPS_BELOW_TARGET",
            (
                "LingBot-Map model inference was "
                f"{inference_fps:.2f} FPS, below the {min_inference_fps:.2f} FPS target for 12GB Ampere+ GPUs"
            ),
        )


def is_cuda_out_of_memory(error: Exception, *, torch_module: Any) -> bool:
    oom_type = getattr(getattr(torch_module, "cuda", None), "OutOfMemoryError", None)
    if oom_type is not None and isinstance(error, oom_type):
        return True
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def release_cuda_exception(error: Exception, *, torch_module: Any) -> None:
    error.__traceback__ = None
    error.__context__ = None
    error.__cause__ = None
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


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
    enable_point: bool = False,
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
        model = GCTStream(
            img_size=model_image_size,
            patch_size=14,
            enable_3d_rope=True,
            max_frame_num=max(1024, window_size * 16),
            kv_cache_sliding_window=kv_cache_sliding_window,
            kv_cache_scale_frames=num_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=use_sdpa,
            camera_num_iterations=camera_iterations,
            enable_point=enable_point,
            enable_depth=True,
        )
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
) -> dict[str, Any]:
    frame_indices = strided_frame_indices(prediction_frame_count(predictions), frame_stride)
    selected_indices = []
    total_points = 0
    point_source = None
    for frame_index, frame in iter_prediction_frames(predictions, frame_indices):
        if not should_export_prediction_frame(frame, keyframes_only_points=keyframes_only_points):
            continue
        points, _, source = extract_frame_points(
            frame,
            pixel_stride=pixel_stride,
            conf_percentile=conf_percentile,
            min_conf=min_conf,
            source_name=f"in-memory frame {frame_index}",
        )
        total_points += int(points.shape[0])
        point_source = point_source or source
        selected_indices.append(frame_index)

    if not selected_indices:
        raise PreviewFailure("LINGBOT_PREDICTIONS_MISSING", "No LingBot-Map frames selected for point export")
    if total_points <= 0:
        raise PreviewFailure("LINGBOT_EMPTY_POINT_CLOUD", "LingBot-Map produced no valid 3D points")

    target_points = total_points if max_points <= 0 else min(total_points, int(max_points))
    written = write_spark_plain_ply_records(
        selected_indices,
        output_ply,
        total_points=total_points,
        target_points=target_points,
        extract_points=lambda frame_index: extract_prediction_frame_points(
            predictions,
            frame_index,
            pixel_stride=pixel_stride,
            conf_percentile=conf_percentile,
            min_conf=min_conf,
        ),
    )
    return {
        "point_count": int(written),
        "lingbot_point_source": point_source,
        "lingbot_ply_format": "gaussian_splat",
        "lingbot_points_before_downsample": int(total_points),
        "lingbot_points_after_downsample": int(written),
        "lingbot_point_frame_count": len(selected_indices),
    }


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
) -> dict[str, Any]:
    files = sorted(predictions_dir.glob("frame_*.npz"))
    files = files[:: max(1, int(frame_stride))]
    if not files:
        raise PreviewFailure("LINGBOT_PREDICTIONS_MISSING", f"No frame_*.npz found in {predictions_dir}")

    selected_files = []
    total_points = 0
    point_source = None
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if not should_export_prediction_frame(data, keyframes_only_points=keyframes_only_points):
                continue
            points, _, source = extract_frame_points(
                data,
                pixel_stride=pixel_stride,
                conf_percentile=conf_percentile,
                min_conf=min_conf,
                source_name=str(path),
            )
        total_points += int(points.shape[0])
        point_source = point_source or source
        selected_files.append(path)
    if not selected_files:
        raise PreviewFailure("LINGBOT_PREDICTIONS_MISSING", f"No frame_*.npz selected in {predictions_dir}")
    if total_points <= 0:
        raise PreviewFailure("LINGBOT_EMPTY_POINT_CLOUD", "LingBot-Map produced no valid 3D points")

    target_points = total_points if max_points <= 0 else min(total_points, int(max_points))
    written = write_spark_plain_ply_records(
        selected_files,
        output_ply,
        total_points=total_points,
        target_points=target_points,
        extract_points=lambda path: extract_npz_frame_points(
            path,
            pixel_stride=pixel_stride,
            conf_percentile=conf_percentile,
            min_conf=min_conf,
        ),
    )
    return {
        "point_count": int(written),
        "lingbot_point_source": point_source,
        "lingbot_ply_format": "gaussian_splat",
        "lingbot_points_before_downsample": int(total_points),
        "lingbot_points_after_downsample": int(written),
        "lingbot_point_frame_count": len(selected_files),
    }


def write_spark_plain_ply_records(
    frame_sources: list[Any],
    output_ply: Path,
    *,
    total_points: int,
    target_points: int,
    extract_points: Callable[[Any], tuple[np.ndarray, np.ndarray, str]],
) -> int:
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    dtype = gaussian_splat_record_dtype()
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {target_points}\n"
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

    written = 0
    global_start = 0
    with output_ply.open("wb") as handle:
        handle.write(header)
        for source in frame_sources:
            points, colors, _ = extract_points(source)
            count = int(points.shape[0])
            if target_points < total_points:
                keep = uniform_chunk_indices(global_start, count, total_points, target_points)
                points = points[keep]
                colors = colors[keep]
            global_start += count
            if points.shape[0] == 0:
                continue

            records = gaussian_splat_records(points, colors, dtype=dtype)
            records.tofile(handle)
            written += int(points.shape[0])

    if written != target_points:
        raise PreviewFailure("LINGBOT_PLY_WRITE_MISMATCH", f"wrote {written} points, expected {target_points}")
    return written


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


def gaussian_splat_records(points: np.ndarray, colors: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    records = np.zeros(points.shape[0], dtype=dtype)
    records["x"] = points[:, 0]
    records["y"] = points[:, 1]
    records["z"] = points[:, 2]
    sh_c0 = np.float32(0.28209479177387814)
    sh_dc = (colors.astype(np.float32) / 255.0 - 0.5) / sh_c0
    records["f_dc_0"] = sh_dc[:, 0]
    records["f_dc_1"] = sh_dc[:, 1]
    records["f_dc_2"] = sh_dc[:, 2]
    log_scale = np.float32(np.log(0.002))
    records["opacity"] = np.float32(-2.0)
    records["scale_0"] = log_scale
    records["scale_1"] = log_scale
    records["scale_2"] = log_scale
    records["rot_0"] = np.float32(1.0)
    return records


def extract_prediction_frame_points(
    predictions: dict[str, np.ndarray],
    frame_index: int,
    *,
    pixel_stride: int,
    conf_percentile: float,
    min_conf: float,
) -> tuple[np.ndarray, np.ndarray, str]:
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
