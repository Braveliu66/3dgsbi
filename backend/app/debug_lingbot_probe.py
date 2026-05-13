from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.preview.io.ply import write_gaussian_splat_ply, write_point_cloud_ply
from app.preview.types import PreviewFailure
from app.preview.vendor.lingbot_runtime import (
    compile_lingbot_model,
    extract_video_frames,
    flashinfer_available,
    load_lingbot_model,
    resolve_keyframe_interval,
    resolve_kv_cache_sliding_window,
    resolve_mode,
    resolve_preprocess_mode,
)


_BATCHED_NDIMS = {
    "pose_enc": 3,
    "depth": 5,
    "depth_conf": 4,
    "world_points": 5,
    "world_points_conf": 4,
    "extrinsic": 4,
    "intrinsic": 4,
    "chunk_scales": 2,
    "chunk_transforms": 4,
    "images": 5,
}


def main() -> None:
    args = parse_args()
    if not args.compile:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from lingbot_map.utils.geometry import closed_form_inverse_se3_general, unproject_depth_map_to_point_map
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    if not torch.cuda.is_available():
        raise PreviewFailure("GPU_RESOURCE_UNAVAILABLE", "LingBot probe requires CUDA")

    video_path = Path(args.video)
    model_path = resolve_model_path(args.model_path)
    out_dir = Path(args.out_dir)
    frame_dir = out_dir / "lingbot_probe_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_frame_dir(frame_dir)

    print(f"Extracting frames: {video_path}", flush=True)
    frames = extract_video_frames(video_path, frame_dir, fps=args.fps, max_frames=args.max_frames)
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    print(f"Extracted {len(frame_paths)} frames, source_fps={frames.source_fps}", flush=True)

    preprocess_mode = resolve_preprocess_mode(args.preprocess_mode, frames.width, frames.height)
    print(f"Preprocessing frames at image_size={args.image_size}, mode={preprocess_mode}", flush=True)
    images = load_and_preprocess_images(
        [str(path) for path in frame_paths],
        mode=preprocess_mode,
        image_size=args.image_size,
        patch_size=14,
    )

    device = torch.device("cuda:0")
    frame_count = int(images.shape[0])
    mode = resolve_mode(args.mode, frame_count)
    keyframe_interval = resolve_keyframe_interval(args.keyframe_interval, mode, frame_count)
    has_flashinfer = flashinfer_available()
    use_sdpa = bool(args.use_sdpa or not has_flashinfer)

    print(
        "Loading LingBot model: "
        f"mode={mode}, keyframe_interval={keyframe_interval}, use_sdpa={use_sdpa}",
        flush=True,
    )
    model = load_lingbot_model(
        model_path,
        device,
        mode=mode,
        image_size=args.image_size,
        use_sdpa=use_sdpa,
        camera_iterations=args.camera_iterations,
        num_scale_frames=args.num_scale_frames,
        window_size=args.window_size,
        kv_cache_sliding_window=resolve_kv_cache_sliding_window(args.window_size),
        enable_point=not args.depth_only,
    )

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)

    images_for_inference = images
    if args.images_on_gpu:
        images_for_inference = images.to(device)
    elif torch.cuda.is_available() and hasattr(images_for_inference, "pin_memory"):
        images_for_inference = images_for_inference.pin_memory()
    if args.compile:
        model = compile_lingbot_model(model)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    output_device = torch.device("cpu") if args.offload_to_cpu else None
    print(f"Running inference on {frame_count} frames, dtype={dtype}", flush=True)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if mode == "windowed":
            print_window_plan(
                frame_count=frame_count,
                window_size=args.window_size,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=keyframe_interval,
                overlap_keyframes=args.overlap_keyframes,
                overlap_size=args.overlap_size,
            )
            predictions = call_inference_windowed(
                model,
                images_for_inference,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                overlap_keyframes=args.overlap_keyframes,
                num_scale_frames=args.num_scale_frames,
                scale_mode=args.scale_mode,
                keyframe_interval=keyframe_interval,
                output_device=output_device,
                flow_threshold=args.flow_threshold,
                max_non_keyframe_gap=args.max_non_keyframe_gap,
            )
        else:
            predictions = model.inference_streaming(
                images_for_inference,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=keyframe_interval,
                output_device=output_device,
            )

    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    inference_fps = frame_count / max(inference_seconds, 1e-6)
    peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024 / 1024
    print(f"Inference: {inference_seconds:.3f}s, {inference_fps:.2f} FPS", flush=True)

    predictions, images_cpu = postprocess_predictions(
        predictions,
        images,
        pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
        closed_form_inverse_se3_general=closed_form_inverse_se3_general,
        torch_module=torch,
    )
    pred_np = prepare_for_visualization(predictions, images_cpu)
    keyframe_mask = None
    if args.keyframes_only_points:
        keyframe_mask = build_keyframe_mask(frame_count, args.num_scale_frames, keyframe_interval)
        if np.any(keyframe_mask):
            pred_np.setdefault("is_keyframe", keyframe_mask)
            pred_np.setdefault("frame_type", frame_types_from_keyframes(keyframe_mask, args.num_scale_frames))
        else:
            keyframe_mask = None

    if args.save_predictions:
        predictions_dir = save_predictions_npz(pred_np, out_dir / "predictions.npz")
    else:
        predictions_dir = None

    images_np = pred_np.get("images")
    if images_np is None:
        images_np = to_numpy(images_cpu)

    depth_path = out_dir / "depth_reprojected.ply"
    raw_path = out_dir / "world_points_raw.ply"
    viewable_path = out_dir / "viewable_point_cloud.ply"
    depth_points = unproject_depth_map_to_point_map(
        pred_np["depth"],
        pred_np["extrinsic"],
        pred_np["intrinsic"],
    )

    outputs: dict[str, Any] = {}
    viewer_clouds: list[dict[str, Any]] = []

    source_points = pred_np.get("world_points") if args.viewable_source == "world_points" else depth_points
    source_confidence = (
        pred_np.get("world_points_conf") if args.viewable_source == "world_points" else pred_np.get("depth_conf")
    )
    if source_points is None:
        source_points = depth_points
        source_confidence = pred_np.get("depth_conf")
    viewable_cloud = prepare_viewable_cloud_data(
        points=source_points,
        images=images_np,
        confidence=source_confidence,
        keep_ratio=args.viewable_keep_ratio,
        spatial_keep_quantile=args.spatial_keep_quantile,
        axis_trim_low_quantile=args.axis_trim_low_quantile,
        axis_trim_high_quantile=args.axis_trim_high_quantile,
        max_points=args.max_points,
        frame_mask=keyframe_mask,
    )
    outputs["viewable_point_cloud"] = export_cloud(viewable_cloud, viewable_path)
    viewer_clouds.append({"name": f"viewable_{args.viewable_source}", **viewable_cloud})
    camera_cloud = prepare_camera_cloud(pred_np.get("extrinsic"))
    if camera_cloud is not None:
        viewer_clouds.append({"name": "camera_centers", **camera_cloud})

    depth_cloud = prepare_cloud_data(
        points=depth_points,
        images=images_np,
        confidence=pred_np.get("depth_conf"),
        conf_threshold=args.conf_threshold,
        downsample_factor=args.downsample_factor,
        max_points=args.max_points,
        frame_mask=keyframe_mask,
    )
    outputs["depth_reprojected"] = export_cloud(depth_cloud, depth_path)
    viewer_clouds.append({"name": "depth_reprojected", **depth_cloud})

    if not args.depth_only and "world_points" in pred_np:
        raw_cloud = prepare_cloud_data(
            points=pred_np["world_points"],
            images=images_np,
            confidence=pred_np.get("world_points_conf"),
            conf_threshold=args.conf_threshold,
            downsample_factor=args.downsample_factor,
            max_points=args.max_points,
            frame_mask=keyframe_mask,
        )
        outputs["world_points_raw"] = export_cloud(raw_cloud, raw_path)
        viewer_name = "world_points_keyframes" if keyframe_mask is not None else "world_points_raw"
        viewer_clouds.append({"name": viewer_name, **raw_cloud})

        splat_path = out_dir / "world_points_splat.ply"
        splat_count = write_gaussian_splat_ply(
            raw_cloud["points"],
            raw_cloud["colors"],
            splat_path,
            confidence=raw_cloud["confidence"],
            max_points=0,
            scale=args.splat_scale,
            opacity_logit=args.splat_opacity_logit,
        )
        outputs["world_points_splat"] = {
            "path": str(splat_path),
            "points": splat_count,
            "scale": float(args.splat_scale),
            "opacity_logit": float(args.splat_opacity_logit),
            "size_mb": round(splat_path.stat().st_size / 1024 / 1024, 2),
        }

    if not args.no_viewer and viewer_clouds:
        viewer_path = out_dir / "viewer.html"
        write_viewer_html(
            viewer_path,
            viewer_clouds,
            max_points=args.viewer_max_points,
            point_size=args.viewer_point_size,
        )
        outputs["viewer_html"] = {
            "path": str(viewer_path),
            "size_mb": round(viewer_path.stat().st_size / 1024 / 1024, 2),
            "viewer_max_points": int(args.viewer_max_points),
        }

    metrics = {
        "video_path": str(video_path),
        "model_path": str(model_path),
        "source_fps": frames.source_fps,
        "sampled_fps": frames.sampled_fps,
        "frame_count": frame_count,
        "input_width": frames.width,
        "input_height": frames.height,
        "image_size": args.image_size,
        "preprocess_mode": preprocess_mode,
        "mode": mode,
        "keyframe_interval": keyframe_interval,
        "camera_iterations": args.camera_iterations,
        "num_scale_frames": args.num_scale_frames,
        "scale_mode": args.scale_mode,
        "window_size": args.window_size if mode == "windowed" else None,
        "overlap_keyframes": args.overlap_keyframes if mode == "windowed" else None,
        "overlap_size": args.overlap_size if mode == "windowed" else None,
        "keyframes_only_points": bool(args.keyframes_only_points),
        "keyframe_point_frames": int(np.count_nonzero(keyframe_mask)) if keyframe_mask is not None else None,
        "flashinfer_available": has_flashinfer,
        "use_sdpa": use_sdpa,
        "compile": args.compile,
        "images_on_gpu": bool(args.images_on_gpu),
        "offload_to_cpu": args.offload_to_cpu,
        "depth_only": bool(args.depth_only),
        "predictions_dir": str(predictions_dir) if predictions_dir else None,
        "inference_seconds": round(inference_seconds, 3),
        "inference_fps": round(inference_fps, 3),
        "cuda_peak_allocated_mb": round(peak_allocated_mb, 2),
        "cuda_peak_reserved_mb": round(peak_reserved_mb, 2),
        "outputs": outputs,
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote metrics: {metrics_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local LingBot video probe.")
    parser.add_argument("--video", required=True, help="Video path inside the worker container.")
    parser.add_argument("--model-path", default="/model-cache/lingbot/lingbot-map-long.pt")
    parser.add_argument("--out-dir", default="/app/data/work/lingbot_probe")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--preprocess-mode", choices=["auto", "crop", "pad"], default="auto")
    parser.add_argument("--mode", choices=["auto", "streaming", "windowed"], default="windowed")
    parser.add_argument("--keyframe-interval", type=int, default=1)
    parser.add_argument("--camera-iterations", type=int, default=4)
    parser.add_argument("--num-scale-frames", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--overlap-size", type=int, default=None)
    parser.add_argument("--overlap-keyframes", type=int, default=8)
    parser.add_argument("--scale-mode", choices=["median", "trimmed_mean", "median_all", "trimmed_mean_all"], default="median")
    parser.add_argument("--flow-threshold", type=float, default=0.0)
    parser.add_argument("--max-non-keyframe-gap", type=int, default=30)
    parser.add_argument("--conf-threshold", type=float, default=1.5)
    parser.add_argument("--downsample-factor", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=1_500_000)
    parser.add_argument("--viewable-source", choices=["world_points", "depth"], default="depth")
    parser.add_argument("--viewable-keep-ratio", type=float, default=0.28)
    parser.add_argument("--spatial-keep-quantile", type=float, default=0.995)
    parser.add_argument("--axis-trim-low-quantile", type=float, default=0.0005)
    parser.add_argument("--axis-trim-high-quantile", type=float, default=0.9995)
    parser.add_argument("--keyframes-only-points", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--depth-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-sdpa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--images-on-gpu", action="store_true")
    parser.add_argument("--offload-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--viewer-max-points", type=int, default=350_000)
    parser.add_argument("--viewer-point-size", type=float, default=2.0)
    parser.add_argument("--splat-scale", type=float, default=0.002)
    parser.add_argument("--splat-opacity-logit", type=float, default=-2.0)
    parser.add_argument("--no-viewer", action="store_true")
    return parser.parse_args()


def resolve_model_path(value: str) -> Path:
    requested = Path(value)
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        requested,
        Path(str(value).replace("\\", "/")),
        Path(os.environ.get("MODEL_CACHE_DIR", "")) / "lingbot" / "lingbot-map-long.pt",
        repo_root / "model-cache" / "lingbot" / "lingbot-map-long.pt",
        Path("/model-cache/lingbot/lingbot-map-long.pt"),
    ]

    seen: set[str] = set()
    checked: list[Path] = []
    for candidate in candidates:
        if str(candidate) in {"", "."}:
            continue
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(candidate)
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    checked_text = "\n  - ".join(str(path) for path in checked)
    raise PreviewFailure(
        "LINGBOT_WEIGHT_MISSING",
        f"LingBot weight not found. Requested {value!r}. Checked:\n  - {checked_text}",
    )


def clear_frame_dir(frame_dir: Path) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for path in frame_dir.glob("*.jpg"):
        path.unlink()


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return array


def call_inference_windowed(model: Any, images: Any, **kwargs: Any) -> dict[str, Any]:
    signature = inspect.signature(model.inference_windowed)
    supported = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return model.inference_windowed(images, **supported)


def print_window_plan(
    *,
    frame_count: int,
    window_size: int,
    num_scale_frames: int,
    keyframe_interval: int,
    overlap_keyframes: int | None,
    overlap_size: int | None,
) -> None:
    scale_frames = min(num_scale_frames, frame_count)
    keyframe_interval = max(keyframe_interval, 1)
    if overlap_keyframes is not None:
        overlap = max(scale_frames, overlap_keyframes * keyframe_interval)
    elif overlap_size is not None:
        overlap = overlap_size
    else:
        overlap = scale_frames
    phase2_keyframes = max(window_size - scale_frames, 0)
    actual_window = scale_frames + phase2_keyframes * keyframe_interval
    effective_window = min(actual_window, frame_count)
    step = max(effective_window - overlap, 1)
    if effective_window < frame_count:
        window_count = max(1, (frame_count - overlap + step - 1) // step)
    else:
        window_count = 1
    print(
        "Windowed plan: "
        f"{window_count} windows, window_size={window_size} cache slots, "
        f"actual_window={actual_window} frames, overlap={overlap} frames, "
        f"keyframe_interval={keyframe_interval}",
        flush=True,
    )


def _squeeze_single_batch(key: str, value: Any) -> Any:
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value
    if value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


def postprocess_predictions(
    predictions: dict[str, Any],
    images: Any,
    *,
    pose_encoding_to_extri_intri: Any,
    closed_form_inverse_se3_general: Any,
    torch_module: Any,
) -> tuple[dict[str, Any], Any]:
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

    print("Moving results to CPU...", flush=True)
    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch_module.Tensor):
            predictions[key] = _squeeze_single_batch(
                key,
                predictions[key].to("cpu", non_blocking=True),
            )
    images_cpu = images.to("cpu", non_blocking=True)
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    return predictions, images_cpu


def prepare_for_visualization(predictions: dict[str, Any], images: Any | None = None) -> dict[str, Any]:
    vis_predictions: dict[str, Any] = {}
    for key, value in predictions.items():
        if hasattr(value, "detach"):
            value = _squeeze_single_batch(key, value.detach().cpu()).numpy()
        elif isinstance(value, np.ndarray):
            value = _squeeze_single_batch(key, value)
        vis_predictions[key] = value

    if images is None:
        images = predictions.get("images")
    if hasattr(images, "detach"):
        images = images.detach().cpu()
    if isinstance(images, np.ndarray):
        images = _squeeze_single_batch("images", images)
    elif hasattr(images, "numpy"):
        images = _squeeze_single_batch("images", images).numpy()
    if images is not None:
        vis_predictions["images"] = images
    return vis_predictions


def build_keyframe_mask(frame_count: int, num_scale_frames: int, keyframe_interval: int) -> np.ndarray:
    keyframe_interval = max(int(keyframe_interval), 1)
    scale_frames = min(max(int(num_scale_frames), 0), frame_count)
    mask = np.zeros(frame_count, dtype=np.bool_)
    mask[:scale_frames] = True
    if scale_frames < frame_count:
        frame_indices = np.arange(scale_frames, frame_count)
        mask[scale_frames:] = ((frame_indices - scale_frames) % keyframe_interval) == 0
    return mask


def frame_types_from_keyframes(keyframe_mask: np.ndarray, num_scale_frames: int) -> np.ndarray:
    frame_type = np.full(keyframe_mask.shape[0], "non_keyframe", dtype="<U12")
    frame_type[keyframe_mask] = "keyframe"
    frame_type[: min(int(num_scale_frames), keyframe_mask.shape[0])] = "scale"
    return frame_type


def save_predictions_npz(predictions: dict[str, Any], output_path: Path) -> Path:
    dir_path = output_path.with_suffix("") if output_path.suffix == ".npz" else output_path
    if dir_path.exists():
        for old_frame in dir_path.glob("frame_*.npz"):
            old_frame.unlink()
        old_meta = dir_path / "meta.npz"
        if old_meta.exists():
            old_meta.unlink()
    dir_path.mkdir(parents=True, exist_ok=True)

    sequence_keys: list[str] = []
    metadata: dict[str, Any] = {}
    sequence_length: int | None = None
    for key, value in predictions.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.ndim >= 2 and sequence_length is None:
            sequence_length = int(value.shape[0])
        if value.ndim >= 2 and value.shape[0] == sequence_length:
            sequence_keys.append(key)
        else:
            metadata[key] = value

    if sequence_length is None:
        save_dict = {key: value for key, value in predictions.items() if isinstance(value, np.ndarray)}
        np.savez(dir_path / "frame_000000.npz", **save_dict)
        print(f"Predictions saved to {dir_path} (1 file, {len(save_dict)} keys)", flush=True)
        return dir_path

    def save_frame(frame_index: int) -> None:
        frame_dict = {key: predictions[key][frame_index] for key in sequence_keys}
        np.savez(dir_path / f"frame_{frame_index:06d}.npz", **frame_dict)

    with ThreadPoolExecutor(max_workers=min(32, sequence_length)) as pool:
        list(pool.map(save_frame, range(sequence_length)))
    if metadata:
        np.savez(dir_path / "meta.npz", **metadata)
    print(
        f"Predictions saved to {dir_path} ({sequence_length} frames, {len(sequence_keys)} keys/frame)",
        flush=True,
    )
    return dir_path


def prepare_viewable_cloud_data(
    *,
    points: Any,
    images: Any,
    confidence: Any | None,
    keep_ratio: float,
    spatial_keep_quantile: float,
    axis_trim_low_quantile: float,
    axis_trim_high_quantile: float,
    max_points: int,
    frame_mask: np.ndarray | None,
) -> dict[str, Any]:
    point_array = np.asarray(points, dtype=np.float32)
    image_array = np.asarray(images)
    if image_array.ndim != 4:
        raise PreviewFailure("LINGBOT_IMAGE_COLORS_INVALID", f"unexpected image shape: {image_array.shape}")
    if frame_mask is not None:
        frame_mask = np.asarray(frame_mask, dtype=np.bool_)
        if frame_mask.shape[0] != point_array.shape[0]:
            raise PreviewFailure("LINGBOT_FRAME_MASK_MISMATCH", "frame mask does not match point frame count")
        point_array = point_array[frame_mask]
        image_array = image_array[frame_mask]
        if confidence is not None:
            confidence = np.asarray(confidence)[frame_mask]
    if point_array.shape[:3] != image_array.shape[0:1] + image_array.shape[2:4]:
        raise PreviewFailure(
            "LINGBOT_POINT_IMAGE_SHAPE_MISMATCH",
            f"points {point_array.shape} do not match images {image_array.shape}",
        )

    flat_points = point_array.reshape(-1, 3)
    flat_colors = image_array.transpose(0, 2, 3, 1).reshape(-1, 3)
    if flat_colors.dtype.kind == "f":
        flat_colors = flat_colors * 255.0
    flat_colors = np.clip(flat_colors, 0, 255).astype(np.uint8)
    if confidence is None:
        flat_conf = np.ones(flat_points.shape[0], dtype=np.float32)
    else:
        flat_conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
        if flat_conf.shape[0] != flat_points.shape[0]:
            raise PreviewFailure("LINGBOT_CONF_SHAPE_MISMATCH", "confidence does not match point count")

    valid = np.isfinite(flat_points).all(axis=1) & np.isfinite(flat_conf)
    flat_points = flat_points[valid]
    flat_colors = flat_colors[valid]
    flat_conf = flat_conf[valid]
    before_filter = int(flat_points.shape[0])

    if flat_points.shape[0] == 0:
        raise PreviewFailure("LINGBOT_EMPTY_POINT_CLOUD", "viewable point cloud has no valid points")

    keep_ratio = float(np.clip(keep_ratio, 0.001, 1.0))
    conf_cutoff = float(np.quantile(flat_conf, 1.0 - keep_ratio)) if keep_ratio < 1.0 else float(flat_conf.min())
    keep = flat_conf >= conf_cutoff

    low_q = float(np.clip(axis_trim_low_quantile, 0.0, 0.25))
    high_q = float(np.clip(axis_trim_high_quantile, 0.75, 1.0))
    if low_q > 0.0 or high_q < 1.0:
        low = np.quantile(flat_points[keep], low_q, axis=0)
        high = np.quantile(flat_points[keep], high_q, axis=0)
        keep &= np.all((flat_points >= low) & (flat_points <= high), axis=1)

    spatial_q = float(np.clip(spatial_keep_quantile, 0.5, 1.0))
    if spatial_q < 1.0 and np.count_nonzero(keep) > 8:
        center = np.median(flat_points[keep], axis=0)
        distance = np.linalg.norm(flat_points - center, axis=1)
        radius = float(np.quantile(distance[keep], spatial_q))
        keep &= distance <= radius

    flat_points = flat_points[keep]
    flat_colors = flat_colors[keep]
    flat_conf = flat_conf[keep]

    before_cap = int(flat_points.shape[0])
    if max_points > 0 and flat_points.shape[0] > max_points:
        selected = np.argpartition(flat_conf, -max_points)[-max_points:]
        flat_points = flat_points[selected]
        flat_colors = flat_colors[selected]
        flat_conf = flat_conf[selected]

    return {
        "points": flat_points.astype(np.float32, copy=False),
        "colors": flat_colors.astype(np.uint8, copy=False),
        "confidence": flat_conf.astype(np.float32, copy=False),
        "points_before_filter": before_filter,
        "points_before_cap": before_cap,
        "conf_threshold": conf_cutoff,
        "conf_threshold_used": keep_ratio < 1.0,
        "downsample_factor": 1,
        "viewable_keep_ratio": keep_ratio,
        "spatial_keep_quantile": spatial_q,
        "axis_trim_low_quantile": low_q,
        "axis_trim_high_quantile": high_q,
    }


def prepare_camera_cloud(extrinsic: Any) -> dict[str, Any] | None:
    if extrinsic is None:
        return None
    extrinsic_array = np.asarray(extrinsic, dtype=np.float32)
    if extrinsic_array.ndim != 3 or extrinsic_array.shape[-2:] != (3, 4):
        return None
    centers = extrinsic_array[:, :3, 3]
    valid = np.isfinite(centers).all(axis=1)
    centers = centers[valid]
    if centers.shape[0] == 0:
        return None

    colors = np.zeros((centers.shape[0], 3), dtype=np.uint8)
    colors[:, 0] = 255
    if centers.shape[0] > 1:
        ramp = np.linspace(0, 255, centers.shape[0], dtype=np.uint8)
        colors[:, 1] = ramp
    confidence = np.ones(centers.shape[0], dtype=np.float32)
    return {
        "points": centers.astype(np.float32, copy=False),
        "colors": colors,
        "confidence": confidence,
        "points_before_cap": int(centers.shape[0]),
        "conf_threshold": 0.0,
        "conf_threshold_used": False,
        "downsample_factor": 1,
    }


def prepare_cloud_data(
    *,
    points: Any,
    images: Any,
    confidence: Any | None,
    conf_threshold: float,
    downsample_factor: int,
    max_points: int,
    frame_mask: np.ndarray | None,
) -> dict[str, Any]:
    point_array = np.asarray(points, dtype=np.float32)
    image_array = np.asarray(images)
    if image_array.ndim != 4:
        raise PreviewFailure("LINGBOT_IMAGE_COLORS_INVALID", f"unexpected image shape: {image_array.shape}")
    if frame_mask is not None:
        frame_mask = np.asarray(frame_mask, dtype=np.bool_)
        if frame_mask.shape[0] != point_array.shape[0]:
            raise PreviewFailure("LINGBOT_FRAME_MASK_MISMATCH", "frame mask does not match point frame count")
        point_array = point_array[frame_mask]
        image_array = image_array[frame_mask]
        if confidence is not None:
            confidence = np.asarray(confidence)[frame_mask]
    if point_array.shape[:3] != image_array.shape[0:1] + image_array.shape[2:4]:
        raise PreviewFailure(
            "LINGBOT_POINT_IMAGE_SHAPE_MISMATCH",
            f"points {point_array.shape} do not match images {image_array.shape}",
        )

    flat_points = point_array.reshape(-1, 3)
    flat_colors = image_array.transpose(0, 2, 3, 1).reshape(-1, 3)
    if flat_colors.dtype.kind == "f":
        flat_colors = flat_colors * 255.0
    flat_colors = np.clip(flat_colors, 0, 255).astype(np.uint8)

    if confidence is None:
        flat_conf = np.ones(flat_points.shape[0], dtype=np.float32)
    else:
        flat_conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
        if flat_conf.shape[0] != flat_points.shape[0]:
            raise PreviewFailure("LINGBOT_CONF_SHAPE_MISMATCH", "confidence does not match point count")

    valid = np.isfinite(flat_points).all(axis=1) & np.isfinite(flat_conf)
    threshold_valid = valid & (flat_conf >= float(conf_threshold))
    used_threshold = bool(np.any(threshold_valid))
    if used_threshold:
        valid = threshold_valid

    flat_points = flat_points[valid]
    flat_colors = flat_colors[valid]
    flat_conf = flat_conf[valid]
    if downsample_factor > 1 and flat_points.shape[0] > 0:
        keep = np.arange(0, flat_points.shape[0], int(downsample_factor))
        flat_points = flat_points[keep]
        flat_colors = flat_colors[keep]
        flat_conf = flat_conf[keep]

    before_cap = int(flat_points.shape[0])
    if max_points > 0 and flat_points.shape[0] > max_points:
        keep = np.argpartition(flat_conf, -max_points)[-max_points:]
        flat_points = flat_points[keep]
        flat_colors = flat_colors[keep]
        flat_conf = flat_conf[keep]

    return {
        "points": flat_points.astype(np.float32, copy=False),
        "colors": flat_colors.astype(np.uint8, copy=False),
        "confidence": flat_conf.astype(np.float32, copy=False),
        "points_before_cap": before_cap,
        "conf_threshold": float(conf_threshold),
        "conf_threshold_used": used_threshold,
        "downsample_factor": int(downsample_factor),
    }


def export_cloud(cloud: dict[str, Any], output_path: Path) -> dict[str, Any]:
    count = write_point_cloud_ply(
        cloud["points"],
        cloud["colors"],
        output_path,
        confidence=cloud["confidence"],
        max_points=0,
    )
    metrics = {
        "path": str(output_path),
        "points": count,
        "points_before_cap": cloud["points_before_cap"],
        "conf_threshold": cloud["conf_threshold"],
        "conf_threshold_used": cloud["conf_threshold_used"],
        "downsample_factor": cloud["downsample_factor"],
        "size_mb": round(output_path.stat().st_size / 1024 / 1024, 2),
    }
    for key in (
        "points_before_filter",
        "viewable_keep_ratio",
        "spatial_keep_quantile",
        "axis_trim_low_quantile",
        "axis_trim_high_quantile",
    ):
        if key in cloud:
            metrics[key] = cloud[key]
    return metrics


def write_viewer_html(
    output_path: Path,
    clouds: list[dict[str, Any]],
    *,
    max_points: int,
    point_size: float,
) -> None:
    payload = {
        "point_size": float(point_size),
        "clouds": [viewer_cloud_payload(cloud, max_points=max_points) for cloud in clouds],
    }
    html = VIEWER_HTML.replace("__VIEWER_DATA__", json.dumps(payload, separators=(",", ":")))
    output_path.write_text(html, encoding="utf-8")


def viewer_cloud_payload(cloud: dict[str, Any], *, max_points: int) -> dict[str, Any]:
    points = np.asarray(cloud["points"], dtype=np.float32)
    colors = np.asarray(cloud["colors"], dtype=np.uint8)
    confidence = np.asarray(cloud["confidence"], dtype=np.float32)
    point_count = int(points.shape[0])
    if max_points > 0 and point_count > max_points:
        keep = np.linspace(0, point_count - 1, int(max_points), dtype=np.int64)
        points = points[keep]
        colors = colors[keep]
        confidence = confidence[keep]

    if points.size and points.shape[0] > 32:
        center = np.median(points, axis=0).astype(np.float32)
        distance = np.linalg.norm(points - center, axis=1)
        display_radius = float(np.quantile(distance, 0.995))
        if np.isfinite(display_radius) and display_radius > 0:
            display = distance <= display_radius
            points = points[display]
            colors = colors[display]
            confidence = confidence[display]
    bounds_min = points.min(axis=0) if points.size else np.zeros(3, dtype=np.float32)
    bounds_max = points.max(axis=0) if points.size else np.ones(3, dtype=np.float32)
    center = np.median(points, axis=0).astype(np.float32) if points.size else ((bounds_min + bounds_max) * 0.5)
    shifted = points - center
    radius = float(np.quantile(np.linalg.norm(shifted, axis=1), 0.995)) if points.size else 1.0
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    normalized = (shifted / radius).astype(np.float32, copy=False)

    return {
        "name": cloud["name"],
        "points": int(point_count),
        "shown": int(normalized.shape[0]),
        "center": [float(value) for value in center],
        "radius": radius,
        "conf_min": float(confidence.min()) if confidence.size else 0.0,
        "conf_max": float(confidence.max()) if confidence.size else 0.0,
        "positions": encode_array(normalized),
        "colors": encode_array(colors),
    }


def encode_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


VIEWER_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LingBot Probe Viewer</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #101412; color: #ecf1eb; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    canvas { display: block; width: 100vw; height: 100vh; cursor: grab; }
    canvas:active { cursor: grabbing; }
    .panel { position: fixed; left: 16px; top: 16px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; padding: 10px 12px; border: 1px solid rgba(255,255,255,.14); background: rgba(13,18,16,.82); backdrop-filter: blur(10px); border-radius: 6px; }
    select, input, button { color: #ecf1eb; background: #18211d; border: 1px solid rgba(255,255,255,.18); border-radius: 4px; height: 30px; }
    select { padding: 0 8px; }
    input { width: 120px; }
    button { padding: 0 10px; }
    .stats { position: fixed; left: 16px; bottom: 16px; max-width: min(620px, calc(100vw - 32px)); padding: 10px 12px; border: 1px solid rgba(255,255,255,.12); background: rgba(13,18,16,.78); border-radius: 6px; font-size: 13px; line-height: 1.45; }
  </style>
</head>
<body>
  <canvas id="view"></canvas>
  <div class="panel">
    <select id="cloud"></select>
    <label>Point size <input id="pointSize" type="range" min="0.5" max="8" step="0.1"></label>
    <button id="reset" type="button">Reset view</button>
  </div>
  <div id="stats" class="stats"></div>
  <script id="viewer-data" type="application/json">__VIEWER_DATA__</script>
  <script>
const data = JSON.parse(document.getElementById("viewer-data").textContent);
const canvas = document.getElementById("view");
const gl = canvas.getContext("webgl", { antialias: false, alpha: false });
if (!gl) throw new Error("WebGL is unavailable");

const vs = `
attribute vec3 aPosition;
attribute vec3 aColor;
uniform mat4 uMatrix;
uniform float uPointSize;
varying vec3 vColor;
void main() {
  gl_Position = uMatrix * vec4(aPosition, 1.0);
  gl_PointSize = uPointSize;
  vColor = aColor / 255.0;
}`;
const fs = `
precision mediump float;
varying vec3 vColor;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  gl_FragColor = vec4(vColor, 1.0);
}`;

const program = createProgram(vs, fs);
gl.useProgram(program);
const attribs = {
  position: gl.getAttribLocation(program, "aPosition"),
  color: gl.getAttribLocation(program, "aColor")
};
const uniforms = {
  matrix: gl.getUniformLocation(program, "uMatrix"),
  pointSize: gl.getUniformLocation(program, "uPointSize")
};
const positionBuffer = gl.createBuffer();
const colorBuffer = gl.createBuffer();
const cloudSelect = document.getElementById("cloud");
const pointSizeInput = document.getElementById("pointSize");
const stats = document.getElementById("stats");
let currentCloud = null;
let yaw = 0.7;
let pitch = 0.25;
let distance = 2.8;
let dragging = false;
let lastX = 0;
let lastY = 0;

for (const [index, cloud] of data.clouds.entries()) {
  const option = document.createElement("option");
  option.value = String(index);
  option.textContent = cloud.name;
  cloudSelect.appendChild(option);
}
pointSizeInput.value = String(data.point_size || 2);
cloudSelect.addEventListener("change", () => setCloud(Number(cloudSelect.value)));
pointSizeInput.addEventListener("input", render);
document.getElementById("reset").addEventListener("click", () => { yaw = 0.7; pitch = 0.25; distance = 2.8; render(); });
canvas.addEventListener("mousedown", event => { dragging = true; lastX = event.clientX; lastY = event.clientY; });
window.addEventListener("mouseup", () => { dragging = false; });
window.addEventListener("mousemove", event => {
  if (!dragging) return;
  yaw += (event.clientX - lastX) * 0.008;
  pitch = clamp(pitch + (event.clientY - lastY) * 0.008, -1.45, 1.45);
  lastX = event.clientX;
  lastY = event.clientY;
  render();
});
canvas.addEventListener("wheel", event => {
  event.preventDefault();
  distance = clamp(distance * Math.exp(event.deltaY * 0.001), 0.35, 20);
  render();
}, { passive: false });
window.addEventListener("resize", render);
setCloud(0);

function setCloud(index) {
  currentCloud = data.clouds[index];
  const positions = decodeFloat32(currentCloud.positions);
  const colors = decodeUint8(currentCloud.colors);
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
  stats.textContent = `${currentCloud.name}: showing ${currentCloud.shown.toLocaleString()} / ${currentCloud.points.toLocaleString()} points, source center [${currentCloud.center.map(v => v.toFixed(3)).join(", ")}], radius ${currentCloud.radius.toFixed(3)}, confidence ${currentCloud.conf_min.toFixed(3)} - ${currentCloud.conf_max.toFixed(3)}`;
  render();
}

function render() {
  if (!currentCloud) return;
  const width = canvas.clientWidth * window.devicePixelRatio;
  const height = canvas.clientHeight * window.devicePixelRatio;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.06, 0.08, 0.07, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.useProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.enableVertexAttribArray(attribs.position);
  gl.vertexAttribPointer(attribs.position, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.enableVertexAttribArray(attribs.color);
  gl.vertexAttribPointer(attribs.color, 3, gl.UNSIGNED_BYTE, false, 0, 0);

  const aspect = canvas.width / Math.max(canvas.height, 1);
  const projection = perspective(Math.PI / 4, aspect, 0.01, 100);
  const eye = [
    distance * Math.sin(yaw) * Math.cos(pitch),
    distance * Math.sin(pitch),
    distance * Math.cos(yaw) * Math.cos(pitch)
  ];
  const view = lookAt(eye, [0, 0, 0], [0, 1, 0]);
  gl.uniformMatrix4fv(uniforms.matrix, false, multiply(projection, view));
  gl.uniform1f(uniforms.pointSize, Number(pointSizeInput.value) * window.devicePixelRatio);
  gl.drawArrays(gl.POINTS, 0, currentCloud.shown);
}

function createProgram(vertexSource, fragmentSource) {
  const vertex = compileShader(gl.VERTEX_SHADER, vertexSource);
  const fragment = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
  const linked = gl.createProgram();
  gl.attachShader(linked, vertex);
  gl.attachShader(linked, fragment);
  gl.linkProgram(linked);
  if (!gl.getProgramParameter(linked, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(linked));
  return linked;
}
function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}
function decodeFloat32(value) {
  const bytes = decodeBytes(value);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}
function decodeUint8(value) {
  return decodeBytes(value);
}
function decodeBytes(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0
  ]);
}
function lookAt(eye, target, up) {
  const z = normalize(subtract(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
  ]);
}
function multiply(a, b) {
  const out = new Float32Array(16);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      out[col * 4 + row] =
        a[0 * 4 + row] * b[col * 4 + 0] +
        a[1 * 4 + row] * b[col * 4 + 1] +
        a[2 * 4 + row] * b[col * 4 + 2] +
        a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return out;
}
function subtract(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function normalize(v) {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / length, v[1] / length, v[2] / length];
}
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    try:
        main()
    except PreviewFailure as exc:
        raise SystemExit(f"{exc.code}: {exc.message}") from exc
