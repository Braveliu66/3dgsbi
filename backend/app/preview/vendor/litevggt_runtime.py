from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.preview.io.ply import fixed_preview_radius, write_point_cloud_ply
from app.preview.types import PreviewFailure
from app.preview.utils import VENDOR_ROOT, image_files, prepend_sys_path


Progress = Callable[[str, int, str], None]
IDLE_UNLOAD_SECONDS = 60.0
_CPU_MODEL = None
_CPU_MODEL_CHECKPOINT_PATH: Path | None = None
_GPU_LOADED = False
_MODEL_LOCK = threading.RLock()
_IDLE_UNLOAD_TIMER: threading.Timer | None = None
_IDLE_UNLOAD_TOKEN = 0
_LAST_MODEL_CACHE_METRICS: dict[str, Any] = {
    "litevggt_cpu_model_cached": False,
    "litevggt_gpu_loaded_from_cpu": False,
    "litevggt_model_loaded_from_disk": False,
    "litevggt_gpu_idle_unload_seconds": IDLE_UNLOAD_SECONDS,
}


@dataclass(slots=True)
class LiteVGGTReconstruction:
    files: list[Path]
    frame_indices: np.ndarray
    images: np.ndarray
    valid_masks: np.ndarray
    w2c: np.ndarray
    intrinsics: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    point_frame_indices: np.ndarray
    metrics: dict[str, Any]


@dataclass(slots=True)
class LiteVGGTBatchResult:
    frame_indices: np.ndarray
    images: np.ndarray
    valid_masks: np.ndarray
    w2c: np.ndarray
    intrinsics: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    point_frame_indices: np.ndarray
    valid_pixel_count: int
    point_count_before_filter: int
    point_count_after_filter: int


@dataclass(slots=True)
class LiteVGGTQualitySettings:
    target_size: int
    keep_ratio: float
    quality_profile: str
    keep_ratio_source: str
    target_size_source: str


@dataclass(slots=True)
class LiteVGGTFrameSelection:
    files: list[Path]
    frame_indices: np.ndarray
    frame_stride: int
    frame_stride_source: str
    frame_budget: int


@dataclass(slots=True)
class LiteVGGTImageBatch:
    tensors: list[Any]
    valid_masks: np.ndarray
    preprocess_mode: str


@dataclass(slots=True)
class LiteVGGTWindowSpec:
    index: int
    start: int
    end: int
    files: list[Path]
    frame_indices: np.ndarray


@dataclass(slots=True)
class Sim3Transform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(slots=True)
class LiteVGGTWindowState:
    spec: LiteVGGTWindowSpec
    transform: Sim3Transform
    points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    point_frame_indices: np.ndarray
    alignment_points: np.ndarray
    alignment_frame_indices: np.ndarray
    local_centers_by_frame: dict[int, np.ndarray]


@dataclass(slots=True)
class LiteVGGTLoopEdge:
    source: int
    target: int
    delta: np.ndarray
    score: float


@dataclass(slots=True)
class LiteVGGTIcpResult:
    transform: Sim3Transform
    residual: float
    inlier_ratio: float
    accepted: bool
    reason: str


def get_litevggt_model_on_gpu(checkpoint_path: Path):
    import torch

    global _CPU_MODEL, _CPU_MODEL_CHECKPOINT_PATH, _GPU_LOADED, _LAST_MODEL_CACHE_METRICS

    resolved_checkpoint = Path(checkpoint_path).resolve()
    with _MODEL_LOCK:
        cancel_litevggt_idle_unload()
        loaded_from_disk = False
        if _CPU_MODEL is None or _CPU_MODEL_CHECKPOINT_PATH != resolved_checkpoint:
            if _CPU_MODEL is not None and _GPU_LOADED:
                _CPU_MODEL.to("cpu")
                _GPU_LOADED = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            with prepend_sys_path(VENDOR_ROOT / "litevggt"):
                from vggt.models.vggt import VGGT

            checkpoint = torch.load(resolved_checkpoint, map_location="cpu")
            model = VGGT(enable_camera=True, enable_depth=True, enable_point=False, enable_track=False)
            model.load_state_dict(checkpoint, strict=False)
            model.to(torch.bfloat16)
            model.eval()

            _CPU_MODEL = model
            _CPU_MODEL_CHECKPOINT_PATH = resolved_checkpoint
            _GPU_LOADED = False
            loaded_from_disk = True

        moved_to_gpu = False
        if not _GPU_LOADED:
            _CPU_MODEL.to("cuda:0", non_blocking=True)
            _GPU_LOADED = True
            moved_to_gpu = True

        _LAST_MODEL_CACHE_METRICS = {
            "litevggt_cpu_model_cached": True,
            "litevggt_gpu_loaded_from_cpu": moved_to_gpu,
            "litevggt_model_loaded_from_disk": loaded_from_disk,
            "litevggt_gpu_idle_unload_seconds": IDLE_UNLOAD_SECONDS,
        }
        return _CPU_MODEL


def get_litevggt_model_cache_metrics() -> dict[str, Any]:
    with _MODEL_LOCK:
        return {
            **_LAST_MODEL_CACHE_METRICS,
            "litevggt_gpu_model_loaded": bool(_GPU_LOADED),
        }


def schedule_litevggt_gpu_idle_unload(delay_seconds: float = IDLE_UNLOAD_SECONDS) -> None:
    global _IDLE_UNLOAD_TIMER, _IDLE_UNLOAD_TOKEN

    with _MODEL_LOCK:
        cancel_litevggt_idle_unload()
        if _CPU_MODEL is None or not _GPU_LOADED:
            return
        _IDLE_UNLOAD_TOKEN += 1
        token = _IDLE_UNLOAD_TOKEN
        timer = threading.Timer(float(delay_seconds), unload_litevggt_model_from_gpu_if_idle, args=(token,))
        timer.daemon = True
        _IDLE_UNLOAD_TIMER = timer
        timer.start()


def cancel_litevggt_idle_unload() -> None:
    global _IDLE_UNLOAD_TIMER, _IDLE_UNLOAD_TOKEN

    timer = _IDLE_UNLOAD_TIMER
    if timer is not None:
        timer.cancel()
        _IDLE_UNLOAD_TIMER = None
        _IDLE_UNLOAD_TOKEN += 1


def unload_litevggt_model_from_gpu_if_idle(token: int) -> None:
    with _MODEL_LOCK:
        if token != _IDLE_UNLOAD_TOKEN:
            return
        unload_litevggt_model_from_gpu()


def unload_litevggt_model_from_gpu() -> None:
    import torch

    global _GPU_LOADED, _IDLE_UNLOAD_TIMER, _IDLE_UNLOAD_TOKEN

    with _MODEL_LOCK:
        timer = _IDLE_UNLOAD_TIMER
        _IDLE_UNLOAD_TIMER = None
        _IDLE_UNLOAD_TOKEN += 1
        if timer is not None:
            timer.cancel()
        if _CPU_MODEL is None or not _GPU_LOADED:
            return
        _CPU_MODEL.to("cpu")
        _GPU_LOADED = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def reset_litevggt_model_cache_for_tests() -> None:
    global _CPU_MODEL, _CPU_MODEL_CHECKPOINT_PATH, _GPU_LOADED, _IDLE_UNLOAD_TOKEN, _LAST_MODEL_CACHE_METRICS

    with _MODEL_LOCK:
        cancel_litevggt_idle_unload()
        if _CPU_MODEL is not None and _GPU_LOADED:
            unload_litevggt_model_from_gpu()
        _CPU_MODEL = None
        _CPU_MODEL_CHECKPOINT_PATH = None
        _GPU_LOADED = False
        _IDLE_UNLOAD_TOKEN += 1
        _LAST_MODEL_CACHE_METRICS = {
            "litevggt_cpu_model_cached": False,
            "litevggt_gpu_loaded_from_cpu": False,
            "litevggt_model_loaded_from_disk": False,
            "litevggt_gpu_idle_unload_seconds": IDLE_UNLOAD_SECONDS,
        }


def resolve_litevggt_quality_settings(frame_count: int, options: dict[str, Any] | None = None) -> LiteVGGTQualitySettings:
    options = options or {}
    _ = int(frame_count)

    profile = "official"
    target_size = 518
    keep_ratio = 0.42

    target_size_source = "auto"
    if options.get("target_size") is not None:
        target_size = _align_litevggt_target_size(options["target_size"])
        target_size_source = "user"

    keep_ratio_source = "auto"
    if options.get("keep_ratio") is not None:
        keep_ratio = float(np.clip(float(options["keep_ratio"]), 0.01, 1.0))
        keep_ratio_source = "user"

    return LiteVGGTQualitySettings(
        target_size=_align_litevggt_target_size(target_size),
        keep_ratio=keep_ratio,
        quality_profile=profile,
        keep_ratio_source=keep_ratio_source,
        target_size_source=target_size_source,
    )


def _align_litevggt_target_size(value: Any) -> int:
    target_size = int(round(float(value) / 14.0) * 14)
    return max(14, target_size)


def select_aligned_frames(
    files: list[Path],
    *,
    multiple: int = 8,
    max_frames: int | None = None,
    frame_stride: int | None = None,
) -> list[Path]:
    return resolve_litevggt_frame_selection(
        files,
        multiple=multiple,
        max_frames=max_frames,
        frame_stride=frame_stride,
    ).files


def resolve_litevggt_frame_selection(
    files: list[Path],
    *,
    multiple: int = 8,
    max_frames: int | None = None,
    frame_stride: int | None = None,
) -> LiteVGGTFrameSelection:
    if multiple <= 0:
        raise PreviewFailure("LITEVGGT_INVALID_FRAME_MULTIPLE", "LiteVGGT frame multiple must be positive")

    total = len(files)
    if total <= 0:
        return LiteVGGTFrameSelection([], np.empty((0,), dtype=np.int32), 1, "auto", 0)

    requested_stride = int(frame_stride or 0)
    if requested_stride > 0:
        candidate_indices = list(range(0, total, requested_stride))
        if candidate_indices[-1] != total - 1:
            candidate_indices.append(total - 1)
        stride_source = "user"
        effective_budget = len(candidate_indices)
    else:
        candidate_indices = list(range(total))
        stride_source = "auto"
        effective_budget = int(max_frames or total)

    if max_frames is not None and max_frames > 0:
        effective_budget = min(effective_budget, int(max_frames))
    effective_budget = max(0, min(total, effective_budget))
    usable = (effective_budget // multiple) * multiple
    if usable <= 0:
        return LiteVGGTFrameSelection([], np.empty((0,), dtype=np.int32), max(1, requested_stride), stride_source, effective_budget)

    if requested_stride <= 0 and (max_frames is None or max_frames >= total):
        selected_indices = candidate_indices[:usable]
    else:
        selected_indices = _evenly_select_indices(candidate_indices, usable)
    selected_files = [files[index] for index in selected_indices]
    if requested_stride > 0:
        effective_stride = requested_stride
    elif len(selected_indices) > 1:
        effective_stride = max(1, int(round((selected_indices[-1] - selected_indices[0]) / (len(selected_indices) - 1))))
    else:
        effective_stride = 1

    return LiteVGGTFrameSelection(
        selected_files,
        np.asarray(selected_indices, dtype=np.int32),
        effective_stride,
        stride_source,
        effective_budget,
    )


def _evenly_select_indices(indices: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= len(indices):
        return list(indices)
    if count == 1:
        return [indices[0]]
    positions = np.linspace(0, len(indices) - 1, num=count)
    return [indices[int(np.floor(position))] for position in positions]


def point_indices_to_frame_indices(
    selected_pixel_indices: np.ndarray,
    *,
    frame_indices: list[int] | np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    pixels_per_frame = int(height) * int(width)
    if pixels_per_frame <= 0:
        raise PreviewFailure("LITEVGGT_INVALID_IMAGE_SHAPE", "LiteVGGT point-frame mapping requires positive image dimensions")

    local_frame_ids = np.asarray(selected_pixel_indices, dtype=np.int64) // pixels_per_frame
    original = np.asarray(frame_indices, dtype=np.int32)
    if np.any(local_frame_ids < 0) or np.any(local_frame_ids >= original.shape[0]):
        raise PreviewFailure("LITEVGGT_POINT_FRAME_MAPPING_FAILED", "LiteVGGT selected point indices exceed batch frame count")
    return original[local_frame_ids]


def select_points_by_confidence(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    *,
    frame_indices: list[int] | np.ndarray,
    height: int,
    width: int,
    keep_ratio: float,
    max_points: int,
    depth_conf_thresh: float | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts, rgb, conf, valid_indices, keep_count = prepare_litevggt_point_selection(
        points,
        colors,
        confidence,
        keep_ratio=keep_ratio,
        max_points=max_points,
        depth_conf_thresh=depth_conf_thresh,
        valid_mask=valid_mask,
    )

    ranked = valid_indices[np.argsort(conf[valid_indices])[::-1]]
    selected = ranked[:keep_count]
    point_frame_indices = point_indices_to_frame_indices(
        selected,
        frame_indices=frame_indices,
        height=height,
        width=width,
    )
    return pts[selected], rgb[selected], conf[selected], point_frame_indices


def select_points_by_scene_coverage(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    *,
    frame_indices: list[int] | np.ndarray,
    height: int,
    width: int,
    keep_ratio: float,
    max_points: int,
    depth_conf_thresh: float | None = None,
    valid_mask: np.ndarray | None = None,
    grid_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts, rgb, conf, valid_indices, keep_count = prepare_litevggt_point_selection(
        points,
        colors,
        confidence,
        keep_ratio=keep_ratio,
        max_points=max_points,
        depth_conf_thresh=depth_conf_thresh,
        valid_mask=valid_mask,
    )
    pixels_per_frame = int(height) * int(width)
    original_frames = np.asarray(frame_indices, dtype=np.int32)
    if pixels_per_frame <= 0 or original_frames.size <= 0:
        raise PreviewFailure("LITEVGGT_INVALID_IMAGE_SHAPE", "LiteVGGT scene coverage selection requires positive frame dimensions")

    local_frame_ids = valid_indices // pixels_per_frame
    pixel_offsets = valid_indices % pixels_per_frame
    rows = pixel_offsets // int(width)
    cols = pixel_offsets % int(width)
    grid_rows = max(1, min(int(grid_size), int(height)))
    grid_cols = max(1, min(int(grid_size), int(width)))
    tile_rows = np.minimum((rows * grid_rows) // int(height), grid_rows - 1)
    tile_cols = np.minimum((cols * grid_cols) // int(width), grid_cols - 1)
    bucket_ids = local_frame_ids * grid_rows * grid_cols + tile_rows * grid_cols + tile_cols
    buckets = np.unique(bucket_ids)
    if buckets.size == 0:
        raise PreviewFailure("LITEVGGT_EMPTY_POINT_CLOUD", "LiteVGGT produced no valid points")

    selected_parts: list[np.ndarray] = []
    if buckets.size > keep_count:
        bucket_positions = np.linspace(0, buckets.size - 1, num=keep_count)
        buckets_to_use = buckets[np.asarray(np.floor(bucket_positions), dtype=np.int64)]
        per_bucket = 1
    else:
        buckets_to_use = buckets
        per_bucket = max(1, keep_count // int(buckets_to_use.size))

    for bucket in buckets_to_use:
        members = valid_indices[bucket_ids == bucket]
        if members.size == 0:
            continue
        take = min(per_bucket, members.size)
        order = np.argsort(conf[members])[::-1]
        selected_parts.append(members[order[:take]])

    selected = np.unique(np.concatenate(selected_parts)) if selected_parts else np.empty((0,), dtype=np.int64)
    if selected.size < keep_count:
        ranked = valid_indices[np.argsort(conf[valid_indices])[::-1]]
        needed = keep_count - selected.size
        filler = ranked[~np.isin(ranked, selected, assume_unique=False)][:needed]
        selected = np.concatenate([selected, filler])
    elif selected.size > keep_count:
        ranked_selected = selected[np.argsort(conf[selected])[::-1]]
        selected = ranked_selected[:keep_count]

    point_frame_indices = point_indices_to_frame_indices(
        selected,
        frame_indices=frame_indices,
        height=height,
        width=width,
    )
    return pts[selected], rgb[selected], conf[selected], point_frame_indices


def prepare_litevggt_point_selection(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    *,
    keep_ratio: float,
    max_points: int,
    depth_conf_thresh: float | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = np.clip(np.asarray(colors).reshape(-1, 3), 0, 255).astype(np.uint8)
    conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if pts.shape[0] != rgb.shape[0] or pts.shape[0] != conf.shape[0]:
        raise PreviewFailure("LITEVGGT_POINT_SHAPE_MISMATCH", "LiteVGGT points, colors and confidence must have equal length")

    valid = np.isfinite(pts).all(axis=1) & np.isfinite(conf)
    if depth_conf_thresh is not None:
        valid &= conf >= float(depth_conf_thresh)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != pts.shape[0]:
            raise PreviewFailure("LITEVGGT_MASK_SHAPE_MISMATCH", "LiteVGGT valid mask must match point count")
        valid &= mask
    valid_indices = np.where(valid)[0]
    if valid_indices.size == 0:
        raise PreviewFailure("LITEVGGT_EMPTY_POINT_CLOUD", "LiteVGGT produced no valid points")

    keep_ratio = float(np.clip(keep_ratio, 0.01, 1.0))
    keep_count = max(1, int(valid_indices.size * keep_ratio))
    if max_points > 0:
        keep_count = min(keep_count, int(max_points))
    return pts, rgb, conf, valid_indices, keep_count


def litevggt_outlier_mask(
    points: np.ndarray,
    base_mask: np.ndarray,
    *,
    spatial_keep_quantile: float,
    axis_trim_low_quantile: float,
    axis_trim_high_quantile: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    mask = np.asarray(base_mask, dtype=bool).reshape(-1).copy()
    if mask.shape[0] != pts.shape[0]:
        raise PreviewFailure("LITEVGGT_MASK_SHAPE_MISMATCH", "LiteVGGT outlier mask must match point count")
    mask &= np.isfinite(pts).all(axis=1)
    if not np.any(mask):
        return mask

    low = float(np.clip(axis_trim_low_quantile, 0.0, 0.49))
    high = float(np.clip(axis_trim_high_quantile, low + 1e-6, 1.0))
    if low > 0.0 or high < 1.0:
        valid_pts = pts[mask]
        lower = np.quantile(valid_pts, low, axis=0)
        upper = np.quantile(valid_pts, high, axis=0)
        mask &= np.all((pts >= lower) & (pts <= upper), axis=1)

    spatial = float(np.clip(spatial_keep_quantile, 0.0, 1.0))
    if 0.0 < spatial < 1.0 and np.any(mask):
        valid_pts = pts[mask]
        center = np.median(valid_pts, axis=0)
        distances = np.linalg.norm(valid_pts - center, axis=1)
        threshold = float(np.quantile(distances, spatial))
        all_distances = np.linalg.norm(pts - center, axis=1)
        mask &= all_distances <= threshold
    return mask


def reset_litevggt_aggregator_cache(model) -> None:
    aggregator = getattr(model, "aggregator", None)
    if aggregator is not None and hasattr(aggregator, "m_u"):
        aggregator.m_u = None


def load_litevggt_image_tensors(files: list[Path], load_image_file_crop, target_size: int) -> list:
    import torch

    return [
        torch.from_numpy(np.transpose(load_image_file_crop(str(file), target_size=target_size), (2, 0, 1))).float()
        for file in files
    ]


def load_litevggt_image_batch(
    files: list[Path],
    load_image_file_crop,
    target_size: int,
    *,
    preprocess_mode: str = "pad",
) -> LiteVGGTImageBatch:
    import torch

    mode = str(preprocess_mode or "pad").strip().lower()
    if mode == "pad":
        tensors = []
        masks = []
        for file in files:
            image, mask = load_litevggt_padded_image(file, target_size)
            tensors.append(torch.from_numpy(np.transpose(image, (2, 0, 1))).float())
            masks.append(mask)
        return LiteVGGTImageBatch(tensors=tensors, valid_masks=np.stack(masks, axis=0), preprocess_mode="pad")

    tensors = load_litevggt_image_tensors(files, load_image_file_crop, target_size)
    if not tensors:
        return LiteVGGTImageBatch(tensors=[], valid_masks=np.empty((0, 0, 0), dtype=bool), preprocess_mode="crop")
    masks = [np.ones((int(tensor.shape[-2]), int(tensor.shape[-1])), dtype=bool) for tensor in tensors]
    return LiteVGGTImageBatch(tensors=tensors, valid_masks=np.stack(masks, axis=0), preprocess_mode="crop")


def load_litevggt_padded_image(path: Path, target_size: int) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image, ImageOps

    with Image.open(path) as original:
        original.load()
        image = ImageOps.exif_transpose(original).convert("RGB")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise PreviewFailure("LITEVGGT_INVALID_IMAGE_SHAPE", f"invalid image dimensions: {path}")

    target_size = _align_litevggt_target_size(target_size)
    if width >= height:
        resized_width = target_size
        resized_height = _align_litevggt_target_size(height * (target_size / float(width)))
    else:
        resized_height = target_size
        resized_width = _align_litevggt_target_size(width * (target_size / float(height)))
    resized_width = max(1, min(target_size, resized_width))
    resized_height = max(1, min(target_size, resized_height))
    image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)

    canvas = np.ones((target_size, target_size, 3), dtype=np.float32)
    mask = np.zeros((target_size, target_size), dtype=bool)
    left = (target_size - resized_width) // 2
    top = (target_size - resized_height) // 2
    array = np.asarray(image, dtype=np.float32) / 255.0
    canvas[top : top + resized_height, left : left + resized_width] = array
    mask[top : top + resized_height, left : left + resized_width] = True
    return canvas, mask


def run_litevggt_reconstruction(
    *,
    input_dir: Path,
    checkpoint_path: Path,
    keep_ratio: float | None,
    max_points: int,
    progress: Progress,
    max_input_frames: int | None = None,
    target_size: int | None = None,
    frame_stride: int | None = None,
    depth_conf_thresh: float | None = None,
    preprocess_mode: str = "pad",
    spatial_keep_quantile: float = 1.0,
    preserve_full_image: bool = False,
    letterbox_size: int = 518,
    frame_selection: str = "all",
    min_scene_change: float = 0.0,
    edge_keep_ratio: float = 0.0,
    axis_trim_low_quantile: float = 0.0,
    axis_trim_high_quantile: float = 1.0,
    selection_strategy: str = "global_confidence",
    **_unused_options,
) -> LiteVGGTReconstruction:
    import torch

    require_transformer_engine()
    with prepend_sys_path(VENDOR_ROOT / "litevggt"):
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_image_file_crop
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        original_files = image_files(input_dir)
        original_frame_count = len(original_files)
        if original_frame_count < 8:
            raise PreviewFailure("LITEVGGT_NOT_ENOUGH_IMAGES", "LiteVGGT preview requires at least 8 images")

        frame_selection_result = resolve_litevggt_frame_selection(
            original_files,
            multiple=8,
            max_frames=max_input_frames,
            frame_stride=frame_stride,
        )
        files = frame_selection_result.files
        if len(files) < 8:
            raise PreviewFailure("LITEVGGT_NOT_ENOUGH_FRAMES", f"LiteVGGT requires at least 8 images, got {len(files)}")

        quality = resolve_litevggt_quality_settings(
            len(files),
            {
                "target_size": target_size,
                "keep_ratio": keep_ratio,
            },
        )
        frame_indices = frame_selection_result.frame_indices.tolist()

        if not torch.cuda.is_available():
            raise PreviewFailure("GPU_RESOURCE_UNAVAILABLE", "LiteVGGT requires CUDA")
        device = "cuda:0"
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        progress(
            "litevggt_preprocess",
            28,
            f"loading {len(files)} LiteVGGT images at target_size={quality.target_size}, stride={frame_selection_result.frame_stride}, mode={preprocess_mode}",
        )
        loaded_images = load_litevggt_image_batch(
            files,
            load_image_file_crop,
            quality.target_size,
            preprocess_mode=preprocess_mode,
        )
        image_batch = torch.stack(loaded_images.tensors, dim=0).to(device)
        height = int(image_batch.shape[-2])
        width = int(image_batch.shape[-1])

        progress("litevggt_loading_model", 34, f"loading LiteVGGT model on GPU: {checkpoint_path.name}")
        model = get_litevggt_model_on_gpu(checkpoint_path)
        model_cache_metrics = get_litevggt_model_cache_metrics()

        aggregated_tokens_list = None
        pose_enc = None
        w2c_pre = None
        intrinsic = None
        depth_map = None
        depth_conf = None
        points_3d = None

        try:
            patch_width = width // 14
            patch_height = height // 14
            model.update_patch_dimensions(patch_width, patch_height)
            image_batch = image_batch[None]

            progress("litevggt_inference", 42, f"running LiteVGGT on {len(files)} aligned images")
            with torch.no_grad():
                fp8_recipe = DelayedScaling(fp8_format=Format.E4M3, amax_history_len=80, amax_compute_algo="max")
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    aggregated_tokens_list, patch_start_idx = model.aggregator(image_batch)

                with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
                    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                    w2c_pre, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_batch.shape[-2:])
                    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, image_batch, patch_start_idx)

                points_3d = unproject_depth_map_to_point_map(
                    depth_map.squeeze(0),
                    w2c_pre.squeeze(0),
                    intrinsic.squeeze(0),
                )

            image_array = image_batch[0].permute(0, 2, 3, 1).detach().float().cpu().numpy()
            valid_masks = loaded_images.valid_masks
            points = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
            colors = np.clip(image_array.reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8)
            confidence = depth_conf.reshape(-1).detach().float().cpu().numpy()
            valid_point_mask = litevggt_outlier_mask(
                points,
                valid_masks.reshape(-1),
                spatial_keep_quantile=spatial_keep_quantile,
                axis_trim_low_quantile=axis_trim_low_quantile,
                axis_trim_high_quantile=axis_trim_high_quantile,
            )

            point_selection_strategy = str(selection_strategy or "global_confidence").strip().lower()
            if point_selection_strategy in {"global", "global_confidence"}:
                selected_points, selected_colors, selected_confidence, point_frame_indices = select_points_by_confidence(
                    points,
                    colors,
                    confidence,
                    frame_indices=frame_indices,
                    height=height,
                    width=width,
                    keep_ratio=quality.keep_ratio,
                    max_points=max_points,
                    depth_conf_thresh=depth_conf_thresh,
                    valid_mask=valid_point_mask,
                )
                point_selection_metric = "global_confidence"
            else:
                selected_points, selected_colors, selected_confidence, point_frame_indices = select_points_by_scene_coverage(
                    points,
                    colors,
                    confidence,
                    frame_indices=frame_indices,
                    height=height,
                    width=width,
                    keep_ratio=quality.keep_ratio,
                    max_points=max_points,
                    depth_conf_thresh=depth_conf_thresh,
                    valid_mask=valid_point_mask,
                )
                point_selection_metric = "scene_coverage"

            w2c_array = w2c_pre.squeeze(0).detach().float().cpu().numpy()
            intrinsic_array = intrinsic.squeeze(0).detach().float().cpu().numpy()
            selected_count = int(selected_points.shape[0])
            peak_mb = int(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0
            frame_selection_metric = (
                "official_prefix_aligned"
                if frame_selection_result.frame_stride_source == "auto"
                and (max_input_frames is None or max_input_frames >= original_frame_count)
                else "bounded_even_selection"
            )
            return LiteVGGTReconstruction(
                files=files,
                frame_indices=np.asarray(frame_indices, dtype=np.int32),
                images=image_array,
                valid_masks=valid_masks,
                w2c=w2c_array,
                intrinsics=intrinsic_array,
                points=selected_points,
                colors=selected_colors,
                confidence=selected_confidence,
                point_frame_indices=point_frame_indices,
                metrics={
                    "original_frame_count": int(original_frame_count),
                    "input_frame_count": int(len(files)),
                    "aligned_frame_count": int(len(files)),
                    "litevggt_pose_frame_count": int(w2c_array.shape[0]),
                    "skipped_frame_count": int(original_frame_count - len(files)),
                    "frame_selection": frame_selection_metric,
                    "litevggt_frame_stride": int(frame_selection_result.frame_stride),
                    "litevggt_frame_stride_source": frame_selection_result.frame_stride_source,
                    "litevggt_frame_budget": int(frame_selection_result.frame_budget),
                    "litevggt_first_frame_index": int(frame_indices[0]),
                    "litevggt_last_frame_index": int(frame_indices[-1]),
                    "point_selection_strategy": point_selection_metric,
                    "keep_ratio": float(quality.keep_ratio),
                    "keep_ratio_source": quality.keep_ratio_source,
                    "depth_conf_thresh": None if depth_conf_thresh is None else float(depth_conf_thresh),
                    "spatial_keep_quantile": float(spatial_keep_quantile),
                    "axis_trim_low_quantile": float(axis_trim_low_quantile),
                    "axis_trim_high_quantile": float(axis_trim_high_quantile),
                    "litevggt_preprocess_mode": loaded_images.preprocess_mode,
                    "max_points": int(max_points),
                    "litevggt_target_size": int(quality.target_size),
                    "litevggt_target_size_source": quality.target_size_source,
                    "litevggt_quality_profile": quality.quality_profile,
                    "litevggt_inference_mode": "single",
                    "litevggt_inference_mode_requested": "single",
                    "litevggt_inference_mode_effective": "single",
                    "valid_pixel_count": int((np.isfinite(points).all(axis=1) & valid_point_mask).sum()),
                    "valid_image_pixel_count": int(valid_masks.sum()),
                    "point_count_before_filter": int(points.shape[0]),
                    "point_count_after_filter": selected_count,
                    "point_count_before_downsample": selected_count,
                    "point_count_after_downsample": selected_count,
                    "point_count_after_voxel_downsample": selected_count,
                    "cuda_memory_peak_mb": float(peak_mb),
                    "official_single_path": True,
                    **model_cache_metrics,
                },
            )
        finally:
            aggregated_tokens_list = None
            pose_enc = None
            w2c_pre = None
            intrinsic = None
            depth_map = None
            depth_conf = None
            points_3d = None
            reset_litevggt_aggregator_cache(model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            schedule_litevggt_gpu_idle_unload()


LITEVGGT_POINT_SOURCE = "litevggt_depth_unprojected"


def litevggt_preview_bounds(points: Any) -> dict[str, Any]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = pts[np.isfinite(pts).all(axis=1)]
    if finite.shape[0] == 0:
        raise PreviewFailure(
            "LITEVGGT_EMPTY_POINT_CLOUD",
            "LiteVGGT produced no finite points for preview",
        )

    bbox_min = np.percentile(finite, 1, axis=0).astype(np.float32)
    bbox_max = np.percentile(finite, 99, axis=0).astype(np.float32)
    bbox_center = ((bbox_min + bbox_max) * 0.5).astype(np.float32)
    bbox_radius = max(float(np.linalg.norm(bbox_max - bbox_min) * 0.5), 0.05)
    return {
        "bbox_min": [float(value) for value in bbox_min],
        "bbox_max": [float(value) for value in bbox_max],
        "bbox_center": [float(value) for value in bbox_center],
        "bbox_radius": bbox_radius,
    }


def write_litevggt_preview_meta_json(
    output_meta_json: Path,
    *,
    point_count_raw: int,
    point_count_exported: int,
    bounds: dict[str, Any],
    recommended_view: dict[str, Any] | None = None,
) -> None:
    payload = {
        "asset_type": "litevggt_preview_points",
        "point_source": LITEVGGT_POINT_SOURCE,
        "num_points": int(point_count_exported),
        "point_count_raw": int(point_count_raw),
        "point_count_exported": int(point_count_exported),
        "bbox_min": bounds["bbox_min"],
        "bbox_max": bounds["bbox_max"],
        "bbox_center": bounds["bbox_center"],
        "bbox_radius": bounds["bbox_radius"],
        "center": bounds["bbox_center"],
        "radius": bounds["bbox_radius"],
        "scale_applied": 1.0,
        "coordinate_system": "litevggt_world",
        "recommended_frontend": {
            "default_view_mode": "points",
            "default_point_size": 2.0,
        },
    }
    if recommended_view is not None:
        payload["recommended_view"] = recommended_view
    output_meta_json.parent.mkdir(parents=True, exist_ok=True)
    output_meta_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def litevggt_camera_view(
    w2c: np.ndarray,
    intrinsic: np.ndarray,
    *,
    radius: float,
    image_height: int,
) -> dict[str, Any] | None:
    try:
        rotation = np.asarray(w2c[:3, :3], dtype=np.float32)
        translation = np.asarray(w2c[:3, 3], dtype=np.float32)
        camera_to_world = rotation.T
        position = (-camera_to_world @ translation).astype(np.float32)
        forward = normalize(camera_to_world @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        up = normalize(camera_to_world @ np.asarray([0.0, -1.0, 0.0], dtype=np.float32))
        if forward is None or up is None:
            return None
        target = position + forward * max(float(radius), 0.35)
        view: dict[str, Any] = {
            "source": "first_frame_camera",
            "position": [float(value) for value in position],
            "target": [float(value) for value in target],
            "up": [float(value) for value in up],
        }
        focal_y = float(np.asarray(intrinsic)[1, 1])
        if image_height > 0 and focal_y > 0:
            view["fov_y_degrees"] = float(np.degrees(2.0 * np.arctan(float(image_height) / (2.0 * focal_y))))
        return view
    except Exception:
        return None


def normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-6:
        return None
    return (vector / norm).astype(np.float32)


def identity_sim3() -> Sim3Transform:
    return Sim3Transform(
        scale=1.0,
        rotation=np.eye(3, dtype=np.float32),
        translation=np.zeros(3, dtype=np.float32),
    )


def apply_sim3(points: np.ndarray, transform: Sim3Transform) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return (float(transform.scale) * (pts @ np.asarray(transform.rotation, dtype=np.float32).T) + transform.translation).astype(
        np.float32,
        copy=False,
    )


def compose_sim3(outer: Sim3Transform, inner: Sim3Transform) -> Sim3Transform:
    outer_rotation = np.asarray(outer.rotation, dtype=np.float32)
    inner_rotation = np.asarray(inner.rotation, dtype=np.float32)
    outer_translation = np.asarray(outer.translation, dtype=np.float32)
    inner_translation = np.asarray(inner.translation, dtype=np.float32)
    return Sim3Transform(
        scale=float(outer.scale) * float(inner.scale),
        rotation=(outer_rotation @ inner_rotation).astype(np.float32),
        translation=(float(outer.scale) * (outer_rotation @ inner_translation) + outer_translation).astype(np.float32),
    )


def invert_sim3(transform: Sim3Transform) -> Sim3Transform:
    scale = max(float(transform.scale), 1e-8)
    rotation = np.asarray(transform.rotation, dtype=np.float32).T
    translation = (-(rotation @ np.asarray(transform.translation, dtype=np.float32)) / scale).astype(np.float32)
    return Sim3Transform(scale=1.0 / scale, rotation=rotation.astype(np.float32), translation=translation)


def estimate_sim3(source: np.ndarray, target: np.ndarray) -> Sim3Transform | None:
    src = np.asarray(source, dtype=np.float32).reshape(-1, 3)
    dst = np.asarray(target, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[valid]
    dst = dst[valid]
    if src.shape[0] == 0:
        return None
    if src.shape[0] < 3:
        return Sim3Transform(
            scale=1.0,
            rotation=np.eye(3, dtype=np.float32),
            translation=(dst.mean(axis=0) - src.mean(axis=0)).astype(np.float32),
        )

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    variance = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if not np.isfinite(variance) or variance <= 1e-8:
        return Sim3Transform(
            scale=1.0,
            rotation=np.eye(3, dtype=np.float32),
            translation=(dst_mean - src_mean).astype(np.float32),
        )

    covariance = (dst_centered.T @ src_centered) / float(src.shape[0])
    try:
        u, singular_values, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    correction = np.eye(3, dtype=np.float32)
    if float(np.linalg.det(u) * np.linalg.det(vt)) < 0.0:
        correction[-1, -1] = -1.0
    rotation = (u @ correction @ vt).astype(np.float32)
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    translation = (dst_mean - scale * (rotation @ src_mean)).astype(np.float32)
    return Sim3Transform(scale=scale, rotation=rotation, translation=translation)


def camera_centers_from_w2c(w2c: np.ndarray) -> np.ndarray:
    matrices = np.asarray(w2c, dtype=np.float32).reshape(-1, 4, 4)
    rotations = matrices[:, :3, :3]
    translations = matrices[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(rotations, (0, 2, 1)), translations).astype(np.float32)


def make_litevggt_window_specs(
    files: list[Path],
    frame_indices: np.ndarray,
    *,
    chunk_size: int,
    overlap: int,
) -> list[LiteVGGTWindowSpec]:
    total = len(files)
    if total <= 0:
        return []
    chunk = max(8, (int(chunk_size) // 8) * 8)
    if total <= chunk:
        return [
            LiteVGGTWindowSpec(
                index=0,
                start=0,
                end=total,
                files=list(files),
                frame_indices=np.asarray(frame_indices, dtype=np.int32),
            )
        ]
    overlap = max(0, min(int(overlap), chunk - 8))
    step = max(8, chunk - overlap)
    starts: list[int] = []
    start = 0
    while start + chunk < total:
        starts.append(start)
        start += step
    last_start = max(0, total - chunk)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    windows: list[LiteVGGTWindowSpec] = []
    indices = np.asarray(frame_indices, dtype=np.int32)
    for window_index, start in enumerate(starts):
        end = min(total, start + chunk)
        windows.append(
            LiteVGGTWindowSpec(
                index=window_index,
                start=start,
                end=end,
                files=list(files[start:end]),
                frame_indices=indices[start:end],
            )
        )
    return windows


def materialize_litevggt_window_dir(root: Path, spec: LiteVGGTWindowSpec) -> Path:
    window_dir = root / f"window_{spec.index:04d}"
    if window_dir.exists():
        shutil.rmtree(window_dir)
    window_dir.mkdir(parents=True, exist_ok=True)
    for local_index, source in enumerate(spec.files):
        suffix = source.suffix.lower() or ".jpg"
        destination = window_dir / f"{local_index:06d}{suffix}"
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    return window_dir


def map_reconstruction_frame_indices(reconstruction: LiteVGGTReconstruction, spec: LiteVGGTWindowSpec) -> tuple[np.ndarray, np.ndarray]:
    local_indices = np.asarray(reconstruction.frame_indices, dtype=np.int64)
    original_frame_indices = spec.frame_indices[local_indices]
    point_local_indices = np.asarray(reconstruction.point_frame_indices, dtype=np.int64)
    point_frame_indices = spec.frame_indices[point_local_indices]
    return original_frame_indices.astype(np.int32), point_frame_indices.astype(np.int32)


def sample_values(values: np.ndarray, count: int) -> np.ndarray:
    unique = np.unique(np.asarray(values, dtype=np.int32))
    if unique.size <= count:
        return unique
    positions = np.linspace(0, unique.size - 1, num=count)
    return unique[np.asarray(np.floor(positions), dtype=np.int64)]


def select_litevggt_keyframes(
    files: list[Path],
    frame_indices: np.ndarray,
    *,
    target: int | None,
    min_frame_gap: int,
    min_scene_change: float,
    min_sharpness_percentile: float = 20.0,
) -> tuple[list[Path], np.ndarray, dict[str, Any]]:
    if target is None or target <= 0 or len(files) <= int(target):
        return list(files), np.asarray(frame_indices, dtype=np.int32), {
            "frame_selection_strategy": "all_frames",
            "selected_keyframe_count": int(len(files)),
        }

    thumbs: list[np.ndarray | None] = []
    sharpness: list[float] = []
    for path in files:
        thumb = load_litevggt_gray_thumb(path)
        thumbs.append(thumb)
        sharpness.append(image_sharpness_gray(thumb) if thumb is not None else 0.0)

    sharp_array = np.asarray(sharpness, dtype=np.float32)
    threshold = float(np.percentile(sharp_array[np.isfinite(sharp_array)], min_sharpness_percentile)) if sharp_array.size else 0.0
    selected = [0]
    last_thumb = thumbs[0]
    gap = max(1, int(min_frame_gap))
    diff_threshold = max(0.0, float(min_scene_change))

    for index in range(1, len(files) - 1):
        if index - selected[-1] < gap:
            continue
        thumb = thumbs[index]
        if thumb is None:
            continue
        if sharp_array[index] < threshold:
            continue
        diff = image_diff_gray(last_thumb, thumb) if last_thumb is not None else diff_threshold
        if diff < diff_threshold:
            continue
        selected.append(index)
        last_thumb = thumb

    selected.append(len(files) - 1)
    selected = sorted(set(selected))
    if len(selected) > int(target):
        positions = np.linspace(0, len(selected) - 1, num=int(target))
        selected = [selected[int(np.floor(position))] for position in positions]

    if len(selected) < min(8, len(files)):
        needed = min(8, len(files))
        uniform = np.linspace(0, len(files) - 1, num=needed)
        selected = sorted(set(selected + [int(np.floor(position)) for position in uniform]))

    selected_files = [files[index] for index in selected]
    selected_indices = np.asarray(frame_indices, dtype=np.int32)[selected]
    return selected_files, selected_indices, {
        "frame_selection_strategy": "keyframe_sharpness_difference_uniform",
        "selected_keyframe_count": int(len(selected_files)),
    }


def load_litevggt_gray_thumb(path: Path, size: int = 160) -> np.ndarray | None:
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        return cv2.resize(image, (size, size))
    except Exception:
        from PIL import Image, ImageOps

        try:
            with Image.open(path) as image:
                return np.asarray(ImageOps.grayscale(image).resize((size, size)), dtype=np.uint8)
        except Exception:
            return None


def image_sharpness_gray(image: np.ndarray | None) -> float:
    if image is None:
        return 0.0
    try:
        import cv2

        return float(cv2.Laplacian(image, cv2.CV_64F).var())
    except Exception:
        arr = image.astype(np.float32)
        gy, gx = np.gradient(arr)
        return float(np.var(gx) + np.var(gy))


def image_diff_gray(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))


def sample_alignment_points(
    points: np.ndarray,
    frame_indices: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    frames: set[int] | None = None,
    max_points: int = 40000,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    ids = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    valid = np.isfinite(pts).all(axis=1)
    if frames is not None:
        valid &= np.isin(ids, np.asarray(sorted(frames), dtype=np.int32))
    indices = np.where(valid)[0]
    if indices.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)

    limit = max(1, int(max_points))
    if indices.size > limit:
        conf = None if confidence is None else np.asarray(confidence, dtype=np.float32).reshape(-1)
        if conf is not None and conf.shape[0] == pts.shape[0] and np.isfinite(conf[indices]).any():
            order = np.argsort(conf[indices])[::-1][:limit]
            indices = indices[order]
        else:
            positions = np.linspace(0, indices.size - 1, num=limit)
            indices = indices[np.asarray(np.floor(positions), dtype=np.int64)]
    return pts[indices], ids[indices]


def collect_alignment_target(
    states: list[LiteVGGTWindowState],
    frames: set[int],
    *,
    max_points: int = 60000,
) -> tuple[np.ndarray, np.ndarray]:
    point_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    for state in reversed(states):
        pts, ids = sample_alignment_points(
            apply_sim3(state.alignment_points, state.transform),
            state.alignment_frame_indices,
            frames=frames,
            max_points=max_points,
        )
        if pts.size == 0:
            continue
        point_parts.append(pts)
        frame_parts.append(ids)
        if sum(part.shape[0] for part in point_parts) >= max_points:
            break
    if not point_parts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)
    points = np.concatenate(point_parts, axis=0)
    ids = np.concatenate(frame_parts, axis=0)
    return sample_alignment_points(points, ids, max_points=max_points)


def estimate_translation_from_centers(
    local_centers_by_frame: dict[int, np.ndarray],
    global_centers_by_frame: dict[int, np.ndarray],
    frames: set[int],
) -> Sim3Transform | None:
    deltas = []
    for frame in frames:
        if frame in local_centers_by_frame and frame in global_centers_by_frame:
            local = np.asarray(local_centers_by_frame[frame], dtype=np.float32)
            global_center = np.asarray(global_centers_by_frame[frame], dtype=np.float32)
            if np.isfinite(local).all() and np.isfinite(global_center).all():
                deltas.append(global_center - local)
    if not deltas:
        return None
    return Sim3Transform(
        scale=1.0,
        rotation=np.eye(3, dtype=np.float32),
        translation=np.median(np.stack(deltas, axis=0), axis=0).astype(np.float32),
    )


def estimate_translation_from_points(source: np.ndarray, target: np.ndarray) -> Sim3Transform | None:
    if source.shape[0] == 0 or target.shape[0] == 0:
        return None
    src_center = np.median(source, axis=0)
    dst_center = np.median(target, axis=0)
    if not np.isfinite(src_center).all() or not np.isfinite(dst_center).all():
        return None
    return Sim3Transform(
        scale=1.0,
        rotation=np.eye(3, dtype=np.float32),
        translation=(dst_center - src_center).astype(np.float32),
    )


def nearest_neighbor_distances(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(source, dtype=np.float32).reshape(-1, 3)
    dst = np.asarray(target, dtype=np.float32).reshape(-1, 3)
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(dst).query(src, k=1, workers=-1)
        return distances.astype(np.float32), indices.astype(np.int64)
    except Exception:
        if dst.shape[0] > 8000:
            positions = np.linspace(0, dst.shape[0] - 1, num=8000)
            dst = dst[np.asarray(np.floor(positions), dtype=np.int64)]
        distances = np.empty((src.shape[0],), dtype=np.float32)
        indices = np.empty((src.shape[0],), dtype=np.int64)
        for start in range(0, src.shape[0], 1024):
            chunk = src[start : start + 1024]
            diff = chunk[:, None, :] - dst[None, :, :]
            squared = np.sum(diff * diff, axis=2)
            local = np.argmin(squared, axis=1)
            distances[start : start + chunk.shape[0]] = np.sqrt(squared[np.arange(chunk.shape[0]), local])
            indices[start : start + chunk.shape[0]] = local
        return distances, indices


def trimmed_icp_sim3(
    source: np.ndarray,
    target: np.ndarray,
    *,
    initial: Sim3Transform | None,
    max_iterations: int = 10,
    max_points: int = 12000,
    trim_quantile: float = 0.7,
    min_inlier_ratio: float = 0.12,
    scale_min: float = 0.75,
    scale_max: float = 1.35,
) -> LiteVGGTIcpResult:
    src, _ = sample_alignment_points(source, np.zeros((source.shape[0],), dtype=np.int32), max_points=max_points)
    dst, _ = sample_alignment_points(target, np.zeros((target.shape[0],), dtype=np.int32), max_points=max_points * 2)
    if src.shape[0] < 50 or dst.shape[0] < 50:
        return LiteVGGTIcpResult(identity_sim3(), float("inf"), 0.0, False, "not_enough_points")

    current = initial or estimate_translation_from_points(src, dst) or identity_sim3()
    residual = float("inf")
    inlier_ratio = 0.0
    for _ in range(max(1, int(max_iterations))):
        transformed = apply_sim3(src, current)
        distances, indices = nearest_neighbor_distances(transformed, dst)
        finite = np.isfinite(distances)
        if not np.any(finite):
            return LiteVGGTIcpResult(current, float("inf"), 0.0, False, "no_finite_matches")
        threshold = float(np.quantile(distances[finite], np.clip(trim_quantile, 0.1, 0.95)))
        inliers = finite & (distances <= threshold)
        inlier_count = int(inliers.sum())
        inlier_ratio = float(inlier_count / max(1, distances.shape[0]))
        if inlier_count < 30:
            return LiteVGGTIcpResult(current, float("inf"), inlier_ratio, False, "too_few_inliers")
        candidate = estimate_sim3(src[inliers], dst[indices[inliers]])
        if candidate is None:
            return LiteVGGTIcpResult(current, float("inf"), inlier_ratio, False, "sim3_failed")
        current = candidate
        residual = float(np.median(distances[inliers]))

    if not np.isfinite(residual):
        return LiteVGGTIcpResult(current, residual, inlier_ratio, False, "invalid_residual")
    if current.scale < scale_min or current.scale > scale_max:
        return LiteVGGTIcpResult(current, residual, inlier_ratio, False, "scale_out_of_range")
    target_diag = pointcloud_diag(dst)
    max_residual = max(0.025, target_diag * 0.035)
    if residual > max_residual:
        return LiteVGGTIcpResult(current, residual, inlier_ratio, False, "residual_too_high")
    if inlier_ratio < min_inlier_ratio:
        return LiteVGGTIcpResult(current, residual, inlier_ratio, False, "inlier_ratio_too_low")
    return LiteVGGTIcpResult(current, residual, inlier_ratio, True, "accepted")


def pointcloud_diag(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = pts[np.isfinite(pts).all(axis=1)]
    if finite.shape[0] == 0:
        return 0.0
    low = np.percentile(finite, 5, axis=0)
    high = np.percentile(finite, 95, axis=0)
    diag = float(np.linalg.norm(high - low))
    return diag if np.isfinite(diag) else 0.0


def align_litevggt_window(
    states: list[LiteVGGTWindowState],
    *,
    source_points: np.ndarray,
    source_frame_indices: np.ndarray,
    local_centers_by_frame: dict[int, np.ndarray],
    global_centers_by_frame: dict[int, np.ndarray],
) -> tuple[Sim3Transform, dict[str, Any]]:
    if not states:
        return identity_sim3(), {
            "strategy": "identity",
            "residual": 0.0,
            "inlier_ratio": 1.0,
            "fallback": False,
            "rejected": False,
        }

    previous_frames = set()
    for state in states:
        previous_frames.update(int(frame) for frame in np.unique(state.alignment_frame_indices))
    current_frames = set(int(frame) for frame in np.unique(source_frame_indices))
    common_frames = previous_frames & current_frames
    if not common_frames:
        return states[-1].transform, {
            "strategy": "previous_transform_no_overlap",
            "residual": None,
            "inlier_ratio": 0.0,
            "fallback": True,
            "rejected": True,
        }

    source_common, _ = sample_alignment_points(source_points, source_frame_indices, frames=common_frames, max_points=40000)
    target_common, _ = collect_alignment_target(states, common_frames, max_points=60000)
    center_initial = estimate_translation_from_centers(local_centers_by_frame, global_centers_by_frame, common_frames)
    initial = center_initial or estimate_translation_from_points(source_common, target_common)
    if source_common.shape[0] >= 50 and target_common.shape[0] >= 50:
        result = trimmed_icp_sim3(source_common, target_common, initial=initial)
        if result.accepted:
            return result.transform, {
                "strategy": "overlap_icp_sim3",
                "residual": result.residual,
                "inlier_ratio": result.inlier_ratio,
                "fallback": False,
                "rejected": False,
            }
        if initial is not None:
            return initial, {
                "strategy": f"translation_fallback_after_{result.reason}",
                "residual": result.residual if np.isfinite(result.residual) else None,
                "inlier_ratio": result.inlier_ratio,
                "fallback": True,
                "rejected": True,
            }

    if initial is not None:
        return initial, {
            "strategy": "translation_fallback_not_enough_overlap_points",
            "residual": None,
            "inlier_ratio": 0.0,
            "fallback": True,
            "rejected": True,
        }
    return states[-1].transform, {
        "strategy": "previous_transform_no_alignment",
        "residual": None,
        "inlier_ratio": 0.0,
        "fallback": True,
        "rejected": True,
    }


def voxel_downsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    *,
    diag_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    keep = voxel_downsample_keep_indices(points, confidence, diag_ratio=diag_ratio)
    return pts[keep], np.asarray(colors)[keep], np.asarray(confidence, dtype=np.float32)[keep]


def voxel_downsample_keep_indices(points: np.ndarray, confidence: np.ndarray, *, diag_ratio: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    ratio = float(diag_ratio or 0.0)
    if ratio <= 0.0 or pts.shape[0] <= 1:
        return np.arange(pts.shape[0], dtype=np.int64)
    diag = pointcloud_diag(pts)
    voxel = diag * ratio
    if not np.isfinite(voxel) or voxel <= 0.0:
        return np.arange(pts.shape[0], dtype=np.int64)
    finite = np.isfinite(pts).all(axis=1)
    if not np.any(finite):
        return np.arange(pts.shape[0], dtype=np.int64)
    origin = np.nanmin(pts[finite], axis=0)
    keys = np.floor((pts - origin) / voxel).astype(np.int64)
    hashed = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
    conf = np.asarray(confidence, dtype=np.float32)
    order = np.lexsort((-conf, hashed))
    unique_hashes, first = np.unique(hashed[order], return_index=True)
    del unique_hashes
    keep = order[first]
    keep.sort()
    return keep.astype(np.int64)


def litevggt_image_similarity(path_a: Path, path_b: Path) -> float:
    try:
        import cv2

        image_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
        image_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
        if image_a is None or image_b is None:
            return 0.0
        image_a = cv2.resize(image_a, (160, 160))
        image_b = cv2.resize(image_b, (160, 160))
        orb = cv2.ORB_create(nfeatures=300)
        keypoints_a, desc_a = orb.detectAndCompute(image_a, None)
        keypoints_b, desc_b = orb.detectAndCompute(image_b, None)
        if desc_a is None or desc_b is None or not keypoints_a or not keypoints_b:
            diff = float(np.mean(np.abs(image_a.astype(np.float32) - image_b.astype(np.float32)))) / 255.0
            return float(np.clip(1.0 - diff, 0.0, 1.0))
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc_a, desc_b)
        good = sum(1 for match in matches if match.distance <= 48)
        return float(np.clip(good / 80.0, 0.0, 1.0))
    except Exception:
        from PIL import Image, ImageOps

        try:
            with Image.open(path_a) as a, Image.open(path_b) as b:
                arr_a = np.asarray(ImageOps.grayscale(a).resize((96, 96)), dtype=np.float32)
                arr_b = np.asarray(ImageOps.grayscale(b).resize((96, 96)), dtype=np.float32)
            diff = float(np.mean(np.abs(arr_a - arr_b))) / 255.0
            return float(np.clip(1.0 - diff, 0.0, 1.0))
        except Exception:
            return 0.0


def find_litevggt_loop_candidates(
    states: list[LiteVGGTWindowState],
    *,
    max_candidates: int = 3,
    min_score: float = 0.55,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[int, int, float]] = []
    for left in range(len(states)):
        for right in range(left + 2, len(states)):
            left_file = states[left].spec.files[len(states[left].spec.files) // 2]
            right_file = states[right].spec.files[len(states[right].spec.files) // 2]
            score = litevggt_image_similarity(left_file, right_file)
            if score >= min_score:
                candidates.append((left, right, score))
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[:max_candidates]


def validate_litevggt_loop_edge(
    *,
    candidate: tuple[int, int, float],
    states: list[LiteVGGTWindowState],
    original_files: list[Path],
    bridge_root: Path,
    checkpoint_path: Path,
    keep_ratio: float | None,
    target_size: int | None,
    depth_conf_thresh: float | None,
    preprocess_mode: str,
    selection_strategy: str,
    axis_trim_low_quantile: float,
    axis_trim_high_quantile: float,
    spatial_keep_quantile: float,
    progress: Progress,
) -> LiteVGGTLoopEdge | None:
    left_index, right_index, score = candidate
    left_state = states[left_index]
    right_state = states[right_index]
    left_frames = sample_values(left_state.spec.frame_indices, 4)
    right_frames = sample_values(right_state.spec.frame_indices, 4)
    if left_frames.size < 3 or right_frames.size < 3:
        return None
    bridge_indices = np.concatenate([left_frames, right_frames]).astype(np.int32)
    bridge_files = [original_files[int(index)] for index in bridge_indices]
    bridge_spec = LiteVGGTWindowSpec(
        index=left_index * 1000 + right_index,
        start=0,
        end=len(bridge_files),
        files=bridge_files,
        frame_indices=bridge_indices,
    )
    bridge_dir = materialize_litevggt_window_dir(bridge_root, bridge_spec)
    bridge = run_litevggt_reconstruction(
        input_dir=bridge_dir,
        checkpoint_path=checkpoint_path,
        keep_ratio=keep_ratio,
        max_points=1,
        max_input_frames=None,
        target_size=target_size,
        frame_stride=None,
        depth_conf_thresh=depth_conf_thresh,
        preprocess_mode=preprocess_mode,
        spatial_keep_quantile=spatial_keep_quantile,
        axis_trim_low_quantile=axis_trim_low_quantile,
        axis_trim_high_quantile=axis_trim_high_quantile,
        selection_strategy=selection_strategy,
        progress=progress,
    )
    bridge_frame_indices, _ = map_reconstruction_frame_indices(bridge, bridge_spec)
    bridge_centers = camera_centers_from_w2c(bridge.w2c)
    bridge_center_count = min(int(bridge_centers.shape[0]), int(bridge_frame_indices.shape[0]))
    bridge_centers_by_frame = {
        int(frame_index): bridge_centers[index] for index, frame_index in enumerate(bridge_frame_indices[:bridge_center_count])
    }

    left_common = [int(frame) for frame in left_frames if int(frame) in bridge_centers_by_frame and int(frame) in left_state.local_centers_by_frame]
    right_common = [int(frame) for frame in right_frames if int(frame) in bridge_centers_by_frame and int(frame) in right_state.local_centers_by_frame]
    if len(left_common) < 3 or len(right_common) < 3:
        return None

    left_to_bridge = estimate_sim3(
        np.stack([left_state.local_centers_by_frame[frame] for frame in left_common], axis=0),
        np.stack([bridge_centers_by_frame[frame] for frame in left_common], axis=0),
    )
    right_to_bridge = estimate_sim3(
        np.stack([right_state.local_centers_by_frame[frame] for frame in right_common], axis=0),
        np.stack([bridge_centers_by_frame[frame] for frame in right_common], axis=0),
    )
    if left_to_bridge is None or right_to_bridge is None:
        return None

    right_to_left = compose_sim3(invert_sim3(left_to_bridge), right_to_bridge)
    expected_right_global = compose_sim3(left_state.transform, right_to_left)
    delta = (expected_right_global.translation - left_state.transform.translation).astype(np.float32)
    if not np.isfinite(delta).all():
        return None
    return LiteVGGTLoopEdge(source=left_index, target=right_index, delta=delta, score=score)


def optimize_litevggt_window_translations(
    states: list[LiteVGGTWindowState],
    loop_edges: list[LiteVGGTLoopEdge],
) -> dict[str, Any]:
    if len(states) <= 1:
        return {"loop_optimization_residual": None, "loop_optimization_skipped_reason": "single_window"}
    if not loop_edges:
        return {"loop_optimization_residual": None, "loop_optimization_skipped_reason": "no_valid_loop_edges"}
    try:
        from scipy.optimize import least_squares
    except Exception:
        return {"loop_optimization_residual": None, "loop_optimization_skipped_reason": "scipy_unavailable"}

    initial = np.stack([state.transform.translation for state in states], axis=0).astype(np.float32)
    sequence_edges = [
        LiteVGGTLoopEdge(
            source=index,
            target=index + 1,
            delta=(initial[index + 1] - initial[index]).astype(np.float32),
            score=1.0,
        )
        for index in range(len(states) - 1)
    ]
    edges = sequence_edges + loop_edges
    weights = [1.0 for _ in sequence_edges] + [0.65 for _ in loop_edges]

    def unpack(values: np.ndarray) -> np.ndarray:
        translations = initial.copy()
        translations[1:] = values.reshape(len(states) - 1, 3)
        return translations

    def residual(values: np.ndarray) -> np.ndarray:
        translations = unpack(values)
        parts: list[np.ndarray] = []
        for edge, weight in zip(edges, weights, strict=False):
            parts.append((translations[edge.target] - translations[edge.source] - edge.delta) * weight)
        return np.concatenate(parts)

    result = least_squares(residual, initial[1:].reshape(-1), max_nfev=80)
    optimized = unpack(result.x)
    for index, state in enumerate(states):
        state.transform = Sim3Transform(
            scale=state.transform.scale,
            rotation=state.transform.rotation,
            translation=optimized[index].astype(np.float32),
        )
    residual_norm = float(np.linalg.norm(residual(result.x)) / max(1, len(edges)))
    return {"loop_optimization_residual": residual_norm, "loop_optimization_skipped_reason": None}


def run_litevggt_pointcloud_windowed(
    *,
    input_dir: Path,
    checkpoint_path: Path,
    output_ply: Path,
    output_meta_json: Path | None,
    keep_ratio: float | None,
    max_points: int,
    max_input_frames: int | None,
    target_size: int | None,
    frame_stride: int | None,
    depth_conf_thresh: float | None,
    preprocess_mode: str,
    progress: Progress,
    chunk_size: int,
    overlap: int,
    loop_closure: bool,
    scene_profile: str = "mixed_balanced",
    keyframe_target: int | None = None,
    min_frame_gap: int = 1,
    min_scene_change: float = 0.0,
    window_voxel_diag_ratio: float = 0.0,
    final_voxel_diag_ratio: float = 0.0,
    selection_strategy: str = "global_confidence",
    axis_trim_low_quantile: float = 0.0,
    axis_trim_high_quantile: float = 1.0,
    spatial_keep_quantile: float = 1.0,
    **_unused_options,
) -> dict[str, Any]:
    original_files = image_files(input_dir)
    original_frame_count = len(original_files)
    if original_frame_count < 8:
        raise PreviewFailure("LITEVGGT_NOT_ENOUGH_IMAGES", "LiteVGGT preview requires at least 8 images")

    if max_input_frames is not None or frame_stride is not None:
        frame_selection_result = resolve_litevggt_frame_selection(
            original_files,
            multiple=8,
            max_frames=max_input_frames,
            frame_stride=frame_stride,
        )
        files = frame_selection_result.files
        frame_indices = frame_selection_result.frame_indices
    else:
        files = list(original_files)
        frame_indices = np.arange(original_frame_count, dtype=np.int32)
        frame_selection_result = LiteVGGTFrameSelection(
            files=files,
            frame_indices=frame_indices,
            frame_stride=1,
            frame_stride_source="auto",
            frame_budget=original_frame_count,
        )
    keyframe_metrics: dict[str, Any] = {
        "frame_selection_strategy": "all_frames",
        "selected_keyframe_count": int(len(files)),
    }
    if max_input_frames is None and frame_stride is None and str(scene_profile).strip().lower() != "indoor_full":
        files, frame_indices, keyframe_metrics = select_litevggt_keyframes(
            files,
            frame_indices,
            target=keyframe_target,
            min_frame_gap=min_frame_gap,
            min_scene_change=min_scene_change,
        )
        frame_selection_result = LiteVGGTFrameSelection(
            files=files,
            frame_indices=frame_indices,
            frame_stride=frame_selection_result.frame_stride,
            frame_stride_source=frame_selection_result.frame_stride_source,
            frame_budget=len(files),
        )
    if len(files) < 8:
        raise PreviewFailure("LITEVGGT_NOT_ENOUGH_FRAMES", f"LiteVGGT requires at least 8 images, got {len(files)}")

    windows = make_litevggt_window_specs(files, frame_indices, chunk_size=chunk_size, overlap=overlap)
    if len(windows) <= 1:
        single_input_dir = input_dir
        if len(files) != len(original_files) or any(left != right for left, right in zip(files, original_files, strict=False)):
            single_root = output_ply.parent / "single_selected"
            if single_root.exists():
                shutil.rmtree(single_root)
            single_root.mkdir(parents=True, exist_ok=True)
            single_input_dir = materialize_litevggt_window_dir(single_root, windows[0])
        reconstruction = run_litevggt_reconstruction(
            input_dir=single_input_dir,
            checkpoint_path=checkpoint_path,
            keep_ratio=keep_ratio,
            max_points=max_points,
            max_input_frames=max_input_frames,
            target_size=target_size,
            frame_stride=frame_stride,
            depth_conf_thresh=depth_conf_thresh,
            preprocess_mode=preprocess_mode,
            progress=progress,
            selection_strategy=selection_strategy,
            axis_trim_low_quantile=axis_trim_low_quantile,
            axis_trim_high_quantile=axis_trim_high_quantile,
            spatial_keep_quantile=spatial_keep_quantile,
        )
        return write_litevggt_reconstruction_pointcloud(
            reconstruction,
            output_ply=output_ply,
            output_meta_json=output_meta_json,
            max_points=max_points,
            extra_metrics={
                "litevggt_inference_mode_requested": "windowed",
                "litevggt_inference_mode_effective": "single",
                "litevggt_window_count": 1,
                "litevggt_chunk_size": int(chunk_size),
                "litevggt_overlap": int(overlap),
                "loop_candidate_count": 0,
                "loop_edge_count": 0,
                "loop_optimization_enabled": False,
                "loop_optimization_residual": None,
                "loop_optimization_skipped_reason": "single_window",
                **keyframe_metrics,
            },
        )

    progress(
        "litevggt_preprocess",
        26,
        f"running LiteVGGT windowed preview on {len(files)} images, windows={len(windows)}, chunk={int(chunk_size)}, overlap={int(overlap)}",
    )
    window_root = output_ply.parent / "windows"
    if window_root.exists():
        shutil.rmtree(window_root)
    window_root.mkdir(parents=True, exist_ok=True)

    states: list[LiteVGGTWindowState] = []
    global_centers_by_frame: dict[int, np.ndarray] = {}
    processed_frames: set[int] = set()
    raw_point_count = 0
    valid_pixel_count = 0
    valid_image_pixel_count = 0
    point_count_after_filter = 0
    pose_frame_count = 0
    cuda_memory_peak_mb = 0.0
    window_alignment_strategies: list[str] = []
    window_alignment_residuals: list[float | None] = []
    window_alignment_inlier_ratios: list[float] = []
    window_alignment_fallback_count = 0
    window_alignment_rejected_count = 0
    model_cache_metrics: dict[str, Any] = {}
    point_selection_metric = str(selection_strategy or "global_confidence").strip().lower()
    quality = resolve_litevggt_quality_settings(len(files), {"target_size": target_size, "keep_ratio": keep_ratio})
    window_point_budget = max(1, int(np.ceil(max(1, int(max_points)) / max(1, len(windows)) * 1.25))) if max_points > 0 else 0

    for window_index, spec in enumerate(windows):
        progress_value = min(76, 28 + int(46 * (window_index / max(1, len(windows)))))
        progress(
            "litevggt_window",
            progress_value,
            f"running LiteVGGT window {window_index + 1}/{len(windows)} with {len(spec.files)} images",
        )
        window_dir = materialize_litevggt_window_dir(window_root, spec)
        reconstruction = run_litevggt_reconstruction(
            input_dir=window_dir,
            checkpoint_path=checkpoint_path,
            keep_ratio=keep_ratio,
            max_points=window_point_budget,
            max_input_frames=None,
            target_size=target_size,
            frame_stride=None,
            depth_conf_thresh=depth_conf_thresh,
            preprocess_mode=preprocess_mode,
            progress=progress,
            selection_strategy=selection_strategy,
            axis_trim_low_quantile=axis_trim_low_quantile,
            axis_trim_high_quantile=axis_trim_high_quantile,
            spatial_keep_quantile=spatial_keep_quantile,
        )
        mapped_frame_indices, point_frame_indices = map_reconstruction_frame_indices(reconstruction, spec)
        centers = camera_centers_from_w2c(reconstruction.w2c)
        center_count = min(int(centers.shape[0]), int(mapped_frame_indices.shape[0]))
        pose_frame_count += center_count
        center_frame_indices = mapped_frame_indices[:center_count]
        local_centers_by_frame = {
            int(frame_index): centers[index] for index, frame_index in enumerate(center_frame_indices)
        }
        alignment_points, alignment_frame_indices = sample_alignment_points(
            reconstruction.points,
            point_frame_indices,
            reconstruction.confidence,
            max_points=50000,
        )

        transform, alignment_metrics = align_litevggt_window(
            states,
            source_points=alignment_points,
            source_frame_indices=alignment_frame_indices,
            local_centers_by_frame=local_centers_by_frame,
            global_centers_by_frame=global_centers_by_frame,
        )
        window_alignment_strategies.append(str(alignment_metrics["strategy"]))
        window_alignment_residuals.append(alignment_metrics["residual"])
        window_alignment_inlier_ratios.append(float(alignment_metrics["inlier_ratio"]))
        if alignment_metrics["fallback"]:
            window_alignment_fallback_count += 1
        if alignment_metrics["rejected"]:
            window_alignment_rejected_count += 1

        keep_mask = ~np.isin(point_frame_indices, np.asarray(sorted(processed_frames), dtype=np.int32))
        if not np.any(keep_mask):
            keep_mask = np.ones(point_frame_indices.shape, dtype=bool)
        points = reconstruction.points[keep_mask]
        colors = reconstruction.colors[keep_mask]
        confidence = reconstruction.confidence[keep_mask]
        kept_point_frame_indices = point_frame_indices[keep_mask]
        voxel_keep = voxel_downsample_keep_indices(
            points,
            confidence,
            diag_ratio=window_voxel_diag_ratio,
        )
        points = points[voxel_keep]
        colors = colors[voxel_keep]
        confidence = confidence[voxel_keep]
        kept_point_frame_indices = kept_point_frame_indices[voxel_keep]
        states.append(
            LiteVGGTWindowState(
                spec=spec,
                transform=transform,
                points=points,
                colors=colors,
                confidence=confidence,
                point_frame_indices=kept_point_frame_indices,
                alignment_points=alignment_points,
                alignment_frame_indices=alignment_frame_indices,
                local_centers_by_frame=local_centers_by_frame,
            )
        )

        transformed_centers = apply_sim3(centers, transform)
        for index, frame_index in enumerate(center_frame_indices):
            global_centers_by_frame[int(frame_index)] = transformed_centers[index]
        processed_frames.update(int(frame_index) for frame_index in kept_point_frame_indices)

        raw_point_count += int(reconstruction.metrics.get("point_count_before_filter", reconstruction.points.shape[0]))
        valid_pixel_count += int(reconstruction.metrics.get("valid_pixel_count", 0))
        valid_image_pixel_count += int(reconstruction.metrics.get("valid_image_pixel_count", 0))
        point_count_after_filter += int(points.shape[0])
        cuda_memory_peak_mb = max(cuda_memory_peak_mb, float(reconstruction.metrics.get("cuda_memory_peak_mb", 0.0)))
        model_cache_metrics.update(
            {
                key: value
                for key, value in reconstruction.metrics.items()
                if key in {
                    "litevggt_cpu_model_cached",
                    "litevggt_gpu_loaded_from_cpu",
                    "litevggt_model_loaded_from_disk",
                    "litevggt_gpu_idle_unload_seconds",
                    "litevggt_gpu_model_loaded",
                }
            }
        )

    loop_candidates = find_litevggt_loop_candidates(states) if loop_closure else []
    loop_edges: list[LiteVGGTLoopEdge] = []
    if loop_candidates:
        progress("litevggt_loop_closure", 76, f"validating {len(loop_candidates)} LiteVGGT loop candidates")
        bridge_root = output_ply.parent / "loop_bridges"
        if bridge_root.exists():
            shutil.rmtree(bridge_root)
        bridge_root.mkdir(parents=True, exist_ok=True)
        for candidate in loop_candidates:
            edge = validate_litevggt_loop_edge(
                candidate=candidate,
                states=states,
                original_files=original_files,
                bridge_root=bridge_root,
                checkpoint_path=checkpoint_path,
                keep_ratio=keep_ratio,
                target_size=target_size,
                depth_conf_thresh=depth_conf_thresh,
                preprocess_mode=preprocess_mode,
                selection_strategy=selection_strategy,
                axis_trim_low_quantile=axis_trim_low_quantile,
                axis_trim_high_quantile=axis_trim_high_quantile,
                spatial_keep_quantile=spatial_keep_quantile,
                progress=progress,
            )
            if edge is not None:
                loop_edges.append(edge)

    loop_metrics = (
        optimize_litevggt_window_translations(states, loop_edges)
        if loop_closure
        else {"loop_optimization_residual": None, "loop_optimization_skipped_reason": "disabled"}
    )
    all_points = np.concatenate([apply_sim3(state.points, state.transform) for state in states], axis=0)
    all_colors = np.concatenate([state.colors for state in states], axis=0)
    all_confidence = np.concatenate([state.confidence for state in states], axis=0)
    all_points, all_colors, all_confidence = voxel_downsample_points(
        all_points,
        all_colors,
        all_confidence,
        diag_ratio=final_voxel_diag_ratio,
    )
    bounds = litevggt_preview_bounds(all_points)
    point_count = write_point_cloud_ply(
        all_points,
        all_colors,
        output_ply,
        confidence=all_confidence,
        max_points=max_points,
    )
    if output_meta_json is not None:
        write_litevggt_preview_meta_json(
            output_meta_json,
            point_count_raw=raw_point_count,
            point_count_exported=point_count,
            bounds=bounds,
            recommended_view=None,
        )

    return {
        "original_frame_count": int(original_frame_count),
        "input_frame_count": int(len(files)),
        "aligned_frame_count": int(len(files)),
        "litevggt_pose_frame_count": int(pose_frame_count),
        "skipped_frame_count": int(original_frame_count - len(files)),
        "frame_selection": "windowed_scene_coverage",
        **keyframe_metrics,
        "litevggt_frame_stride": int(frame_selection_result.frame_stride),
        "litevggt_frame_stride_source": frame_selection_result.frame_stride_source,
        "litevggt_frame_budget": int(frame_selection_result.frame_budget),
        "litevggt_first_frame_index": int(frame_indices[0]),
        "litevggt_last_frame_index": int(frame_indices[-1]),
        "point_selection_strategy": "global_confidence" if point_selection_metric in {"global", "global_confidence"} else "scene_coverage",
        "keep_ratio": float(quality.keep_ratio),
        "keep_ratio_source": quality.keep_ratio_source,
        "depth_conf_thresh": None if depth_conf_thresh is None else float(depth_conf_thresh),
        "spatial_keep_quantile": float(spatial_keep_quantile),
        "axis_trim_low_quantile": float(axis_trim_low_quantile),
        "axis_trim_high_quantile": float(axis_trim_high_quantile),
        "litevggt_preprocess_mode": str(preprocess_mode or "pad").strip().lower(),
        "max_points": int(max_points),
        "litevggt_target_size": int(quality.target_size),
        "litevggt_target_size_source": quality.target_size_source,
        "litevggt_quality_profile": quality.quality_profile,
        "litevggt_inference_mode": "windowed",
        "litevggt_inference_mode_requested": "windowed",
        "litevggt_inference_mode_effective": "windowed",
        "litevggt_window_count": int(len(windows)),
        "litevggt_chunk_size": int(chunk_size),
        "litevggt_overlap": int(overlap),
        "window_alignment_strategy": window_alignment_strategies,
        "window_alignment_residuals": window_alignment_residuals,
        "window_alignment_inlier_ratios": window_alignment_inlier_ratios,
        "window_alignment_fallback_count": int(window_alignment_fallback_count),
        "window_alignment_rejected_count": int(window_alignment_rejected_count),
        "valid_pixel_count": int(valid_pixel_count),
        "valid_image_pixel_count": int(valid_image_pixel_count),
        "point_count_before_filter": int(raw_point_count),
        "point_count_after_filter": int(point_count_after_filter),
        "point_count_before_downsample": int(point_count_after_filter),
        "point_count_after_downsample": int(all_points.shape[0]),
        "point_count_after_voxel_downsample": int(all_points.shape[0]),
        "cuda_memory_peak_mb": float(cuda_memory_peak_mb),
        "official_single_path": False,
        "loop_candidate_count": int(len(loop_candidates)),
        "loop_edge_count": int(len(loop_edges)),
        "loop_optimization_enabled": bool(loop_closure),
        **loop_metrics,
        **model_cache_metrics,
        "point_count": point_count,
        "point_count_raw": int(raw_point_count),
        "point_count_exported": point_count,
        "point_source": LITEVGGT_POINT_SOURCE,
        "litevggt_ply_format": "point_cloud",
        "litevggt_preview_point_radius": fixed_preview_radius(all_points),
        "bbox_min": bounds["bbox_min"],
        "bbox_max": bounds["bbox_max"],
        "bbox_center": bounds["bbox_center"],
        "bbox_radius": bounds["bbox_radius"],
    }


def write_litevggt_reconstruction_pointcloud(
    reconstruction: LiteVGGTReconstruction,
    *,
    output_ply: Path,
    output_meta_json: Path | None,
    max_points: int,
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bounds = litevggt_preview_bounds(reconstruction.points)
    point_count_raw = int(reconstruction.metrics.get("point_count_before_filter", reconstruction.points.shape[0]))
    point_count = write_point_cloud_ply(
        reconstruction.points,
        reconstruction.colors,
        output_ply,
        confidence=reconstruction.confidence,
        max_points=max_points,
    )
    if output_meta_json is not None:
        image_height = int(reconstruction.images.shape[1]) if reconstruction.images.ndim >= 3 else 0
        write_litevggt_preview_meta_json(
            output_meta_json,
            point_count_raw=point_count_raw,
            point_count_exported=point_count,
            bounds=bounds,
            recommended_view=litevggt_camera_view(
                reconstruction.w2c[0],
                reconstruction.intrinsics[0],
                radius=float(bounds["bbox_radius"]),
                image_height=image_height,
            ),
        )
    return {
        **reconstruction.metrics,
        **(extra_metrics or {}),
        "point_count": point_count,
        "point_count_raw": point_count_raw,
        "point_count_exported": point_count,
        "point_source": LITEVGGT_POINT_SOURCE,
        "litevggt_ply_format": "point_cloud",
        "litevggt_preview_point_radius": fixed_preview_radius(reconstruction.points),
        "bbox_min": bounds["bbox_min"],
        "bbox_max": bounds["bbox_max"],
        "bbox_center": bounds["bbox_center"],
        "bbox_radius": bounds["bbox_radius"],
    }


def run_litevggt_pointcloud(
    *,
    input_dir: Path,
    checkpoint_path: Path,
    output_ply: Path,
    output_meta_json: Path | None = None,
    keep_ratio: float | None,
    max_points: int,
    max_input_frames: int | None = None,
    target_size: int | None = None,
    frame_stride: int | None = None,
    depth_conf_thresh: float | None = None,
    preprocess_mode: str = "pad",
    progress: Progress,
    inference_mode: str = "auto",
    chunk_size: int = 64,
    overlap: int = 16,
    loop_closure: bool = True,
    scene_profile: str = "mixed_balanced",
    keyframe_target: int | None = None,
    min_frame_gap: int = 1,
    min_scene_change: float = 0.0,
    window_voxel_diag_ratio: float = 0.0,
    final_voxel_diag_ratio: float = 0.0,
    **unused_options,
) -> dict[str, Any]:
    requested_mode = str(inference_mode or "auto").strip().lower()
    if requested_mode not in {"auto", "single", "windowed"}:
        requested_mode = "auto"
    original_files = image_files(input_dir)
    original_frame_count = len(original_files)
    if max_input_frames is not None or frame_stride is not None:
        selected_count = len(
            resolve_litevggt_frame_selection(
                original_files,
                multiple=8,
                max_frames=max_input_frames,
                frame_stride=frame_stride,
            ).files
        )
    else:
        selected_count = original_frame_count
    if requested_mode == "windowed":
        return run_litevggt_pointcloud_windowed(
            input_dir=input_dir,
            checkpoint_path=checkpoint_path,
            output_ply=output_ply,
            output_meta_json=output_meta_json,
            keep_ratio=keep_ratio,
            max_points=max_points,
            max_input_frames=max_input_frames,
            target_size=target_size,
            frame_stride=frame_stride,
            depth_conf_thresh=depth_conf_thresh,
            preprocess_mode=preprocess_mode,
            progress=progress,
            chunk_size=chunk_size,
            overlap=overlap,
            loop_closure=loop_closure,
            scene_profile=scene_profile,
            keyframe_target=keyframe_target,
            min_frame_gap=min_frame_gap,
            min_scene_change=min_scene_change,
            window_voxel_diag_ratio=window_voxel_diag_ratio,
            final_voxel_diag_ratio=final_voxel_diag_ratio,
            **unused_options,
        )

    reconstruction = run_litevggt_reconstruction(
        input_dir=input_dir,
        checkpoint_path=checkpoint_path,
        keep_ratio=keep_ratio,
        max_points=max_points,
        max_input_frames=max_input_frames,
        target_size=target_size,
        frame_stride=frame_stride,
        depth_conf_thresh=depth_conf_thresh,
        preprocess_mode=preprocess_mode,
        progress=progress,
        **unused_options,
    )
    return write_litevggt_reconstruction_pointcloud(
        reconstruction,
        output_ply=output_ply,
        output_meta_json=output_meta_json,
        max_points=max_points,
        extra_metrics={
            "litevggt_inference_mode_requested": requested_mode,
            "litevggt_inference_mode_effective": "single",
            "litevggt_window_count": 1,
            "litevggt_chunk_size": int(chunk_size),
            "litevggt_overlap": int(overlap),
            "frame_selection_strategy": "all_frames",
            "selected_keyframe_count": int(selected_count),
            "loop_candidate_count": 0,
            "loop_edge_count": 0,
            "loop_optimization_enabled": False,
            "loop_optimization_residual": None,
            "loop_optimization_skipped_reason": "single_path",
        },
    )


def require_transformer_engine() -> None:
    try:
        __import__("transformer_engine")
    except Exception as exc:  # pragma: no cover - depends on CUDA worker image
        raise PreviewFailure("TRANSFORMER_ENGINE_UNAVAILABLE", f"LiteVGGT requires transformer_engine: {exc}") from exc
